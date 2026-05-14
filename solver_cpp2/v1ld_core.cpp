/*
 * v1ld_core.cpp — Boucle sous-gradient pour v1LD.
 * Adapté strictement de solver/v1ld_solver_general.py.
 */

#include "v1ld_core.hpp"
#include "subproblems.hpp"

#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <deque>

// ── Évaluation ponctuelle de L1(λ) et son sous-gradient ─────────────────────
// N'effectue aucune mise à jour : utile pour le bundle method Python.
EvalResult evaluate_L1(
    const Vec1d& c,
    const Vec1d& p,
    const Vec3d& A,
    const Vec2d& b,
    double       epsilon,
    const Vec2d& lam
) {
    int N = static_cast<int>(p.size());
    int n = static_cast<int>(c.size());

    Vec1d Lambda_sum(n, 0.0);
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < n; ++k)
            Lambda_sum[k] += lam[i][k];

    Vec1d theta_vals(N), zeta_vals(N);
    Vec2d x_theta(N, Vec1d(n)), x_zeta(N, Vec1d(n));

    for (int i = 0; i < N; ++i) {
        Vec1d cost_i(n);
        for (int k = 0; k < n; ++k)
            cost_i[k] = c[k] + lam[i][k] / p[i] - Lambda_sum[k];

        auto [vt, xt] = solve_theta_i(cost_i);
        theta_vals[i] = vt;  x_theta[i] = std::move(xt);

        auto [vz, xz] = solve_zeta_i_exact(cost_i, A[i], b[i]);
        zeta_vals[i] = vz;  x_zeta[i] = std::move(xz);
    }

    Vec1d delta(N);
    for (int i = 0; i < N; ++i)
        delta[i] = std::max(0.0, zeta_vals[i] - theta_vals[i]);

    auto [z_star, knapsack_val] = solve_z_knapsack_min(delta, p, epsilon);

    double L1 = knapsack_val;
    for (int i = 0; i < N; ++i) L1 += p[i] * theta_vals[i];

    // x̄ = Σ p_i x_i*
    Vec1d x_bar(n, 0.0);
    for (int i = 0; i < N; ++i) {
        const Vec1d& xi = (z_star[i] > 0.5) ? x_zeta[i] : x_theta[i];
        for (int k = 0; k < n; ++k) x_bar[k] += p[i] * xi[k];
    }

    // Sous-gradient g[i][k] = x_i*[k] - x̄[k]
    Vec2d g(N, Vec1d(n));
    for (int i = 0; i < N; ++i) {
        const Vec1d& xi = (z_star[i] > 0.5) ? x_zeta[i] : x_theta[i];
        for (int k = 0; k < n; ++k)
            g[i][k] = xi[k] - x_bar[k];
    }

    // Rassembler x_i* (pour l'inspection côté Python)
    Vec2d x_scenarios(N, Vec1d(n));
    for (int i = 0; i < N; ++i) {
        const Vec1d& xi = (z_star[i] > 0.5) ? x_zeta[i] : x_theta[i];
        x_scenarios[i] = xi;
    }

    return {L1, g, x_bar, x_scenarios, z_star};
}

SolverResult solve_impl(
    const Vec1d& c,
    const Vec1d& p,
    const Vec3d& A,
    const Vec2d& b,
    double       epsilon,
    int          T,
    bool         decomp,
    bool         verbose,
    double       time_limit_s,
    int          log_interval,
    double       rel_tol,
    int          tol_window,
    const Vec2d& lambda_init
) {
    int N = static_cast<int>(p.size());
    int n = static_cast<int>(c.size());
    int m = static_cast<int>(b[0].size());

    // Initialisation λ ∈ R^{N×n} et μ ∈ R^{N×m×n} (si decomp)
    Vec2d lam = (!lambda_init.empty()) ? lambda_init : Vec2d(N, Vec1d(n, 0.0));
    Vec3d mu;
    if (decomp) mu.assign(N, Vec2d(m, Vec1d(n, 0.0)));

    double best_lower          = -1e18;
    double best_at_tol_check   = -1e18;
    Vec2d  best_lam            = lam;
    Vec3d  best_mu;
    if (decomp) best_mu = mu;

    Vec1d convergence, times_vec, gnorm_vec, zchanges_vec;
    convergence.reserve(std::min(T, 1 << 20));
    times_vec.reserve(std::min(T, 1 << 20));
    gnorm_vec.reserve(std::min(T, 1 << 20));
    zchanges_vec.reserve(std::min(T, 1 << 20));

    Vec1d prev_z(N, 0.0);  // z* de la fenêtre précédente (pour compter les changements)

    // Buffer circulaire pour les snapshots aux multiples de 5
    struct Snapshot { Vec2d g; Vec1d xbar; Vec1d zstar; int iter; };
    std::deque<Snapshot> snap_buf;

    auto t0 = std::chrono::high_resolution_clock::now();

    int iters_run = 0;

    for (int t = 0; t < T; ++t) {

        // ── (A) Lambda_sum = Σ_i λ_i ──────────────────────────────────────
        Vec1d Lambda_sum(n, 0.0);
        for (int i = 0; i < N; ++i)
            for (int k = 0; k < n; ++k)
                Lambda_sum[k] += lam[i][k];

        Vec1d theta_vals(N), zeta_vals(N);
        Vec2d x_theta(N, Vec1d(n)), x_zeta(N, Vec1d(n));
        Vec3d x_by_j_all;
        if (decomp) x_by_j_all.assign(N, Vec2d(m, Vec1d(n)));

        for (int i = 0; i < N; ++i) {
            // cost_i = c + λ_i / p_i − Λ
            Vec1d cost_i(n);
            for (int k = 0; k < n; ++k)
                cost_i[k] = c[k] + lam[i][k] / p[i] - Lambda_sum[k];

            {
                auto [v, x] = solve_theta_i(cost_i);
                theta_vals[i] = v;
                x_theta[i]    = std::move(x);
            }

            if (decomp) {
                auto [v, x_bar, x_by_j] = solve_zeta_i_general(
                    cost_i, A[i], b[i], mu[i]);
                zeta_vals[i]    = v;
                x_zeta[i]       = std::move(x_bar);
                x_by_j_all[i]   = std::move(x_by_j);
            } else {
                auto [v, x] = solve_zeta_i_exact(cost_i, A[i], b[i]);
                zeta_vals[i] = v;
                x_zeta[i]    = std::move(x);
            }
        }

        // ── (B) δ_i = max(0, ζ_i − θ_i) ; knapsack sur z ─────────────────
        Vec1d delta(N);
        for (int i = 0; i < N; ++i)
            delta[i] = std::max(0.0, zeta_vals[i] - theta_vals[i]);

        auto [z_star, knapsack_val] = solve_z_knapsack_min(delta, p, epsilon);

        // ── (C) Borne L1 ───────────────────────────────────────────────────
        double L1 = knapsack_val;
        for (int i = 0; i < N; ++i) L1 += p[i] * theta_vals[i];

        if (L1 > best_lower) {
            best_lower = L1;
            best_lam   = lam;
            if (decomp) best_mu = mu;
        }

        iters_run = t + 1;

        // ── (D) Sous-gradient sur λ (calcul x_bar_outer) ──────────────────
        Vec1d x_bar_outer(n, 0.0);
        for (int i = 0; i < N; ++i) {
            const Vec1d& xi = (z_star[i] > 0.5) ? x_zeta[i] : x_theta[i];
            for (int k = 0; k < n; ++k) x_bar_outer[k] += p[i] * xi[k];
        }

        // ── Snapshot aux multiples de 5 ──────────────────────────────────────
        if (t % 5 == 0) {
            Vec2d g_snap(N, Vec1d(n));
            for (int i = 0; i < N; ++i) {
                const Vec1d& xi = (z_star[i] > 0.5) ? x_zeta[i] : x_theta[i];
                for (int k = 0; k < n; ++k)
                    g_snap[i][k] = xi[k] - x_bar_outer[k];
            }
            snap_buf.push_back({std::move(g_snap), x_bar_outer, z_star, t});
            if (snap_buf.size() > 5) snap_buf.pop_front();
        }

        // ── Calcul de gnorm² (pour le logging) ───────────────────────────────
        bool need_gnorm = (log_interval > 0 && t % log_interval == 0)
                       || (verbose && (t % 50 == 0 || t == T - 1));
        double gnorm_sq = 0.0;
        if (need_gnorm) {
            for (int i = 0; i < N; ++i) {
                const Vec1d& xi = (z_star[i] > 0.5) ? x_zeta[i] : x_theta[i];
                for (int k = 0; k < n; ++k) {
                    double g = xi[k] - x_bar_outer[k];
                    gnorm_sq += g * g;
                }
            }
        }

        // ── Pas de sous-gradient 1/√t ─────────────────────────────────────
        double step = 1.0 / std::sqrt(static_cast<double>(t + 1));

        // ── Enregistrement ────────────────────────────────────────────────
        if (log_interval > 0 && t % log_interval == 0) {
            auto   now     = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - t0).count();
            convergence.push_back(L1);
            times_vec.push_back(elapsed);
            gnorm_vec.push_back(std::sqrt(gnorm_sq));

            double zchg = 0.0;
            for (int i = 0; i < N; ++i)
                if (std::abs(z_star[i] - prev_z[i]) > 0.5) zchg += 1.0;
            zchanges_vec.push_back(zchg);
            std::copy(z_star.begin(), z_star.end(), prev_z.begin());
        }

        // ── Critère d'arrêt : limite de temps ─────────────────────────────
        if (time_limit_s > 0.0 && t % 50 == 0) {
            auto   now     = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - t0).count();
            if (elapsed >= time_limit_s) break;
        }

        // ── Critère d'arrêt : amélioration relative ────────────────────────
        if (rel_tol > 0.0 && tol_window > 0 && t > 0 && t % tol_window == 0) {
            double denom   = std::max(std::abs(best_lower), 1e-10);
            double rel_imp = (best_lower - best_at_tol_check) / denom;
            if (rel_imp < rel_tol) break;
            best_at_tol_check = best_lower;
        }

        // ── Application du pas sur λ ──────────────────────────────────────
        for (int i = 0; i < N; ++i) {
            const Vec1d& xi = (z_star[i] > 0.5) ? x_zeta[i] : x_theta[i];
            for (int k = 0; k < n; ++k)
                lam[i][k] += step * (xi[k] - x_bar_outer[k]);
        }

        // ── (E) Sous-gradient sur μ (decomp uniquement) ────────────────────
        if (decomp) {
            for (int i = 0; i < N; ++i) {
                if (z_star[i] <= 0.5) continue;
                Vec1d x_bar_i(n, 0.0);
                for (int j = 0; j < m; ++j)
                    for (int k = 0; k < n; ++k)
                        x_bar_i[k] += x_by_j_all[i][j][k];
                for (int k = 0; k < n; ++k) x_bar_i[k] /= m;

                for (int j = 0; j < m; ++j)
                    for (int k = 0; k < n; ++k)
                        mu[i][j][k] += step * (x_by_j_all[i][j][k] - x_bar_i[k]) / m;
            }
        }

        if (verbose && (t % 50 == 0 || t == T - 1)) {
            std::cout << "  iter " << t
                      << " | L1=" << L1
                      << "  best=" << best_lower
                      << "  step=" << step
                      << "  ||g_λ||=" << std::sqrt(gnorm_sq) << "\n";
        }
    }

    auto   t1      = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    // Convertir le buffer de snapshots en champs du résultat (ordre ancienne → récente)
    Vec3d  g_history;
    Vec2d  xbar_outer_history;
    Vec2d  zstar_history;
    Vec1d  iter_history;
    for (auto& s : snap_buf) {
        g_history.push_back(s.g);
        xbar_outer_history.push_back(s.xbar);
        zstar_history.push_back(s.zstar);
        iter_history.push_back(static_cast<double>(s.iter));
    }

    SolverResult res;
    res.lambda_star          = best_lam;
    res.mu_star              = best_mu;
    res.v1LD_star            = best_lower;
    res.convergence          = convergence;
    res.times_s              = times_vec;
    res.gnorm_history        = gnorm_vec;
    res.z_changes            = zchanges_vec;
    res.solve_time_s         = elapsed;
    res.iterations_run       = iters_run;
    res.g_history            = std::move(g_history);
    res.xbar_outer_history   = std::move(xbar_outer_history);
    res.zstar_history        = std::move(zstar_history);
    res.iter_history         = std::move(iter_history);
    return res;
}

// ── solve_ahmed : descente de coordonnées multi-départ + recherche ternaire ───
//
// Pour chaque coordonnée λ_{i,k}, on maximise L1 en 1D par recherche ternaire
// (correct car L1 est concave). Un seul départ peut rester bloqué dans un plat
// non-lisse ; on lance donc n_restarts départs aléatoires (+ optionnellement
// lambda_init + λ=0) pour couvrir l'espace.
//
// La recherche ternaire depuis le centre courant avec rayon search_radius
// garantit : |t_opt - t_returned| ≤ search_radius * (2/3)^T_search.
// Pour search_radius=50, T_search=60 : précision ≈ 10^{-11}.
#include <random>
SolverResult solve_ahmed(
    const Vec1d& c,
    const Vec1d& p,
    const Vec3d& A,
    const Vec2d& b,
    double       epsilon,
    int          T_outer,
    int          T_search,
    double       search_radius,
    int          n_restarts,
    double       init_scale,
    bool         verbose,
    double       time_limit_s,
    const Vec2d& lambda_init,
    unsigned int seed
) {
    int N = static_cast<int>(p.size());
    int n = static_cast<int>(c.size());

    double best_L1  = -1e18;
    Vec2d  best_lam(N, Vec1d(n, 0.0));

    Vec1d conv_history;
    Vec1d time_history;

    auto t0 = std::chrono::high_resolution_clock::now();

    // ── Générateur de départs aléatoires ────────────────────────────────────
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, init_scale);

    // Construit la liste des points de départ :
    //   • λ = 0 toujours inclus
    //   • lambda_init si fourni
    //   • n_restarts tirages aléatoires ∼ N(0, init_scale)
    std::vector<Vec2d> starts;
    starts.push_back(Vec2d(N, Vec1d(n, 0.0)));          // départ λ=0
    if (!lambda_init.empty()) starts.push_back(lambda_init);
    for (int r = 0; r < n_restarts; ++r) {
        Vec2d lam_r(N, Vec1d(n));
        for (int i = 0; i < N; ++i)
            for (int k = 0; k < n; ++k)
                lam_r[i][k] = dist(rng);
        starts.push_back(std::move(lam_r));
    }

    // ── Boucle sur les départs ───────────────────────────────────────────────
    for (int r = 0; r < static_cast<int>(starts.size()); ++r) {
        Vec2d lam = starts[r];

        if (verbose) {
            std::cout << "  Ahmed restart=" << r
                      << " (λ0=" << (r == 0 ? "zeros" : (r == 1 && !lambda_init.empty() ? "init" : "random"))
                      << ")\n";
        }

        // Wrapper : évalue L1 en substituant lam[i0][k0] = t temporairement.
        auto eval_coord = [&](int i0, int k0, double t) -> double {
            double saved = lam[i0][k0];
            lam[i0][k0]  = t;
            double val   = evaluate_L1(c, p, A, b, epsilon, lam).L1_val;
            lam[i0][k0]  = saved;
            return val;
        };

        // ── T_outer cycles de descente de coordonnées ────────────────────────
        for (int outer = 0; outer < T_outer; ++outer) {

            for (int i = 0; i < N; ++i) {
                for (int k = 0; k < n; ++k) {
                    double center = lam[i][k];
                    double lo     = center - search_radius;
                    double hi     = center + search_radius;

                    for (int s = 0; s < T_search; ++s) {
                        double m1 = lo + (hi - lo) / 3.0;
                        double m2 = hi - (hi - lo) / 3.0;
                        if (eval_coord(i, k, m1) < eval_coord(i, k, m2))
                            lo = m1;
                        else
                            hi = m2;
                    }
                    lam[i][k] = 0.5 * (lo + hi);
                }
            }

            // Enregistrement après chaque cycle.
            double L1_cycle = evaluate_L1(c, p, A, b, epsilon, lam).L1_val;
            if (L1_cycle > best_L1) {
                best_L1  = L1_cycle;
                best_lam = lam;
            }

            auto   now     = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - t0).count();
            conv_history.push_back(best_L1);
            time_history.push_back(elapsed);

            if (verbose) {
                std::cout << "    outer=" << outer
                          << " L1=" << L1_cycle
                          << "  best=" << best_L1
                          << "  t=" << elapsed << "s\n";
            }

            if (time_limit_s > 0.0 && elapsed >= time_limit_s) goto done;
        }
    }

done:
    auto   t1      = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    SolverResult res;
    res.v1LD_star      = best_L1;
    res.lambda_star    = best_lam;
    res.convergence    = conv_history;
    res.times_s        = time_history;
    res.solve_time_s   = elapsed;
    res.iterations_run = static_cast<int>(conv_history.size());
    return res;
}
