/*
 * v1cld_solver_parallel.cpp — Copie de v1cld_solver.cpp avec la boucle (B)
 * (θ_{i,j}/ζ_{i,j}) parallélisée par OpenMP (#pragma omp parallel for,
 * miroir exact du pattern déjà validé dans v1ld_solver.cpp:101).
 *
 * Voir v1cld_solver_parallel.hpp pour le contexte (pourquoi une copie
 * séparée plutôt qu'une modification en place de v1cld_solver.cpp).
 *
 * Seul changement fonctionnel vs v1cld_solver.cpp : `cost_ij` est déclaré À
 * L'INTÉRIEUR de la boucle `for (int i ...)` (une instance par itération de
 * i, thread-privé) au lieu d'être un buffer unique réutilisé à travers toutes
 * les itérations — nécessaire pour être correct sous OpenMP (sinon plusieurs
 * threads écrivent dans le même buffer). Chaque itération i n'écrit que dans
 * theta_vals[i]/delta_ij[i]/x_theta[i]/x_zeta[i] : pas d'aliasing entre
 * threads, donc sûr (vérifié : SSPCLDProblem::solve_theta/solve_zeta sont
 * const, sans état partagé/statique mutable, cf. solver_generic/ssp/ssp_cld_problem.hpp).
 *
 * Pour la docstring complète de l'algorithme (structure d'une itération A-H),
 * voir l'en-tête de v1cld_solver.cpp — identique ici.
 */

#include "v1cld_solver_parallel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <tuple>

// ── Knapsack glouton (copie locale, linkage interne — pas de collision avec
//    la fonction homonyme de v1cld_solver.cpp) ──────────────────────────────
// min Σ_i p_i z_i Δ_i  s.c.  Σ_i p_i z_i ≥ 1−ε,  z ∈ {0,1}^N
static std::pair<Vec1d, double>
knapsack_min_parallel(const Vec1d& delta, const Vec1d& p, double eps)
{
    int N = static_cast<int>(delta.size());
    std::vector<int> order(N);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b){ return delta[a] < delta[b]; });

    Vec1d  z(N, 0.0);
    double cumsum = 0.0, val = 0.0;
    for (int idx : order) {
        z[idx]  = 1.0;
        cumsum += p[idx];
        val    += p[idx] * delta[idx];
        if (cumsum >= 1.0 - eps - 1e-9) break;
    }
    return {z, val};
}

// ── Solveur générique v1CLD (variante parallèle) ─────────────────────────────
V1CLDResult solve_v1cld_parallel(
    V1CLDProblem& problem,
    int T, int log_interval, bool verbose,
    double time_limit_s, double rel_tol, int tol_window,
    double v_lower, const Vec3d& lambda_init)
{
    const int    d   = problem.n_vars();
    const int    N   = problem.N_scen();
    const int    m   = problem.n_constraints();
    const double eps = problem.epsilon();
    const Vec1d& p   = problem.p();
    const Vec1d& c   = problem.c();
    const double inv_m = 1.0 / m;

    // λ[i][j] ∈ R^d, initialisé à zéro si non fourni
    Vec3d lam = lambda_init.empty()
                ? Vec3d(N, Vec2d(m, Vec1d(d, 0.0)))
                : lambda_init;

    double best_upper        = +1e18;
    double best_at_tol_check = +1e18;
    Vec3d  best_lam          = lam;

    Vec1d convergence, times_vec, gnorm_vec, zchanges_vec;
    convergence.reserve(std::min(T, 1 << 20));
    times_vec.reserve(convergence.capacity());
    gnorm_vec.reserve(convergence.capacity());
    zchanges_vec.reserve(convergence.capacity());

    Vec1d prev_y(N, 0.0);
    auto  t0 = std::chrono::high_resolution_clock::now();
    int   iters_run = 0;
    double t_zeta_theta_accum_s = 0.0;  // cf. V1CLDResult::t_zeta_theta_s

    // Buffers réutilisés à chaque itération (mêmes commentaires que
    // v1cld_solver.cpp sur le centrage par scénario — inchangé ici).
    Vec2d Lambda_sum_i(N, Vec1d(d));

    // θ_{i,j}, δ_{i,j}, x_{i,j} et x_{i,j}^ζ (N × m × d)
    Vec2d theta_vals(N, Vec1d(m, 0.0));
    Vec2d delta_ij(N,   Vec1d(m, 0.0));
    Vec3d x_theta(N,    Vec2d(m, Vec1d(d, 0.0)));
    Vec3d x_zeta(N,     Vec2d(m, Vec1d(d, 0.0)));

    Vec1d effective_delta(N, 0.0);
    Vec1d x_bar(d);
    Vec1d y_star(N);

    for (int t = 0; t < T; ++t) {

        // ── (A) Λ_sum_i = Σ_j λ_{i,j}  (PAR SCENARIO, pas global) ────────────
        for (int i = 0; i < N; ++i) {
            std::fill(Lambda_sum_i[i].begin(), Lambda_sum_i[i].end(), 0.0);
            for (int j = 0; j < m; ++j)
                for (int k = 0; k < d; ++k)
                    Lambda_sum_i[i][k] += lam[i][j][k];
        }

        // ── (B) Sous-problèmes θ_{i,j} et ζ_{i,j} ───────────────────────────
        // Indépendant par scénario i : parallélisable. cost_ij est déclaré DANS
        // la boucle (une instance par itération de i, pas de buffer partagé)
        // pour rester correct sous #pragma omp parallel for (même pattern que
        // v1ld_solver.cpp:97-114).
        auto t_zeta_theta_t0 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < N; ++i) {
            Vec1d cost_ij(d);
            const double p_ij = p[i] * inv_m;
            for (int j = 0; j < m; ++j) {

                // cost_{i,j} = c + λ_{i,j}/p_{i,j} − Λ_sum_i/p_i  (centrage par scenario)
                for (int k = 0; k < d; ++k)
                    cost_ij[k] = c[k] + lam[i][j][k] / p_ij - Lambda_sum_i[i][k] / p[i];

                // θ_{i,j} : max sur le domaine de base, sans contrainte j
                auto [vt, xt] = problem.solve_theta(j, cost_ij);
                theta_vals[i][j] = vt;
                x_theta[i][j]    = std::move(xt);

                // ζ_{i,j} : max sur le domaine de base, avec contrainte j du scénario i
                auto [vz, xz] = problem.solve_zeta(i, j, cost_ij);
                // δ_{i,j} = profit perdu en imposant la contrainte j
                delta_ij[i][j] = std::max(0.0, vt - vz);
                x_zeta[i][j]   = std::move(xz);
            }
        }
        t_zeta_theta_accum_s += std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t_zeta_theta_t0).count();

        // ── (C) Pénalité agrégée par scénario + knapsack glouton ─────────────
        // Δ_i = (1/m) Σ_j δ_{i,j}
        for (int i = 0; i < N; ++i) {
            double sum = 0.0;
            for (int j = 0; j < m; ++j) sum += delta_ij[i][j];
            effective_delta[i] = inv_m * sum;
        }
        double knapsack_val;
        std::tie(y_star, knapsack_val) = knapsack_min_parallel(effective_delta, p, eps);

        // ── (D) L1 = Σ_{i,j} p_{i,j} θ_{i,j} − val ─────────────────────────
        double L1 = -knapsack_val;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < m; ++j)
                L1 += p[i] * inv_m * theta_vals[i][j];

        if (L1 < best_upper) { best_upper = L1; best_lam = lam; }
        iters_run = t + 1;

        // ── (E) x̄ = Σ_{i,j} p_{i,j} x_{i,j}* ──────────────────────────────
        std::fill(x_bar.begin(), x_bar.end(), 0.0);
        for (int i = 0; i < N; ++i) {
            const bool active = (y_star[i] > 0.5);
            for (int j = 0; j < m; ++j) {
                const Vec1d& xij = active ? x_zeta[i][j] : x_theta[i][j];
                for (int k = 0; k < d; ++k)
                    x_bar[k] += p[i] * inv_m * xij[k];
            }
        }

        // ── (F) Sous-gradient g_{i,j} = x_{i,j}* − x̄ ; calcul de ||g||² ────
        double gnorm_sq = 0.0;
        for (int i = 0; i < N; ++i) {
            const bool active = (y_star[i] > 0.5);
            for (int j = 0; j < m; ++j) {
                const Vec1d& xij = active ? x_zeta[i][j] : x_theta[i][j];
                for (int k = 0; k < d; ++k) {
                    double g = xij[k] - x_bar[k];
                    gnorm_sq += g * g;
                }
            }
        }

        // ── (G) Logging + critères d'arrêt ───────────────────────────────────
        if (log_interval > 0 && t % log_interval == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            convergence.push_back(L1);
            times_vec.push_back(std::chrono::duration<double>(now - t0).count());
            gnorm_vec.push_back(std::sqrt(gnorm_sq));
            double zchg = 0.0;
            for (int i = 0; i < N; ++i)
                if (std::abs(y_star[i] - prev_y[i]) > 0.5) zchg += 1.0;
            zchanges_vec.push_back(zchg);
            std::copy(y_star.begin(), y_star.end(), prev_y.begin());
        }

        if (verbose && t % 50 == 0)
            std::cout << "  iter " << t
                      << " | L1="   << L1
                      << "  best="  << best_upper
                      << "  ||g||=" << std::sqrt(gnorm_sq) << "\n";

        if (time_limit_s > 0.0 && t % 50 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            if (std::chrono::duration<double>(now - t0).count() >= time_limit_s) break;
        }

        if (rel_tol > 0.0 && tol_window > 0 && t > 0 && t % tol_window == 0) {
            double impr = best_at_tol_check - best_upper;
            if (impr / std::max(std::abs(best_upper), 1e-10) < rel_tol) break;
            best_at_tol_check = best_upper;
        }

        // ── (H) Pas de descente + mise à jour λ_{i,j} ────────────────────────
        double step;
        if (v_lower > -1e17 && gnorm_sq > 1e-14) {
            double gap = L1 - v_lower;
            step = (gap > 0.0) ? gap / gnorm_sq
                               : 1.0 / std::sqrt(static_cast<double>(t + 1));
        } else {
            step = 1.0 / std::sqrt(static_cast<double>(t + 1));
        }

        for (int i = 0; i < N; ++i) {
            const bool active = (y_star[i] > 0.5);
            for (int j = 0; j < m; ++j) {
                const Vec1d& xij = active ? x_zeta[i][j] : x_theta[i][j];
                for (int k = 0; k < d; ++k)
                    lam[i][j][k] += step * (x_bar[k] - xij[k]);
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    V1CLDResult res;
    res.v1CLD_star     = best_upper;
    res.lambda_star    = std::move(best_lam);
    res.convergence    = std::move(convergence);
    res.times_s        = std::move(times_vec);
    res.gnorm_history  = std::move(gnorm_vec);
    res.z_changes      = std::move(zchanges_vec);
    res.solve_time_s   = std::chrono::duration<double>(t1 - t0).count();
    res.iterations_run = iters_run;
    res.t_zeta_theta_s = t_zeta_theta_accum_s;
    return res;
}
