/*
 * v1ld_solver.cpp — Corps générique de l'algorithme v1LD.
 *
 * Identique à v1ld_mkp_core.cpp mais paramétrisé via V1LDProblem :
 * aucune connaissance de MKP ou SSP ici — uniquement solve_theta et solve_zeta.
 *
 * Structure par itération :
 *   (A) coûts lagrangiens : cost_i = c + λ_i/p_i − Λ_sum
 *   (B) θ_i et ζ_i via V1LDProblem::solve_theta / solve_zeta
 *   (C) knapsack glouton sur y
 *   (D) L1 = Σ p_i θ_i − knapsack_val
 *   (E) x̄ = Σ_i p_i x_i*
 *   (F) sous-gradient g_i = x_i* − x̄
 *   (G) critères d'arrêt
 *   (H) pas de descente + mise à jour λ_i += step · (x̄ − x_i*)
 */

#include "v1ld_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <tuple>

// ── Knapsack glouton ──────────────────────────────────────────────────────────
// min Σ p_i z_i δ_i  s.c.  Σ p_i z_i ≥ 1−ε,  z ∈ {0,1}^N
static std::pair<Vec1d, double>
knapsack_min(const Vec1d& delta, const Vec1d& p, double eps)
{
    int N = static_cast<int>(delta.size());
    std::vector<int> order(N);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b){ return delta[a] < delta[b]; });

    Vec1d z(N, 0.0);
    double cumsum = 0.0, val = 0.0;
    for (int idx : order) {
        z[idx]  = 1.0;
        cumsum += p[idx];
        val    += p[idx] * delta[idx];
        if (cumsum >= 1.0 - eps - 1e-9) break;
    }
    return {z, val};
}

// ── Solveur générique ─────────────────────────────────────────────────────────
V1LDResult solve_v1ld(
    V1LDProblem& problem,
    int T, int log_interval, bool verbose,
    double time_limit_s, double rel_tol, int tol_window,
    double v_lower, const Vec2d& lambda_init,
    double abs_tol, int abs_patience)
{
    int         d   = problem.n_vars();
    int         N   = problem.N_scen();
    double      eps = problem.epsilon();
    const Vec1d& p  = problem.p();
    const Vec1d& c  = problem.c();

    Vec2d lam = lambda_init.empty() ? Vec2d(N, Vec1d(d, 0.0)) : lambda_init;
    double best_upper        = +1e18;
    double best_at_tol_check = +1e18;
    Vec2d  best_lam          = lam;

    Vec1d convergence, times_vec, gnorm_vec, zchanges_vec;
    convergence.reserve(std::min(T, 1 << 20));
    times_vec.reserve(convergence.capacity());
    gnorm_vec.reserve(convergence.capacity());
    zchanges_vec.reserve(convergence.capacity());

    // État pour abs_tol / abs_patience
    double best_at_patience_start = +1e18;
    int    stale_iters             = 0;

    Vec1d prev_y(N, 0.0);
    auto  t0 = std::chrono::high_resolution_clock::now();
    int   iters_run = 0;

    // Buffers réutilisés à chaque itération
    Vec1d Lambda_sum(d);
    Vec1d theta_vals(N), delta_vals(N);
    Vec2d x_theta(N, Vec1d(d)), x_zeta(N, Vec1d(d));
    Vec1d x_bar(d), cost_i(d);
    Vec1d y_star(N);

    for (int t = 0; t < T; ++t) {

        // ── (A) Λ_sum = Σ_i λ_i ─────────────────────────────────────────
        std::fill(Lambda_sum.begin(), Lambda_sum.end(), 0.0);
        for (int i = 0; i < N; ++i)
            for (int k = 0; k < d; ++k)
                Lambda_sum[k] += lam[i][k];

        // ── (B) θ_i et ζ_i ──────────────────────────────────────────────────
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < d; ++k)
                cost_i[k] = c[k] + lam[i][k]/p[i] - Lambda_sum[k];

            auto [vt, xt] = problem.solve_theta(cost_i);
            theta_vals[i] = vt;
            x_theta[i]    = std::move(xt);

            auto [vz, xz] = problem.solve_zeta(i, cost_i);
            delta_vals[i] = std::max(0.0, vt - vz);
            x_zeta[i]     = std::move(xz);
        }

        // ── (C) Knapsack sur y ───────────────────────────────────────────────
        double knapsack_val;
        std::tie(y_star, knapsack_val) = knapsack_min(delta_vals, p, eps);

        // ── (D) L1 ──────────────────────────────────────────────────────────
        double L1 = -knapsack_val;
        for (int i = 0; i < N; ++i) L1 += p[i] * theta_vals[i];
        if (L1 < best_upper) { best_upper = L1; best_lam = lam; }
        iters_run = t + 1;

        // ── (E) x̄ = Σ_i p_i x_i* ───────────────────────────────────────────
        std::fill(x_bar.begin(), x_bar.end(), 0.0);
        for (int i = 0; i < N; ++i) {
            const Vec1d& xi = (y_star[i] > 0.5) ? x_zeta[i] : x_theta[i];
            for (int k = 0; k < d; ++k)
                x_bar[k] += p[i] * xi[k];
        }

        // ── (F) Sous-gradient et norme ───────────────────────────────────────
        double gnorm_sq = 0.0;
        for (int i = 0; i < N; ++i) {
            const Vec1d& xi = (y_star[i] > 0.5) ? x_zeta[i] : x_theta[i];
            for (int k = 0; k < d; ++k) {
                double g = xi[k] - x_bar[k];
                gnorm_sq += g * g;
            }
        }

        // ── (G) Logging et critères d'arrêt ─────────────────────────────────
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
            std::cout << "  iter " << t << " | L1=" << L1
                      << "  best=" << best_upper << "\n";

        if (time_limit_s > 0.0 && t % 50 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            if (std::chrono::duration<double>(now - t0).count() >= time_limit_s) break;
        }
        if (rel_tol > 0.0 && tol_window > 0 && t > 0 && t % tol_window == 0) {
            double impr = best_at_tol_check - best_upper;
            if (impr / std::max(std::abs(best_upper), 1e-10) < rel_tol) break;
            best_at_tol_check = best_upper;
        }
        // Critère patience absolu : arrêt si best_upper ne s'améliore pas de abs_tol
        // pendant abs_patience itérations consécutives.
        if (abs_tol > 0.0 && abs_patience > 0) {
            if (best_at_patience_start - best_upper >= abs_tol) {
                best_at_patience_start = best_upper;
                stale_iters            = 0;
            } else if (++stale_iters >= abs_patience) {
                break;
            }
        }

        // ── (H) Pas de descente + mise à jour λ ─────────────────────────────
        double step;
        if (v_lower > -1e17 && gnorm_sq > 1e-14) {
            double gap = L1 - v_lower;
            step = (gap > 0.0) ? gap / gnorm_sq
                               : 1.0 / std::sqrt(static_cast<double>(t + 1));
        } else {
            step = 1.0 / std::sqrt(static_cast<double>(t + 1));
        }

        for (int i = 0; i < N; ++i) {
            const Vec1d& xi = (y_star[i] > 0.5) ? x_zeta[i] : x_theta[i];
            for (int k = 0; k < d; ++k)
                lam[i][k] += step * (x_bar[k] - xi[k]);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    V1LDResult res;
    res.v1LD_star      = best_upper;
    res.lambda_star    = std::move(best_lam);
    res.convergence    = std::move(convergence);
    res.times_s        = std::move(times_vec);
    res.gnorm_history  = std::move(gnorm_vec);
    res.z_changes      = std::move(zchanges_vec);
    res.solve_time_s   = std::chrono::duration<double>(t1 - t0).count();
    res.iterations_run = iters_run;
    return res;
}
