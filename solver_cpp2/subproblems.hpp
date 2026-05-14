#pragma once
/*
 * Sous-problèmes du solveur v1LD — adaptés strictement de v1ld_solver_general.py.
 *
 * solve_theta_i          : O(n)     — min cost^T x, x ∈ {0,1}^n
 * _solve_single_covering : O(n)     — min cost^T x s.c. A_row x ≥ b (1 contrainte)
 * solve_zeta_i_general   : O(m·n)  — décomposition interne ν (decomp=True)
 * solve_zeta_i_exact     : O(n·2^m) — bitmask DP set cover (decomp=False, m petit)
 * solve_z_knapsack_min   : O(N log N) — knapsack glouton sur z
 */

#include "types.hpp"

#include <algorithm>
#include <numeric>
#include <limits>
#include <tuple>


// ── θ_i : min cost^T x, x ∈ {0,1}^n — O(n) ─────────────────────────────────
inline std::pair<double, Vec1d> solve_theta_i(const Vec1d& cost_i) {
    int n = static_cast<int>(cost_i.size());
    Vec1d x(n, 0.0);
    double val = 0.0;
    for (int k = 0; k < n; ++k)
        if (cost_i[k] < 0.0) { x[k] = 1.0; val += cost_i[k]; }
    return {val, x};
}


// ── Sous-problème 1 contrainte : min cost^T x s.c. A_row x ≥ b — O(n) ───────
// Correspondance exacte avec Python _solve_single_covering.
inline std::pair<Vec1d, double> _solve_single_covering(
    const Vec1d& cost_j,
    const Vec1d& A_row,   // binaire (0 ou 1 en double)
    double       b_j
) {
    int n = static_cast<int>(cost_j.size());

    // Solution non contrainte : x_k = 1 ssi cost_j[k] < 0
    Vec1d x(n, 0.0);
    double val = 0.0;
    for (int k = 0; k < n; ++k)
        if (cost_j[k] < 0.0) { x[k] = 1.0; val += cost_j[k]; }

    if (b_j < 0.5) return {x, val};

    // Vérifier si la contrainte est satisfaite
    double Ax = 0.0;
    for (int k = 0; k < n; ++k) Ax += A_row[k] * x[k];
    if (Ax >= b_j - 1e-9) return {x, val};

    // Trouver l'item couvrant de coût minimal parmi ceux non sélectionnés
    int    k_star    = -1;
    double best_cost = std::numeric_limits<double>::infinity();
    for (int k = 0; k < n; ++k) {
        if (A_row[k] > 0.5 && x[k] < 0.5 && cost_j[k] < best_cost) {
            best_cost = cost_j[k];
            k_star    = k;
        }
    }
    if (k_star < 0)
        return {x, std::numeric_limits<double>::infinity()};  // infaisable

    x[k_star] = 1.0;
    return {x, val + best_cost};
}


// ── ζ̂_i via décomposition interne (decomp=True) — O(m·n) ────────────────────
// Correspondance exacte avec Python solve_zeta_i_general.
// Retourne {val/m, x_bar, x_by_j}
inline std::tuple<double, Vec1d, Vec2d> solve_zeta_i_general(
    const Vec1d& cost_i,  // (n,) coût modifié pour scénario i
    const Vec2d& A_i,     // (m, n) binaire
    const Vec1d& b_i,     // (m,) binaire
    const Vec2d& mu_i     // (m, n) multiplicateurs de consensus interne
) {
    int m = static_cast<int>(A_i.size());
    int n = static_cast<int>(cost_i.size());

    // mu_sum = Σ_{j'} mu_i[j']   (n,)
    Vec1d mu_sum(n, 0.0);
    for (int j = 0; j < m; ++j)
        for (int k = 0; k < n; ++k)
            mu_sum[k] += mu_i[j][k];

    Vec2d  x_by_j(m, Vec1d(n));
    double val = 0.0;

    for (int j = 0; j < m; ++j) {
        // cost_{i,j}[k] = cost_i[k] + mu_i[j][k] - mu_sum[k] / m
        Vec1d cost_ij(n);
        for (int k = 0; k < n; ++k)
            cost_ij[k] = cost_i[k] + mu_i[j][k] - mu_sum[k] / m;

        auto [x_ij, v_ij] = _solve_single_covering(cost_ij, A_i[j], b_i[j]);
        x_by_j[j] = std::move(x_ij);
        val += v_ij;
    }

    // x_bar = (1/m) Σ_j x_{i,j}
    Vec1d x_bar(n, 0.0);
    for (int j = 0; j < m; ++j)
        for (int k = 0; k < n; ++k)
            x_bar[k] += x_by_j[j][k];
    for (int k = 0; k < n; ++k) x_bar[k] /= m;

    return {val / m, x_bar, x_by_j};
}


// ── ζ_i exact par bitmask DP set cover (decomp=False) — O(n · 2^m) ───────────
// Pour m petit (e.g. m=5 → 32 états), bien plus rapide que Gecode BAB.
// Algorithme :
//   1. Glouton : sélectionner tous les items de coût < 0 (solution non contrainte)
//   2. Identifier les contraintes encore non couvertes (needed_mask)
//   3. DP 0/1 sur les 2^M états de couverture restants
//   4. Backtracking pour reconstruire x
inline std::pair<double, Vec1d> solve_zeta_i_exact(
    const Vec1d& cost_i,  // (n,)
    const Vec2d& A_i,     // (m, n) binaire
    const Vec1d& b_i      // (m,) binaire
) {
    int n = static_cast<int>(cost_i.size());
    int m = static_cast<int>(b_i.size());
    const double INF = std::numeric_limits<double>::infinity();

    // Étape 1 : sélection gloutonne des items de coût négatif
    Vec1d  x(n, 0.0);
    double val = 0.0;
    for (int k = 0; k < n; ++k)
        if (cost_i[k] < 0.0) { x[k] = 1.0; val += cost_i[k]; }

    // Étape 2 : contraintes non encore couvertes
    std::vector<int> needed;  // indices dans {0..m-1} des contraintes à couvrir
    needed.reserve(m);
    for (int j = 0; j < m; ++j) {
        if (b_i[j] < 0.5) continue;
        double Ax = 0.0;
        for (int k = 0; k < n; ++k) Ax += A_i[j][k] * x[k];
        if (Ax < b_i[j] - 1e-9) needed.push_back(j);
    }

    int M = static_cast<int>(needed.size());
    if (M == 0) return {val, x};  // déjà faisable

    // Étape 3 : DP 0/1 sur 2^M états
    // dp[S] = coût additionnel minimal pour couvrir toutes les contraintes dans S
    int full_mask = (1 << M) - 1;
    std::vector<double> dp(full_mask + 1, INF);
    std::vector<int>    dp_item(full_mask + 1, -1);  // item ajouté vers cet état
    std::vector<int>    dp_prev(full_mask + 1, -1);  // état précédent
    dp[0] = 0.0;

    for (int k = 0; k < n; ++k) {
        if (x[k] > 0.5) continue;     // déjà sélectionné par le glouton
        // cost_i[k] >= 0 ici (sinon le glouton l'aurait pris)

        // Calculer le masque de couverture de k sur les contraintes needed
        int cover_k = 0;
        for (int jj = 0; jj < M; ++jj)
            if (A_i[needed[jj]][k] > 0.5) cover_k |= (1 << jj);
        if (cover_k == 0) continue;   // n'aide aucune contrainte restante

        // Mise à jour 0/1 : itérer de haut en bas pour éviter la réutilisation
        for (int S = full_mask; S >= 0; --S) {
            if (dp[S] == INF) continue;
            int    NS       = S | cover_k;
            double new_cost = dp[S] + cost_i[k];
            if (new_cost < dp[NS]) {
                dp[NS]      = new_cost;
                dp_item[NS] = k;
                dp_prev[NS] = S;
            }
        }
    }

    if (dp[full_mask] == INF)
        return {INF, x};  // infaisable (ne devrait pas arriver sur instances bien formées)

    // Étape 4 : backtracking pour reconstruire x
    val += dp[full_mask];
    int mask = full_mask;
    while (mask != 0) {
        int k = dp_item[mask];
        x[k]  = 1.0;
        mask   = dp_prev[mask];
    }
    return {val, x};
}


// ── Knapsack z glouton — O(N log N) ──────────────────────────────────────────
// Correspondance exacte avec Python solve_z_knapsack_min.
// Trie par δ croissant, accumule p jusqu'à Σ p_i z_i ≥ 1 - ε.
inline std::pair<Vec1d, double> solve_z_knapsack_min(
    const Vec1d& delta,
    const Vec1d& p,
    double       epsilon
) {
    int N = static_cast<int>(delta.size());

    std::vector<int> order(N);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return delta[a] < delta[b]; });

    Vec1d  z(N, 0.0);
    double cumsum = 0.0;
    double val    = 0.0;
    for (int idx : order) {
        z[idx]   = 1.0;
        cumsum  += p[idx];
        val     += p[idx] * delta[idx];
        if (cumsum >= 1.0 - epsilon - 1e-9) break;
    }
    return {z, val};
}
