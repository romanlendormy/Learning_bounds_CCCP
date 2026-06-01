#pragma once
/*
 * covset_subproblems.hpp — Sous-problèmes pour le CCSP (covering set).
 *
 * Problème : max c^T x  s.t. Pr[ A_i x >= b_i ] >= 1-ε,  x ∈ {0,1}^n
 * Contraintes : couverture (>=), A et b binaires, b_i[j] ∈ {0,1}.
 *
 * Convention MAX (conforme à solver_generic) :
 *   solve_theta_covering       : O(n)        — max cost^T x, x ∈ {0,1}^n
 *   _solve_single_covering_max : O(n)        — max cost^T x, s.c. A_row x >= b_j
 *   solve_zeta_covering_exact  : O(n · 2^M) — max cost^T x, s.c. A_i x >= b_i
 *                                              M = nb contraintes non satisfaites
 *
 * Symétrie avec solver_cpp2/subproblems.hpp (MIN) :
 *   solver_cpp2 minimise  → take items with cost < 0
 *   ici on maximise       → take items with cost > 0
 */

#include "types.hpp"
#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>


// ── θ : max cost^T x, x ∈ {0,1}^n — O(n) ────────────────────────────────────
// Identique à solve_theta_i_max (mkp_subproblems.hpp).
inline std::pair<double, Vec1d> solve_theta_covering(const Vec1d& cost) {
    int n = static_cast<int>(cost.size());
    Vec1d  x(n, 0.0);
    double val = 0.0;
    for (int k = 0; k < n; ++k)
        if (cost[k] > 0.0) { x[k] = 1.0; val += cost[k]; }
    return {val, x};
}


// ── ζ_{i,j} : max cost^T x s.c. A_row x >= b_j,  x ∈ {0,1}^n — O(n) ─────────
//
// Algorithme :
//   1. Solution non contrainte : x_k = 1 ssi cost_k > 0  (tous les items profitables)
//   2. Vérifier A_row x >= b_j ; si oui → retourner
//   3. Parmi les items non sélectionnés avec A_row[k] > 0, ajouter celui
//      de coût maximal (= perte minimale, car cost[k] <= 0 à ce stade)
//
// Hypothèse : b_j ∈ {0,1}. Ajouter un seul item couvrant suffit.
inline std::pair<double, Vec1d> _solve_single_covering_max(
    const Vec1d& cost_j,
    const Vec1d& A_row,
    double       b_j
) {
    int n = static_cast<int>(cost_j.size());

    Vec1d  x(n, 0.0);
    double val = 0.0;
    for (int k = 0; k < n; ++k)
        if (cost_j[k] > 0.0) { x[k] = 1.0; val += cost_j[k]; }

    if (b_j < 0.5) return {val, x};

    double Ax = 0.0;
    for (int k = 0; k < n; ++k) Ax += A_row[k] * x[k];
    if (Ax >= b_j - 1e-9) return {val, x};

    // Trouver l'item couvrant non sélectionné de coût maximal (perte minimale)
    int    k_star    = -1;
    double best_cost = -std::numeric_limits<double>::infinity();
    for (int k = 0; k < n; ++k) {
        if (A_row[k] > 0.5 && x[k] < 0.5 && cost_j[k] > best_cost) {
            best_cost = cost_j[k];
            k_star    = k;
        }
    }
    if (k_star < 0)
        return {-std::numeric_limits<double>::infinity(), x};  // infaisable

    x[k_star] = 1.0;
    return {val + best_cost, x};
}


// ── ζ_i exact : max cost^T x s.c. A_i x >= b_i,  x ∈ {0,1}^n — O(n · 2^M) ───
//
// M = nombre de contraintes non encore satisfaites par la solution gloutonne.
// Algorithme (symétrique MAX de solve_zeta_i_exact dans solver_cpp2) :
//
//   1. Glouton : sélectionner tous items cost > 0  → solution non contrainte
//   2. Identifier contraintes encore insatisfaites (needed_mask)
//   3. DP 0/1 sur 2^M états de couverture :
//      dp[S] = gain additionnel MAXIMAL pour couvrir les contraintes dans S
//      (dp[0]=0, items candidats ont cost <= 0 → on cherche la perte minimale)
//   4. Backtracking pour reconstruire x
inline std::pair<double, Vec1d> solve_zeta_covering_exact(
    const Vec1d& cost_i,
    const Vec2d& A_i,
    const Vec1d& b_i
) {
    int n = static_cast<int>(cost_i.size());
    int m = static_cast<int>(b_i.size());
    const double NEG_INF = -std::numeric_limits<double>::infinity();

    // Étape 1 : glouton MAX
    Vec1d  x(n, 0.0);
    double val = 0.0;
    for (int k = 0; k < n; ++k)
        if (cost_i[k] > 0.0) { x[k] = 1.0; val += cost_i[k]; }

    // Étape 2 : contraintes non couvertes
    std::vector<int> needed;
    needed.reserve(m);
    for (int j = 0; j < m; ++j) {
        if (b_i[j] < 0.5) continue;
        double Ax = 0.0;
        for (int k = 0; k < n; ++k) Ax += A_i[j][k] * x[k];
        if (Ax < b_i[j] - 1e-9) needed.push_back(j);
    }

    int M = static_cast<int>(needed.size());
    if (M == 0) return {val, x};

    // Étape 3 : DP MAX sur 2^M états
    int full_mask = (1 << M) - 1;
    std::vector<double> dp(full_mask + 1, NEG_INF);
    std::vector<int>    dp_item(full_mask + 1, -1);
    std::vector<int>    dp_prev(full_mask + 1, -1);
    dp[0] = 0.0;

    for (int k = 0; k < n; ++k) {
        if (x[k] > 0.5) continue;     // déjà sélectionné
        // cost_i[k] <= 0 ici (sinon le glouton l'aurait pris)

        int cover_k = 0;
        for (int jj = 0; jj < M; ++jj)
            if (A_i[needed[jj]][k] > 0.5) cover_k |= (1 << jj);
        if (cover_k == 0) continue;   // ne couvre aucune contrainte restante

        // 0/1 DP, itérer de haut en bas (pas de réutilisation)
        for (int S = full_mask; S >= 0; --S) {
            if (dp[S] == NEG_INF) continue;
            int    NS      = S | cover_k;
            double new_val = dp[S] + cost_i[k];  // cost_i[k] <= 0
            if (new_val > dp[NS]) {               // MAXIMISER (perte minimale)
                dp[NS]      = new_val;
                dp_item[NS] = k;
                dp_prev[NS] = S;
            }
        }
    }

    if (dp[full_mask] == NEG_INF)
        return {NEG_INF, x};  // infaisable

    // Étape 4 : backtracking
    val += dp[full_mask];
    int mask = full_mask;
    while (mask != 0) {
        int k  = dp_item[mask];
        x[k]   = 1.0;
        mask   = dp_prev[mask];
    }
    return {val, x};
}
