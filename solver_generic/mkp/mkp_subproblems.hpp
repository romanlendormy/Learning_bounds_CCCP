#pragma once
/*
 * mkp_subproblems.hpp — Sous-problèmes MKP utilisés par MKPCLDProblem.
 *
 * solve_theta_i_max        : O(n)       — max cost^T x, x ∈ {0,1}^n
 * solve_zeta_i_packing_lp  : O(n log n) — LP relaxation 1-contrainte (borne supérieure)
 */

#include "types.hpp"
#include <algorithm>
#include <vector>


// ── θ_i : max cost^T x, x ∈ {0,1}^n — O(n) ──────────────────────────────────
inline std::pair<double, Vec1d> solve_theta_i_max(const Vec1d& cost) {
    int n = static_cast<int>(cost.size());
    Vec1d x(n, 0.0);
    double val = 0.0;
    for (int k = 0; k < n; ++k)
        if (cost[k] > 0.0) { x[k] = 1.0; val += cost[k]; }
    return {val, x};
}


// ── ζ_{i,j} : LP-relaxation du knapsack à une contrainte — O(n log n) ─────────
// max cost^T x  s.t.  a_row^T x ≤ b_j,  x ∈ [0,1]^n
// Retourne une BORNE SUPÉRIEURE valide sur le problème entier (LP ≥ IP).
inline std::pair<double, Vec1d> solve_zeta_i_packing_lp(
    const Vec1d& cost,
    const Vec1d& a_row,
    double       b_j
) {
    int n = static_cast<int>(cost.size());
    Vec1d  x(n, 0.0);
    double val      = 0.0;
    double cap_used = 0.0;

    for (int k = 0; k < n; ++k) {
        if (a_row[k] < 1e-9 && cost[k] > 0.0) {
            x[k] = 1.0;
            val  += cost[k];
        }
    }

    std::vector<int> idx;
    idx.reserve(n);
    for (int k = 0; k < n; ++k)
        if (a_row[k] >= 1e-9 && cost[k] > 0.0)
            idx.push_back(k);

    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return cost[a] * a_row[b] > cost[b] * a_row[a];
    });

    for (int k : idx) {
        double remaining = b_j - cap_used;
        if (remaining <= 1e-12) break;
        double fill  = std::min(1.0, remaining / a_row[k]);
        x[k]         = fill;
        val          += fill * cost[k];
        cap_used     += fill * a_row[k];
    }

    return {val, x};
}
