#pragma once
/*
 * mkp_subproblems.hpp — Sous-problèmes MKP utilisés par MKPDLDProblem.
 *
 * solve_theta_i_max      : O(n)                — max cost^T x, x ∈ {0,1}^n
 * solve_zeta_i_exact_bb  : B&B exact            — 0/1 knapsack 1-contrainte, borne de Dantzig
 * solve_zeta_i_exact_dp  : O(n × W)             — 0/1 knapsack 1-contrainte par DP
 * solve_zeta_i_exact_mitm: O(2^(n/2) × n/2)     — 0/1 knapsack 1-contrainte, meet-in-the-middle
 *                          (Horowitz-Sahni). Travaille en flottant natif (pas de
 *                          discrétisation poids/capacité) : pas de risque
 *                          mémoire/précision contrairement à la DP.
 */

#include "types.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
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


// ── ζ_{i,j} : 0/1 knapsack exact 1-contrainte — Branch & Bound ───────────────
// Items triés par ratio décroissant. Borne de Dantzig à chaque nœud.
// DFS : branche "inclure" en premier (heuristique gloutonne).
inline std::pair<double, Vec1d> solve_zeta_i_exact_bb(
    const Vec1d& cost,
    const Vec1d& a_row,
    double       b_j
) {
    int n = static_cast<int>(cost.size());
    Vec1d x(n, 0.0);
    double val_free = 0.0;

    std::vector<int> items;
    items.reserve(n);
    for (int k = 0; k < n; ++k) {
        if (a_row[k] < 1e-9) {
            if (cost[k] > 0.0) { x[k] = 1.0; val_free += cost[k]; }
        } else if (cost[k] > 0.0) {
            items.push_back(k);
        }
    }

    int ni = static_cast<int>(items.size());
    if (ni == 0) return {val_free, std::move(x)};

    std::sort(items.begin(), items.end(), [&](int a, int b) {
        return cost[a] * a_row[b] > cost[b] * a_row[a];
    });

    std::vector<double> p(ni), w(ni);
    for (int i = 0; i < ni; ++i) { p[i] = cost[items[i]]; w[i] = a_row[items[i]]; }

    struct BB {
        const std::vector<double>& p;
        const std::vector<double>& w;
        int ni;
        double best_val;
        std::vector<bool> best_sel, cur_sel;

        double dantzig_bound(int idx, double cap, double cur_val) const {
            double val = cur_val, rem = cap;
            for (int i = idx; i < ni && rem > 1e-12; ++i) {
                double take = std::min(1.0, rem / w[i]);
                val += take * p[i]; rem -= take * w[i];
            }
            return val;
        }

        void solve(int idx, double cap, double cur_val) {
            if (dantzig_bound(idx, cap, cur_val) <= best_val + 1e-9) return;
            if (idx == ni) {
                if (cur_val > best_val + 1e-9) { best_val = cur_val; best_sel = cur_sel; }
                return;
            }
            if (w[idx] <= cap + 1e-9) {
                cur_sel[idx] = true;
                solve(idx + 1, cap - w[idx], cur_val + p[idx]);
                cur_sel[idx] = false;
            }
            solve(idx + 1, cap, cur_val);
        }
    };

    BB bb{p, w, ni, 0.0,
          std::vector<bool>(ni, false), std::vector<bool>(ni, false)};
    bb.solve(0, b_j, 0.0);

    for (int i = 0; i < ni; ++i)
        if (bb.best_sel[i]) x[items[i]] = 1.0;

    return {val_free + bb.best_val, std::move(x)};
}


// ── ζ_{i,j} : 0/1 knapsack exact 1-contrainte — DP O(n × W) ─────────────────
// max cost^T x  s.t.  a_row^T x ≤ b_j,  x ∈ {0,1}^n
//
// Poids a_row et capacité b_j supposés entiers (arrondis).
// Table 2D pour reconstruction exacte de la solution.
inline std::pair<double, Vec1d> solve_zeta_i_exact_dp(
    const Vec1d& cost,
    const Vec1d& a_row,
    double       b_j
) {
    int n = static_cast<int>(cost.size());
    Vec1d x(n, 0.0);
    double val_free = 0.0;

    // Items à poids nul : inclus si coût > 0.  Items à coût ≤ 0 : exclus.
    std::vector<int> items;
    items.reserve(n);
    for (int k = 0; k < n; ++k) {
        if (a_row[k] < 1e-9) {
            if (cost[k] > 0.0) { x[k] = 1.0; val_free += cost[k]; }
        } else if (cost[k] > 0.0) {
            items.push_back(k);
        }
    }

    int ni = static_cast<int>(items.size());
    if (ni == 0) return {val_free, std::move(x)};

    int B = std::max(0, static_cast<int>(std::round(b_j)));

    std::vector<int> w(ni);
    for (int i = 0; i < ni; ++i)
        w[i] = static_cast<int>(std::round(a_row[items[i]]));

    // Table 2D : dp[i+1][c] = meilleure valeur avec items 0..i, capacité ≤ c
    std::vector<std::vector<double>> dp(ni + 1, std::vector<double>(B + 1, 0.0));
    for (int i = 0; i < ni; ++i) {
        const double pi = cost[items[i]];
        const int    wi = w[i];
        for (int c = 0; c <= B; ++c) {
            dp[i+1][c] = dp[i][c];
            if (wi <= c)
                dp[i+1][c] = std::max(dp[i+1][c], dp[i][c - wi] + pi);
        }
    }

    // Reconstruction par backtracking
    int cap = B;
    for (int i = ni - 1; i >= 0; --i) {
        if (w[i] <= cap &&
            dp[i][cap - w[i]] + cost[items[i]] > dp[i][cap] + 1e-9) {
            x[items[i]] = 1.0;
            cap -= w[i];
        }
    }

    return {val_free + dp[ni][B], std::move(x)};
}


// ── ζ_{i,j} : 0/1 knapsack exact 1-contrainte — Meet-in-the-middle ──────────
// (Horowitz-Sahni). Split en 2 moitiés, énumère TOUS les sous-ensembles de
// chaque moitié (2^(n/2) chacune), réduit une moitié à son enveloppe
// monotone (skyline : poids croissant, profit strictement croissant), puis
// combine par recherche binaire. Poids/profits en flottant natif — pas de
// scaling/discrétisation nécessaire, contrairement à solve_zeta_i_exact_dp.
inline std::pair<double, Vec1d> solve_zeta_i_exact_mitm(
    const Vec1d& cost,
    const Vec1d& a_row,
    double       b_j
) {
    int n = static_cast<int>(cost.size());
    Vec1d x(n, 0.0);
    double val_free = 0.0;

    std::vector<int> items;
    items.reserve(n);
    for (int k = 0; k < n; ++k) {
        if (a_row[k] < 1e-9) {
            if (cost[k] > 0.0) { x[k] = 1.0; val_free += cost[k]; }
        } else if (cost[k] > 0.0) {
            items.push_back(k);
        }
    }

    int ni = static_cast<int>(items.size());
    if (ni == 0) return {val_free, std::move(x)};

    const int half1 = ni / 2;
    const int half2 = ni - half1;

    struct Sub { double w, p; uint32_t mask; };

    auto enumerate = [&](int start, int count) {
        std::vector<Sub> subs;
        subs.reserve(static_cast<size_t>(1) << count);
        for (uint32_t mask = 0; mask < (1u << count); ++mask) {
            double w = 0.0, p = 0.0;
            for (int b = 0; b < count; ++b) {
                if (mask & (1u << b)) {
                    int k = items[start + b];
                    w += a_row[k];
                    p += cost[k];
                }
            }
            subs.push_back({w, p, mask});
        }
        return subs;
    };

    auto subsA = enumerate(0, half1);       // items [0, half1)
    auto subsB = enumerate(half1, half2);   // items [half1, ni)

    // Skyline de A : trié par poids croissant, ne garde que les points où le
    // profit augmente STRICTEMENT (points dominés éliminés) — permet, pour
    // toute capacité restante, une recherche binaire O(log) du meilleur A
    // compatible au lieu d'un balayage linéaire.
    std::sort(subsA.begin(), subsA.end(), [](const Sub& x1, const Sub& x2) {
        return x1.w < x2.w;
    });
    std::vector<double>   skylineW, skylineP;
    std::vector<uint32_t> skylineMask;
    skylineW.reserve(subsA.size());
    for (const auto& s : subsA) {
        if (!skylineP.empty() && s.p <= skylineP.back() + 1e-12) continue; // dominé
        skylineW.push_back(s.w);
        skylineP.push_back(s.p);
        skylineMask.push_back(s.mask);
    }

    double   best_val   = 0.0;
    uint32_t best_maskA = 0, best_maskB = 0;
    for (const auto& sb : subsB) {
        if (sb.w > b_j + 1e-9) continue;
        double cap_rem = b_j - sb.w;
        // Plus grand poids de la skyline <= cap_rem.
        auto it = std::upper_bound(skylineW.begin(), skylineW.end(), cap_rem + 1e-9);
        double   best_pA = 0.0;
        uint32_t maskA   = 0;
        if (it != skylineW.begin()) {
            size_t idx = static_cast<size_t>(it - skylineW.begin()) - 1;
            best_pA = skylineP[idx];
            maskA   = skylineMask[idx];
        }
        double total = sb.p + best_pA;
        if (total > best_val + 1e-9) {
            best_val   = total;
            best_maskA = maskA;
            best_maskB = sb.mask;
        }
    }

    for (int b = 0; b < half1; ++b)
        if (best_maskA & (1u << b)) x[items[b]] = 1.0;
    for (int b = 0; b < half2; ++b)
        if (best_maskB & (1u << b)) x[items[half1 + b]] = 1.0;

    return {val_free + best_val, std::move(x)};
}
