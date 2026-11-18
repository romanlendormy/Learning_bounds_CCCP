/*
 * mkp_problem_bnb_surrogate.cpp — solve_zeta par Branch-and-Bound exact,
 * borne surrogate (cf. mkp_surrogate_bound.hpp).
 *
 * Squelette de recherche IDENTIQUE à mkp_problem_bnb.cpp (mêmes items "libres"
 * auto-inclus, même style de remplissage glouton fractionnaire pour la borne)
 * — seules changent : (a) la fonction de borne (agrégée au lieu de
 * min-par-dimension), (b) l'ordre de branchement, qui suit maintenant le tri
 * par ratio agrégé (prep.active_sorted) au lieu du tri "surrogate décroissant"
 * ad hoc de MKPProblemBnB — les deux sont conceptuellement proches, mais ici
 * le MÊME tri sert à la fois pour l'ordre de branchement et pour la borne.
 */

#include "mkp_problem_bnb_surrogate.hpp"
#include "mkp_surrogate_bound.hpp"

#include <vector>

namespace {

struct MkpBnBSurr {
    const MkpSurrogatePrep* prep;   // active_sorted/agg_w_sorted = ordre de branchement ET de borne
    const Vec1d*            cost;
    int na, m;
    double best_val;
    std::vector<int8_t> x_bits, best_x_bits;

    void run(int idx, Vec1d& rem_b, double val, const Vec2d& A_rows) {
        if (idx == na) {
            if (val > best_val + 1e-9) { best_val = val; best_x_bits = x_bits; }
            return;
        }
        double ub = mkp_surrogate_bound(*prep, *cost, idx, rem_b, val, m);
        if (ub <= best_val + 1e-9) return;

        int k = prep->active_sorted[idx];
        bool feasible = true;
        for (int j = 0; j < m; ++j)
            if (A_rows[j][k] > rem_b[j] + 1e-12) { feasible = false; break; }
        if (feasible) {
            for (int j = 0; j < m; ++j) rem_b[j] -= A_rows[j][k];
            x_bits[idx] = 1;
            run(idx + 1, rem_b, val + (*cost)[k], A_rows);
            x_bits[idx] = 0;
            for (int j = 0; j < m; ++j) rem_b[j] += A_rows[j][k];
        }
        run(idx + 1, rem_b, val, A_rows);
    }
};

} // anonymous namespace

std::pair<double, Vec1d> MKPProblemBnBSurrogate::solve_zeta(int i, const Vec1d& cost) const
{
    Vec1d  x_out(n_, 0.0);
    double val_free = 0.0;
    const Vec2d& Ai = A_[i];
    const Vec1d& bi = b_[i];

    std::vector<int> active;
    active.reserve(n_);
    for (int k = 0; k < n_; ++k) {
        if (cost[k] <= 0.0) continue;
        bool free_item = true;
        for (int j = 0; j < m_; ++j)
            if (Ai[j][k] >= 1e-9) { free_item = false; break; }
        if (free_item) { x_out[k] = 1.0; val_free += cost[k]; }
        else           { active.push_back(k); }
    }
    if (active.empty()) return {val_free, std::move(x_out)};

    MkpSurrogatePrep prep = prepare_mkp_surrogate(active, cost, Ai, bi, m_);
    int na = static_cast<int>(prep.active_sorted.size());

    MkpBnBSurr bb;
    bb.prep = &prep;
    bb.cost = &cost;
    bb.na = na; bb.m = m_;
    bb.best_val = val_free;
    bb.x_bits.assign(na, 0); bb.best_x_bits.assign(na, 0);

    Vec1d rem_b(bi.begin(), bi.end());
    bb.run(0, rem_b, val_free, Ai);

    for (int ki = 0; ki < na; ++ki)
        if (bb.best_x_bits[ki]) x_out[prep.active_sorted[ki]] = 1.0;

    return {bb.best_val, std::move(x_out)};
}
