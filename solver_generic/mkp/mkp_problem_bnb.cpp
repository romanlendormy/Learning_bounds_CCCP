/*
 * mkp_problem_bnb.cpp — solve_zeta par Branch-and-Bound exact (variante de ref.).
 *
 * Algorithme identique à l'implémentation originale de mkp_problem.cpp avant
 * l'introduction de CP-SAT.  Conservé pour comparaison expérimentale.
 *
 * Borne supérieure LP par colonne (Dantzig) sur chaque dimension,
 * minimum sur les dimensions → upper bound tight en 1D.
 */

#include "mkp_problem_bnb.hpp"

#include <algorithm>
#include <numeric>
#include <vector>

namespace {

struct MkpBnB {
    const int*    active;
    const double* cost;
    const double* const* A_rows;
    int na, m;
    double best_val;
    std::vector<int8_t> x_bits, best_x_bits;
    std::vector<std::vector<int>> sort_j;

    double lp_j(int j, int idx, double cap_j, double val) const {
        double v = val;
        for (int ki : sort_j[j]) {
            if (ki < idx) continue;
            int k    = active[ki];
            double a = A_rows[j][k];
            if      (a < 1e-12)        { v += cost[k]; }
            else if (a <= cap_j+1e-12) { v += cost[k]; cap_j -= a; }
            else                       { v += cost[k] * cap_j / a; break; }
        }
        return v;
    }

    double upper_bound(int idx, const double* rem_b, double val) const {
        double ub = 1e30;
        for (int j = 0; j < m; ++j) {
            double lj = lp_j(j, idx, rem_b[j], val);
            if (lj < ub) ub = lj;
        }
        return ub;
    }

    void run(int idx, double* rem_b, double val) {
        if (idx == na) {
            if (val > best_val + 1e-9) { best_val = val; best_x_bits = x_bits; }
            return;
        }
        if (upper_bound(idx, rem_b, val) <= best_val + 1e-9) return;
        int k = active[idx];
        bool feasible = true;
        for (int j = 0; j < m; ++j)
            if (A_rows[j][k] > rem_b[j] + 1e-12) { feasible = false; break; }
        if (feasible) {
            for (int j = 0; j < m; ++j) rem_b[j] -= A_rows[j][k];
            x_bits[idx] = 1;
            run(idx + 1, rem_b, val + cost[k]);
            x_bits[idx] = 0;
            for (int j = 0; j < m; ++j) rem_b[j] += A_rows[j][k];
        }
        run(idx + 1, rem_b, val);
    }
};

} // anonymous namespace

std::pair<double, Vec1d> MKPProblemBnB::solve_zeta(int i, const Vec1d& cost) const
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

    // Tri surrogate décroissant
    std::sort(active.begin(), active.end(), [&](int a, int b_idx) {
        double wa = 0.0, wb = 0.0;
        for (int j = 0; j < m_; ++j) {
            if (bi[j] > 1e-12) { wa += Ai[j][a]/bi[j]; wb += Ai[j][b_idx]/bi[j]; }
        }
        double ra = (wa > 1e-12) ? cost[a]/wa   : 1e18;
        double rb = (wb > 1e-12) ? cost[b_idx]/wb : 1e18;
        return ra > rb;
    });

    int na = static_cast<int>(active.size());
    std::vector<std::vector<int>> sort_j(m_, std::vector<int>(na));
    for (int j = 0; j < m_; ++j) {
        std::iota(sort_j[j].begin(), sort_j[j].end(), 0);
        std::sort(sort_j[j].begin(), sort_j[j].end(), [&](int a, int b_idx) {
            int ka = active[a], kb = active[b_idx];
            double ra = (Ai[j][ka] > 1e-12) ? cost[ka]/Ai[j][ka] : 1e18;
            double rb = (Ai[j][kb] > 1e-12) ? cost[kb]/Ai[j][kb] : 1e18;
            return ra > rb;
        });
    }
    std::vector<const double*> A_ptrs(m_);
    for (int j = 0; j < m_; ++j) A_ptrs[j] = Ai[j].data();

    MkpBnB bb;
    bb.active  = active.data(); bb.cost = cost.data();
    bb.A_rows  = A_ptrs.data(); bb.na = na; bb.m = m_;
    bb.best_val = val_free;
    bb.x_bits.assign(na, 0); bb.best_x_bits.assign(na, 0);
    bb.sort_j = std::move(sort_j);

    std::vector<double> rem_b(bi.begin(), bi.end());
    bb.run(0, rem_b.data(), val_free);

    for (int ki = 0; ki < na; ++ki)
        if (bb.best_x_bits[ki]) x_out[active[ki]] = 1.0;

    return {bb.best_val, std::move(x_out)};
}
