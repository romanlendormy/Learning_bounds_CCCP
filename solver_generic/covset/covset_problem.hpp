#pragma once
/*
 * covset_problem.hpp — V1LDProblem pour le CCSP (covering set).
 *
 * max c^T x  s.t.  Pr[ A_i x >= b_i ] >= 1-ε,  x ∈ {0,1}^n
 *
 * solve_theta : greedy max — x_k = 1 ssi cost_k > 0         O(n)
 * solve_zeta  : bitmask DP exact sur les contraintes ≥       O(n · 2^M)
 *               M = nb contraintes non satisfaites (typiquement petit)
 */

#include "v1ld_base.hpp"

class CovsetProblem : public V1LDProblem {
public:
    // c : (n,)   p : (N,)   A : (N, m, n) binaire   b : (N, m) binaire
    CovsetProblem(Vec1d c, Vec1d p, Vec3d A, Vec2d b, double epsilon);

    int           n_vars()  const override { return n_; }
    int           N_scen()  const override { return N_; }
    double        epsilon() const override { return epsilon_; }
    const Vec1d&  p()       const override { return p_; }
    const Vec1d&  c()       const override { return c_; }

    std::pair<double, Vec1d> solve_theta(const Vec1d& cost) const override;
    std::pair<double, Vec1d> solve_zeta(int i, const Vec1d& cost) const override;

private:
    int    n_, m_, N_;
    double epsilon_;
    Vec1d  c_, p_;
    Vec3d  A_;   // (N, m, n) binaire
    Vec2d  b_;   // (N, m) binaire
};
