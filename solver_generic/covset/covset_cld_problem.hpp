#pragma once
/*
 * covset_cld_problem.hpp — V1CLDProblem pour le CCSP (covering set).
 *
 * La décomposition CLD introduit m copies x_{i,j} par scénario i,
 * une par contrainte j.  Les sous-problèmes sont :
 *
 *   solve_theta(j, cost) = max_{x ∈ {0,1}^n} cost^T x    (sans contrainte)
 *   solve_zeta(i, j, cost) = max_{x ∈ {0,1}^n : A[i][j] x >= b[i][j]} cost^T x
 *
 * solve_zeta est résolu par le glouton _solve_single_covering_max — O(n).
 * Hypothèse : b[i][j] ∈ {0, 1} (un seul item couvrant suffit).
 */

#include "v1cld_base.hpp"

class CovsetCLDProblem : public V1CLDProblem {
public:
    // c : (n,)   p : (N,)   A : (N, m, n) binaire   b : (N, m) binaire
    CovsetCLDProblem(Vec1d c, Vec1d p, Vec3d A, Vec2d b, double epsilon);

    int           n_vars()        const override { return n_; }
    int           N_scen()        const override { return N_; }
    int           n_constraints() const override { return m_; }
    double        epsilon()       const override { return epsilon_; }
    const Vec1d&  p()             const override { return p_; }
    const Vec1d&  c()             const override { return c_; }

    std::pair<double, Vec1d> solve_theta(int j, const Vec1d& cost) const override;
    std::pair<double, Vec1d> solve_zeta(int i, int j, const Vec1d& cost) const override;

private:
    Vec1d  c_, p_;
    Vec3d  A_;   // (N, m, n)
    Vec2d  b_;   // (N, m)
    double epsilon_;
    int    n_, N_, m_;
};
