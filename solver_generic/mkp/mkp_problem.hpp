#pragma once
/*
 * mkp_problem.hpp — SLDProblem pour le MKP chance-contraint.
 *
 * x ∈ {0,1}^n,  c déterministe,  A_i x ≤ b_i stochastique.
 *
 * solve_theta : greedy — x_k = 1 ssi cost_k > 0        O(n)
 * solve_zeta  : B&B exact multi-contraintes (Dantzig)   O(B&B)
 *               (copie de solve_zeta_i_mkp_bb dans mkp_subproblems.hpp)
 */

#include "sld_base.hpp"

class MKPProblem : public SLDProblem {
public:
    // c : (n,)   p : (N,)   A : (N, m, n)   b : (N, m)
    MKPProblem(Vec1d c, Vec1d p, Vec3d A, Vec2d b, double epsilon);

    int           n_vars()  const override { return n_; }
    int           N_scen()  const override { return N_; }
    double        epsilon() const override { return epsilon_; }
    const Vec1d&  p()       const override { return p_; }
    const Vec1d&  c()       const override { return c_; }

    std::pair<double, Vec1d> solve_theta(const Vec1d& cost) const override;
    std::pair<double, Vec1d> solve_zeta(int i, const Vec1d& cost) const override;

protected:
    int    n_, m_, N_;
    double epsilon_;
    Vec1d  c_, p_;
    Vec3d  A_;    // (N, m, n)
    Vec2d  b_;    // (N, m)
};
