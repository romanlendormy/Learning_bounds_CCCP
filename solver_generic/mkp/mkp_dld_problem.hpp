#pragma once
#include "dld_base.hpp"

/*
 * MKPDLDProblem — DLDProblem pour le MKP chance-contraint.
 *
 * La décomposition DLD introduit m copies x_{i,j} par scénario i,
 * une par contrainte j.  Les sous-problèmes sont :
 *
 *   solve_theta(j, cost) = max_{x ∈ {0,1}^n} cost^T x          (sans contrainte)
 *   solve_zeta(i, j, cost) = 0/1 knapsack exact 1D via B&B  A[i][j] x ≤ b[i][j]
 *
 * Le DP exact donne ζ = ζ_entier, donc δ = θ−ζ est la vraie pénalité,
 * ce qui produit une borne L1 plus serrée qu'avec la LP fractionnaire.
 */
class MKPDLDProblem : public DLDProblem {
public:
    MKPDLDProblem(Vec1d c, Vec1d p, Vec3d A, Vec2d b, double epsilon);

    int           n_vars()        const override { return n_; }
    int           N_scen()        const override { return N_; }
    int           n_constraints() const override { return m_; }
    double        epsilon()       const override { return epsilon_; }
    const Vec1d&  p()             const override { return p_; }
    const Vec1d&  c()             const override { return c_; }

    std::pair<double, Vec1d> solve_theta(int j, const Vec1d& cost) const override;
    std::pair<double, Vec1d> solve_zeta(int i, int j, const Vec1d& cost) const override;

protected:
    Vec1d  c_;
    Vec1d  p_;
    Vec3d  A_;   // (N, m, n)
    Vec2d  b_;   // (N, m)
    double epsilon_;
    int    n_, N_, m_;
};
