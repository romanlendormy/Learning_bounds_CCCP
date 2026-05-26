#pragma once
#include "v1cld_base.hpp"

/*
 * MKPCLDProblem — V1CLDProblem pour le MKP chance-contraint.
 *
 * La décomposition CLD introduit m copies x_{i,j} par scénario i,
 * une par contrainte j.  Les sous-problèmes sont :
 *
 *   solve_theta(j, cost) = max_{x ∈ {0,1}^n} cost^T x          (sans contrainte)
 *   solve_zeta(i, j, cost) = LP-relax. du knapsack A[i][j] x ≤ b[i][j]  (1D)
 *
 * La LP fractionnaire donne ζ ≥ ζ_entier, donc δ = θ−ζ ≤ δ_entier,
 * ce qui produit une borne supérieure valide (mais légèrement moins serrée
 * que si on utilisait le B&B).  C'est le même choix que solver_mkp/v1mkp_core.
 */
class MKPCLDProblem : public V1CLDProblem {
public:
    MKPCLDProblem(Vec1d c, Vec1d p, Vec3d A, Vec2d b, double epsilon);

    int           n_vars()        const override { return n_; }
    int           N_scen()        const override { return N_; }
    int           n_constraints() const override { return m_; }
    double        epsilon()       const override { return epsilon_; }
    const Vec1d&  p()             const override { return p_; }
    const Vec1d&  c()             const override { return c_; }

    std::pair<double, Vec1d> solve_theta(int j, const Vec1d& cost) const override;
    std::pair<double, Vec1d> solve_zeta(int i, int j, const Vec1d& cost) const override;

private:
    Vec1d  c_;
    Vec1d  p_;
    Vec3d  A_;   // (N, m, n)
    Vec2d  b_;   // (N, m)
    double epsilon_;
    int    n_, N_, m_;
};
