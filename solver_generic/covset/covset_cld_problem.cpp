/*
 * covset_cld_problem.cpp — Implémentation de CovsetCLDProblem.
 */

#include "covset_cld_problem.hpp"
#include "covset_subproblems.hpp"

CovsetCLDProblem::CovsetCLDProblem(Vec1d c, Vec1d p, Vec3d A, Vec2d b, double epsilon)
    : c_(std::move(c)), p_(std::move(p)), A_(std::move(A)), b_(std::move(b)),
      epsilon_(epsilon)
{
    n_ = static_cast<int>(c_.size());
    N_ = static_cast<int>(p_.size());
    m_ = static_cast<int>(b_[0].size());
}

// θ_j : max sans contrainte (j ignoré — domaine de base identique pour toutes les contraintes)
std::pair<double, Vec1d>
CovsetCLDProblem::solve_theta(int /*j*/, const Vec1d& cost) const
{
    return solve_theta_covering(cost);
}

// ζ_{i,j} : max s.c. A[i][j] x >= b[i][j] — glouton 1-contrainte
std::pair<double, Vec1d>
CovsetCLDProblem::solve_zeta(int i, int j, const Vec1d& cost) const
{
    return _solve_single_covering_max(cost, A_[i][j], b_[i][j]);
}
