/*
 * covset_problem.cpp — Implémentation de CovsetProblem.
 */

#include "covset_problem.hpp"
#include "covset_subproblems.hpp"

CovsetProblem::CovsetProblem(Vec1d c, Vec1d p, Vec3d A, Vec2d b, double epsilon)
    : c_(std::move(c)), p_(std::move(p)), A_(std::move(A)), b_(std::move(b)),
      epsilon_(epsilon)
{
    n_ = static_cast<int>(c_.size());
    N_ = static_cast<int>(p_.size());
    m_ = static_cast<int>(b_[0].size());
}

std::pair<double, Vec1d> CovsetProblem::solve_theta(const Vec1d& cost) const
{
    return solve_theta_covering(cost);
}

std::pair<double, Vec1d> CovsetProblem::solve_zeta(int i, const Vec1d& cost) const
{
    return solve_zeta_covering_exact(cost, A_[i], b_[i]);
}
