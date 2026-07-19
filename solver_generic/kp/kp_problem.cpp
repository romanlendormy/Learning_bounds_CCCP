#include "kp_problem.hpp"
#include "mkp_subproblems.hpp"

KPProblem::KPProblem(Vec1d c, Vec1d p, Vec2d A, Vec1d b, double epsilon)
    : c_(std::move(c)), p_(std::move(p)), b_(std::move(b)), A_(std::move(A)),
      epsilon_(epsilon)
{
    n_ = static_cast<int>(c_.size());
    N_ = static_cast<int>(p_.size());
}

// θ : greedy O(n) — identique au cas MKP général (aucune contrainte)
std::pair<double, Vec1d> KPProblem::solve_theta(const Vec1d& cost) const
{
    return solve_theta_i_max(cost);
}

// ζ_i : 0/1 knapsack exact à une contrainte a_i^T x ≤ b_i, via B&B (Dantzig)
std::pair<double, Vec1d> KPProblem::solve_zeta(int i, const Vec1d& cost) const
{
    return solve_zeta_i_exact_bb(cost, A_[i], b_[i]);
}
