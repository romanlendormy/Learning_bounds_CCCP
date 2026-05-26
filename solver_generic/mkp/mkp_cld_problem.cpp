#include "mkp_cld_problem.hpp"
#include "mkp_subproblems.hpp"

MKPCLDProblem::MKPCLDProblem(Vec1d c, Vec1d p, Vec3d A, Vec2d b, double epsilon)
    : c_(std::move(c)), p_(std::move(p)), A_(std::move(A)), b_(std::move(b)),
      epsilon_(epsilon)
{
    n_ = static_cast<int>(c_.size());
    N_ = static_cast<int>(p_.size());
    m_ = static_cast<int>(b_[0].size());
}

// θ_j : max sans contrainte — identique à solve_theta_i_max (j ignoré)
std::pair<double, Vec1d>
MKPCLDProblem::solve_theta(int /*j*/, const Vec1d& cost) const
{
    return solve_theta_i_max(cost);
}

// ζ_{i,j} : LP-relaxation du knapsack à une contrainte A[i][j] x ≤ b[i][j]
std::pair<double, Vec1d>
MKPCLDProblem::solve_zeta(int i, int j, const Vec1d& cost) const
{
    return solve_zeta_i_packing_lp(cost, A_[i][j], b_[i][j]);
}
