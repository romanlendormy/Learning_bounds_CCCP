/*
 * mkp_problem.cpp — MKPProblem: solve_theta (greedy) + solve_zeta (CP-SAT).
 *
 * solve_zeta résout le sous-problème multi-dimensionnel 0-1 knapsack exactement
 * via OR-Tools CP-SAT. Les coefficients float sont convertis en int64 avec
 * SCALE = 1000 (3 décimales de précision). Pas de contrainte Big-M : les
 * contraintes de capacité sont posées directement.
 *
 * Dépendance : brew install or-tools  →  find_package(ortools CONFIG REQUIRED)
 */

#include "mkp_problem.hpp"
#include "ortools/sat/cp_model.h"
#include "ortools/sat/sat_parameters.pb.h"

#include <cmath>
#include <vector>

namespace sat = operations_research::sat;
using operations_research::Domain;

// ── Constructeur ─────────────────────────────────────────────────────────────

MKPProblem::MKPProblem(Vec1d c, Vec1d p, Vec3d A, Vec2d b, double epsilon)
    : c_(std::move(c)), p_(std::move(p)), A_(std::move(A)), b_(std::move(b)),
      epsilon_(epsilon)
{
    n_ = static_cast<int>(c_.size());
    N_ = static_cast<int>(p_.size());
    m_ = static_cast<int>(b_[0].size());
}

// ── θ_i : greedy O(n) ────────────────────────────────────────────────────────

std::pair<double, Vec1d> MKPProblem::solve_theta(const Vec1d& cost) const
{
    Vec1d x(n_, 0.0);
    double val = 0.0;
    for (int k = 0; k < n_; ++k)
        if (cost[k] > 0.0) { x[k] = 1.0; val += cost[k]; }
    return {val, std::move(x)};
}

// ── ζ_i : CP-SAT (0-1 knapsack multi-dimensionnel) ───────────────────────────

std::pair<double, Vec1d> MKPProblem::solve_zeta(int i, const Vec1d& cost) const
{
    const Vec2d& Ai = A_[i];
    const Vec1d& bi = b_[i];

    // Float → int64 : SCALE = 1000 préserve 3 décimales.
    // Si les coefficients A et b sont des entiers, le résultat est exact.
    constexpr int64_t SCALE = 1000;

    sat::CpModelBuilder model;

    // Variables binaires x_k ∈ {0, 1}
    std::vector<sat::BoolVar> x(n_);
    for (int k = 0; k < n_; ++k)
        x[k] = model.NewBoolVar();

    // Contraintes de capacité : ∑_k A[i][j][k] * x_k ≤ b[i][j],  ∀ j = 0..m-1
    for (int j = 0; j < m_; ++j) {
        std::vector<int64_t> w(n_);
        for (int k = 0; k < n_; ++k)
            w[k] = static_cast<int64_t>(std::llround(Ai[j][k] * SCALE));
        // floor sur la capacité : conservatif, garantit la faisabilité originale
        int64_t cap = static_cast<int64_t>(std::floor(bi[j] * SCALE));
        model.AddLinearConstraint(
            sat::LinearExpr::WeightedSum(x, w),
            Domain(0, cap)
        );
    }

    // Objectif : max ∑_k cost[k] * x_k
    std::vector<int64_t> profits(n_);
    for (int k = 0; k < n_; ++k)
        profits[k] = static_cast<int64_t>(std::llround(cost[k] * SCALE));
    model.Maximize(sat::LinearExpr::WeightedSum(x, profits));

    // Résolution — mono-thread, silencieux (sous-problème appelé N fois)
    sat::SatParameters params;
    params.set_log_search_progress(false);
    params.set_num_workers(1);
    const sat::CpSolverResponse resp =
        sat::SolveWithParameters(model.Build(), params);

    Vec1d x_out(n_, 0.0);
    double val = 0.0;
    if (resp.status() == sat::CpSolverStatus::OPTIMAL ||
        resp.status() == sat::CpSolverStatus::FEASIBLE) {
        for (int k = 0; k < n_; ++k) {
            if (sat::SolutionBooleanValue(resp, x[k])) {
                x_out[k] = 1.0;
                val      += cost[k];
            }
        }
    }
    // Si INFEASIBLE (ne devrait pas arriver : x=0 toujours faisable),
    // on retourne la solution nulle par défaut.

    return {val, std::move(x_out)};
}
