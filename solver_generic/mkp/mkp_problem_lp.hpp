#pragma once
/*
 * mkp_problem_lp.hpp — Variante LP-relax de MKPProblem pour benchmarks.
 *
 * solve_zeta(i, cost) utilise solve_zeta_i_packing_lp (mkp_subproblems.hpp)
 * sur chaque contrainte j indépendamment, et retourne le minimum :
 *
 *   ζ_i^LP = min_j  LP_j(cost, A[i][j], b[i][j])
 *
 * C'est une borne supérieure valide sur la vraie ζ_i (solution entière
 * multi-dimensionnelle), car la LP à j contrainte domine la vraie solution.
 * Comme ζ_i^LP ≥ ζ_i^exact, δ_i^LP ≤ δ_i^exact → la borne v1LD est plus lâche.
 *
 * Utilisé pour comparer la qualité et la vitesse de convergence de v1LD selon
 * la précision du solveur de ζ_i.
 */

#include "mkp_problem.hpp"
#include "mkp_subproblems.hpp"

#include <limits>

class MKPProblemLP : public MKPProblem {
public:
    using MKPProblem::MKPProblem;   // même constructeur

    std::pair<double, Vec1d> solve_zeta(int i, const Vec1d& cost) const override
    {
        const Vec2d& Ai = A_[i];
        const Vec1d& bi = b_[i];

        double best_val = std::numeric_limits<double>::max();
        Vec1d  best_x;

        // Pour chaque contrainte j, résout la LP fractionnaire 1D
        // Retourne le min (borne LP la plus serrée sur la multi-dim)
        for (int j = 0; j < m_; ++j) {
            auto [val, x] = solve_zeta_i_packing_lp(cost, Ai[j], bi[j]);
            if (val < best_val) {
                best_val = val;
                best_x   = std::move(x);
            }
        }

        return {best_val, std::move(best_x)};
    }
};
