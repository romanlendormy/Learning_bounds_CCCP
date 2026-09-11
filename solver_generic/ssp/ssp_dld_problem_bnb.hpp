#pragma once
/*
 * ssp_dld_problem_bnb.hpp — SSPDLDProblemBnB : variante masquée de SSPDLDProblem
 *                            pour la borne DLD aux nœuds d'un B&B Gecode.
 *
 * Même principe que ssp_problem_bnb.hpp (SLD), appliqué à la décomposition
 * DLD : mask[e][t] fixe le symbole à (e,t) si ≥ 0. Un employé déjà fixé à
 * la période-contrainte j n'est plus candidat au tri glouton de solve_zeta —
 * sa couverture éventuelle (W) réduit d'autant la demande résiduelle des
 * employés encore libres à j (cf. solve_zeta_masked dans ssp_dld_problem.hpp).
 *
 * mask == tout -1 ⟺ comportement identique à SSPDLDProblem.
 */

#include "ssp_dld_problem.hpp"

class SSPDLDProblemBnB : public SSPDLDProblem {
public:
    SSPDLDProblemBnB(
        const std::vector<std::tuple<int,int,int>>& transitions,
        const std::vector<int>&  accepting,
        int                      initial,
        const Vec1d&             salary,
        const Vec2i&             demand,
        Vec1d                    p,
        double                   epsilon,
        int                      E,
        Vec2i                    mask   // (E, T) : -1 = libre, sinon symbole imposé
    )
        : SSPDLDProblem(transitions, accepting, initial, salary, demand,
                        std::move(p), epsilon, E),
          mask_(std::move(mask))
    {}

    std::pair<double, Vec1d> solve_theta(int j, const Vec1d& cost) const override {
        return solve_theta_masked(j, cost, &mask_);
    }
    std::pair<double, Vec1d> solve_zeta(int i, int j, const Vec1d& cost) const override {
        return solve_zeta_masked(i, j, cost, &mask_);
    }

private:
    Vec2i mask_;
};
