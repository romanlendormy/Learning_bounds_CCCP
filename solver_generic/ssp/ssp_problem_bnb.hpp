#pragma once
/*
 * ssp_problem_bnb.hpp — SSPProblemBnB : variante masquée de SSPShiftProblem
 *                        pour la borne SLD aux nœuds d'un B&B Gecode.
 *
 * À un nœud du B&B, certaines variables x[e,t] sont déjà fixées par les
 * décisions de branchement précédentes. Au lieu de rétrécir l'instance
 * (impossible ici : un automate n'a de sens que sur la séquence complète des
 * T périodes d'un employé), on la MASQUE : mask[e][t] ∈ {-1, O, P, R, W}
 * impose le symbole à (e,t) si ≥ 0, laisse le choix libre si -1.
 *
 * θ/ζ sur le domaine masqué restent des bornes duales valides sur le
 * sous-problème du nœud (le masque restreint exactement le domaine de
 * maximisation au sous-polytope du nœud) — voir solve_theta_masked/
 * solve_zeta_masked dans ssp_problem.hpp pour la justification complète.
 *
 * mask == tout -1 (E×T rempli de -1) ⟺ comportement identique à
 * SSPShiftProblem — utilisé pour le sanity-check de non-régression.
 */

#include "ssp_problem.hpp"

class SSPProblemBnB : public SSPShiftProblem {
public:
    SSPProblemBnB(
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
        : SSPShiftProblem(transitions, accepting, initial, salary, demand,
                          std::move(p), epsilon, E),
          mask_(std::move(mask))
    {}

    std::pair<double, Vec1d> solve_theta(const Vec1d& cost) const override {
        return solve_theta_masked(cost, &mask_);
    }
    std::pair<double, Vec1d> solve_zeta(int i, const Vec1d& cost) const override {
        return solve_zeta_masked(i, cost, &mask_);
    }

private:
    Vec2i mask_;
};
