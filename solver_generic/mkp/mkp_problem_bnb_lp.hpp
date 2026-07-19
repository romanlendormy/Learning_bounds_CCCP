#pragma once
/*
 * mkp_problem_bnb_lp.hpp — Variante B&B de MKPProblem : borne = relaxation
 * LP JOINTE (pas surrogate, pas per-dimension), résolue via GLOP (OR-Tools),
 * avec réutilisation de l'objet LinearProgram/LPSolver d'un nœud à l'autre
 * (warm-start — cf. discussion : GLOP documente une "Incrementality" entre
 * appels Solve() successifs sur le même objet légèrement modifié).
 *
 * Même squelette de branchement que MKPProblemBnBSurrogate (même ordre —
 * réutilise prepare_mkp_surrogate — pour isoler l'effet de la BORNE de celui
 * de l'ordre de branchement), même préfiltrage (items libres/coût≤0).
 *
 * Contrairement au B&B maison (borne calculée à la main), ici la faisabilité
 * ET la borne sont déléguées à GLOP : fixer une variable qui rend le
 * problème infaisable est détecté par le statut de Solve(), pas par une
 * vérification manuelle — comportement standard d'un B&B basé sur LP.
 */

#include "mkp_problem.hpp"

class MKPProblemBnBLP : public MKPProblem {
public:
    using MKPProblem::MKPProblem;   // même constructeur

    std::pair<double, Vec1d> solve_zeta(int i, const Vec1d& cost) const override;
};
