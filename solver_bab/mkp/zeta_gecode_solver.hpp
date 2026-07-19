#pragma once
/*
 * zeta_gecode_solver.hpp — Résout ζ_i (MKP déterministe, un scénario) via un
 * espace de recherche Gecode natif, EXACTEMENT (pas d'approximation).
 *
 * Idée du tuteur testée ici : modéliser x∈{0,1}^n + les m contraintes de
 * capacité comme un espace Gecode standard (filtrage/propagation natif,
 * rapide) et ne compter sur le branchement QUE pour le résidu que la
 * propagation seule ne peut élaguer. Pour que le filtrage soit réellement
 * efficace (la propagation borne-cohérence native de linear() est plus
 * faible qu'une vraie relaxation LP pour ce type de contrainte), la MÊME
 * borne surrogate que MKPProblemBnBSurrogate (cf. mkp_surrogate_bound.hpp)
 * est injectée à chaque nœud via un Brancher custom qui poste
 * rel(obj <= ub) — même mécanisme que BoundBrancher dans gecode_bab.hpp,
 * appliqué ici au sous-problème ζ_i lui-même plutôt qu'au B&B englobant.
 *
 * Comparaison scientifiquement propre avec MKPProblemBnBSurrogate : MÊME
 * borne, MÊME ordre de branchement (ratio décroissant) — seul le moteur de
 * recherche diffère (récursion C++ maison vs Gecode::BAB<> natif).
 */

#include "../../solver_generic/types.hpp"
#include <utility>

// n : nombre total d'items ORIGINAUX (pour dimensionner x_out) ; Ai (m,n),
// bi (m,), cost (n,) : mêmes conventions que MKPProblem::solve_zeta.
std::pair<double, Vec1d> solve_zeta_gecode(
    int n, int m, const Vec2d& Ai, const Vec1d& bi, const Vec1d& cost);
