#pragma once
#include "dld_base.hpp"

/*
 * Descente sous-gradient générique pour DLD.
 *
 * DLD est la version décomposée de SLD : les multiplicateurs λ_{i,j} ∈ R^d
 * sont indexés par (scénario i, contrainte j) plutôt que par scénario seul.
 * Cela permet de décomposer chaque sous-problème scénario en m sous-problèmes
 * indépendants (un par contrainte j).
 *
 * Paramètres
 * ----------
 * problem       : instance de DLDProblem
 * T             : nombre maximal d'itérations
 * log_interval  : fréquence d'enregistrement (0 = désactivé)
 * verbose       : affiche L1 tous les 50 pas
 * time_limit_s  : limite de temps (-1 = illimitée)
 * rel_tol       : tolérance relative pour arrêt anticipé (-1 = désactivée)
 * tol_window    : fenêtre (en itérations) pour rel_tol
 * v_lower       : borne inférieure sur v* pour le pas de Polyak (-1e18 → 1/√t)
 * lambda_init   : λ initial (N × m × d) ; vide → zéros
 * target_bound  : arrêt anticipé dès que best_upper <= target_bound (échelle
 *                relative L1, pas fp) — cf. doc symétrique sur solve_sld
 *                (sld_solver.hpp). 1e18 (défaut) = désactivé.
 */
DLDResult solve_dld(
    DLDProblem&    problem,
    int              T            = 500,
    int              log_interval = 1,
    bool             verbose      = false,
    double           time_limit_s = -1.0,
    double           rel_tol      = -1.0,
    int              tol_window   = 500,
    double           v_lower      = -1e18,
    const Vec3d&     lambda_init  = {},
    double           target_bound = 1e18
);
