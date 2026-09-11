#pragma once
#include "sld_base.hpp"

/*
 * Corps générique de l'algorithme SLD par descente de sous-gradient.
 *
 * Paramètres
 * ----------
 * problem      : instance de SLDProblem (dispatch virtuel)
 * T            : nombre maximal d'itérations
 * log_interval : fréquence d'enregistrement (0 = désactivé)
 * verbose      : affiche L1 tous les 50 pas
 * time_limit_s : limite de temps (-1 = illimitée)
 * rel_tol      : tolérance relative pour arrêt anticipé (-1 = désactivée)
 * tol_window   : fenêtre pour rel_tol (blocs)
 * v_lower      : borne inférieure pour le pas de Polyak (-1e18 → 1/√t)
 * lambda_init  : λ initial (N × n_vars) ; vide → zéros
 * abs_tol      : arrêt si amélioration absolue de best_upper < abs_tol pendant
 *               abs_patience itérations consécutives  (-1 = désactivé)
 * abs_patience : nombre d'itérations stagnantes avant arrêt anticipé
 * target_bound : arrêt anticipé DÈS QUE best_upper <= target_bound (échelle
 *               relative L1, PAS fp — l'appelant doit soustraire son propre
 *               offset avant d'appeler). 1e18 (défaut) = désactivé, jamais
 *               atteint en pratique. Ajouté pour comparer le sous-gradient à
 *               une borne GNN déjà connue sans payer les itérations
 *               post-croisement (cf. gecode_bab.hpp::MKPSharedData::gnn_compare_bound,
 *               mémoire project_ccmkp_trillium_migration — n=25/30, le
 *               sous-gradient SLD/ζ exact devient ~500x plus cher par
 *               itération qu'à n=15, donc courir jusqu'à convergence complète
 *               ou jusqu'à T_root=3000 est impraticable).
 */
SLDResult solve_sld(
    SLDProblem&    problem,
    int             T            = 500,
    int             log_interval = 1,
    bool            verbose      = false,
    double          time_limit_s = -1.0,
    double          rel_tol      = -1.0,
    int             tol_window   = 500,
    double          v_lower      = -1e18,
    const Vec2d&    lambda_init  = {},
    double          abs_tol      = -1.0,
    int             abs_patience = 50,
    double          target_bound = 1e18
);
