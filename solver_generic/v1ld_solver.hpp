#pragma once
#include "v1ld_base.hpp"

/*
 * Corps générique de l'algorithme v1LD par descente de sous-gradient.
 *
 * Paramètres
 * ----------
 * problem      : instance de V1LDProblem (dispatch virtuel)
 * T            : nombre maximal d'itérations
 * log_interval : fréquence d'enregistrement (0 = désactivé)
 * verbose      : affiche L1 tous les 50 pas
 * time_limit_s : limite de temps (-1 = illimitée)
 * rel_tol      : tolérance relative pour arrêt anticipé (-1 = désactivée)
 * tol_window   : fenêtre pour rel_tol
 * v_lower      : borne inférieure pour le pas de Polyak (-1e18 → 1/√t)
 * lambda_init  : λ initial (N × n_vars) ; vide → zéros
 */
V1LDResult solve_v1ld(
    V1LDProblem&    problem,
    int             T            = 500,
    int             log_interval = 1,
    bool            verbose      = false,
    double          time_limit_s = -1.0,
    double          rel_tol      = -1.0,
    int             tol_window   = 500,
    double          v_lower      = -1e18,
    const Vec2d&    lambda_init  = {}
);
