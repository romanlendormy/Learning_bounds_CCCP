#pragma once
#include "v1cld_base.hpp"

/*
 * v1cld_solver_parallel.hpp — Variante EXPÉRIMENTALE de solve_v1cld, avec la
 * boucle (B) (θ_{i,j}/ζ_{i,j}) parallélisée par OpenMP.
 *
 * Créée pour la campagne de benchmark V1LD vs V1CLD sur SSP
 * (test_paper_trillium/ssp/, cf. solver_bab/ssp/campaign_v1ld_v1cld_vs_ref.cpp)
 * — objectif : threader ζ/θ pour V1CLD comme c'est déjà fait pour V1LD
 * (solver_generic/v1ld_solver.cpp, #pragma omp parallel for) et pour le
 * calcul zeta_theta côté B&B (solver_bab/ssp/gnn_v1cld_native.cpp).
 *
 * Copie DÉLIBÉRÉMENT séparée de v1cld_solver.cpp (fonction renommée
 * solve_v1cld_parallel, pas de collision de symbole dans v1ld_generic_core) :
 * v1cld_solver.cpp est partagé avec MKP et d'autres scripts SSP déjà établis
 * (debug_root_bound, bench_ssp_bab_file, test_gnn_v1cld_bab en mode v1cld,
 * etc.) — on ne veut pas risquer d'y introduire une régression pour cette
 * seule campagne. Si la parallélisation est validée comme correcte et utile,
 * elle pourra être fusionnée dans v1cld_solver.cpp plus tard.
 *
 * Paramètres : identiques à solve_v1cld (solver_generic/v1cld_solver.hpp).
 */
V1CLDResult solve_v1cld_parallel(
    V1CLDProblem&    problem,
    int              T            = 500,
    int              log_interval = 1,
    bool             verbose      = false,
    double           time_limit_s = -1.0,
    double           rel_tol      = -1.0,
    int              tol_window   = 500,
    double           v_lower      = -1e18,
    const Vec3d&     lambda_init  = {}
);
