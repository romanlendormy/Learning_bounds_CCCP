#pragma once
#include "dld_base.hpp"

/*
 * dld_solver_parallel.hpp — Variante EXPÉRIMENTALE de solve_dld, avec la
 * boucle (B) (θ_{i,j}/ζ_{i,j}) parallélisée par OpenMP.
 *
 * Créée pour la campagne de benchmark SLD vs DLD sur SSP
 * (test_paper_trillium/ssp/, cf. solver_bab/ssp/campaign_sld_dld_vs_ref.cpp)
 * — objectif : threader ζ/θ pour DLD comme c'est déjà fait pour SLD
 * (solver_generic/sld_solver.cpp, #pragma omp parallel for) et pour le
 * calcul zeta_theta côté B&B (solver_bab/ssp/gnn_dld_native.cpp).
 *
 * Copie DÉLIBÉRÉMENT séparée de dld_solver.cpp (fonction renommée
 * solve_dld_parallel, pas de collision de symbole dans sld_generic_core) :
 * dld_solver.cpp est partagé avec MKP et d'autres scripts SSP déjà établis
 * (debug_root_bound, bench_ssp_bab_file, test_gnn_dld_bab en mode dld,
 * etc.) — on ne veut pas risquer d'y introduire une régression pour cette
 * seule campagne. Si la parallélisation est validée comme correcte et utile,
 * elle pourra être fusionnée dans dld_solver.cpp plus tard.
 *
 * Paramètres : identiques à solve_dld (solver_generic/dld_solver.hpp).
 */
DLDResult solve_dld_parallel(
    DLDProblem&    problem,
    int              T            = 500,
    int              log_interval = 1,
    bool             verbose      = false,
    double           time_limit_s = -1.0,
    double           rel_tol      = -1.0,
    int              tol_window   = 500,
    double           v_lower      = -1e18,
    const Vec3d&     lambda_init  = {}
);
