#pragma once
/*
 * gecode_bab_bucketed.hpp — Variante EXPÉRIMENTALE de solve_gecode_bab_mkp
 * (gecode_bab.cpp) qui groupe ("bucket") les nœuds fils par profondeur
 * d'items (n_g = n - depth) avant de les juger contre l'incumbent, au lieu
 * de les élaguer immédiatement comme le fait la boucle DFS de production.
 *
 * Objectif : mesurer, SANS toucher au calcul de la borne ni au GNN, le
 * risque algorithmique PUR d'un futur batching GNN — grouper K nœuds
 * ensemble pour les juger "en aveugle" les uns des autres (incumbent figé le
 * temps du bucket) peut faire exploser le nombre de nœuds nécessaires à la
 * preuve d'optimalité. Cette variante teste CE risque isolément, à borne
 * strictement inchangée (V1LD/V1CLD/REF, CPU, déjà validées) — le vrai
 * batching GPU du forward GNN est une suite explicite, pas implémentée ici
 * (cf. plan).
 *
 * Fichier séparé de gecode_bab.hpp/.cpp — AUCUNE modification du code de
 * production existant (convention du repo : chaque expérience/sonde est un
 * artefact isolé, cf. gnn_native_probe.cpp/gnn_cudagraph_probe.cpp).
 * Réutilise MKPSharedData/MKPSpaceV1LD/BoundMode/greedy_mkp tels quels.
 *
 * bound_mode restreint à {V1LD, V1CLD, REF} : pas de dépendance
 * LibTorch/OpenMP dans cette version (le mode GNN nécessiterait de batcher
 * le forward lui-même, hors périmètre de cette expérience).
 */

#include "gecode_bab.hpp"

struct BucketedBabResult : public GecodeV1LDResult {
    int     K                         = 1;    // taille de bucket demandée
    int64_t n_bucket_flushes          = 0;
    double  mean_bucket_fill_at_flush = 0.0;  // taille RÉELLE moyenne des buckets au flush
    int64_t max_pending_live          = 0;    // nb max de Space* en attente simultanément (tous n_g confondus)
};

// Mêmes paramètres que solve_gecode_bab_mkp (cf. gecode_bab.hpp), moins tout
// ce qui concerne le GNN (hors périmètre), plus K (taille de bucket).
// K=1 doit reproduire EXACTEMENT (nodes, fails, objective) de
// solve_gecode_bab_mkp — c'est le test de non-régression de cette variante.
BucketedBabResult solve_gecode_bab_mkp_bucketed(
    int n, int m, int N, double epsilon,
    const Vec1d& c, const Vec1d& p,
    const Vec3d& A, const Vec2d& b,
    double    time_lim,
    int       T_root,
    int       T_node,
    BoundMode bound_mode,
    bool      patience_mode,
    double    patience_tol,
    int       K);
