#pragma once
/*
 * gecode_bab_gnn_batched_zaware.hpp — Variante Z-AWARE de
 * solve_gecode_bab_mkp_gnn_batched (gecode_bab_gnn_batched.hpp) : extrait
 * aussi l'état de z (pas seulement x) à chaque nœud, bucket par (n_g, N_z)
 * au lieu de n_g seul, et transmet z_mask au backend GNNSdEditNativeBatchedZAware
 * pour exclure les scénarios déjà résolus par la propagation CP de la somme
 * lagrangienne — cf. gnn_sd_edit_native_batched_zaware.hpp pour la
 * justification mathématique complète (exclusion exacte, pas approximation).
 *
 * Réutilise MKPSpaceGNNBatched/GNNBatchedBabResult TELS QUELS (déjà
 * bound-agnostiques, cf. leur propre doc) via #include de
 * gecode_bab_gnn_batched.hpp — AUCUNE modification de ce fichier ni de
 * gnn_sd_edit_native_batched.hpp/.cpp (le backend/driver d'origine restent
 * intacts et continuent de tourner exactement comme avant : retour en
 * arrière trivial = ne pas utiliser ce binaire/mode).
 *
 * Seul backend supporté : "native" (GNNSdEditNativeBatchedZAware) — pas de
 * variante AOTInductor z-aware pour l'instant (même choix que le portage V1CLD).
 */

#include "gecode_bab_gnn_batched.hpp"
#include "gnn_sd_edit_native_batched_zaware.hpp"

GNNBatchedBabResult solve_gecode_bab_mkp_gnn_batched_zaware(
    int n, int m, int N, double epsilon,
    const Vec1d& c, const Vec1d& p, const Vec3d& A, const Vec2d& b,
    double time_lim, int K,
    std::string gnn_checkpoint_path, std::string gnn_device = "cuda",
    double gnn_max_depth_frac = 1.0,
    bool use_surrogate_zeta = false,
    int gnn_node_stride = 1);
