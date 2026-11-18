#pragma once
/*
 * gecode_bab_dld_gnn_batched.hpp — Équivalent DLD de
 * gecode_bab_gnn_batched.hpp/solve_gecode_bab_mkp_gnn_batched : même moteur
 * DFS + bucketing par n_g, même MKPSpaceGNNBatched (RÉUTILISÉE telle quelle
 * — elle est déjà bound-agnostique : post_gnn_bound(int) ne connaît que la
 * valeur scalaire de borne, jamais la forme de λ qui l'a produite), même
 * GNNBatchedBabResult (idem) — SEUL le backend GNN change
 * (GNNDLDSdEditNativeBatched au lieu de GNNSdEditNativeBatched).
 *
 * Fichier séparé de gecode_bab_gnn_batched.hpp/.cpp — AUCUNE modification du
 * driver SLD de production (même convention que tout ce dossier).
 *
 * Seul backend supporté pour l'instant : "native" (GNNDLDSdEditNativeBatched,
 * validé contre train_DLD_sd_edit.py — cf. test_gnn_dld_native_batched.cpp).
 */

#include "gecode_bab_gnn_batched.hpp"
#include "gnn_dld_sd_edit_native_batched.hpp"

GNNBatchedBabResult solve_gecode_bab_mkp_dld_gnn_batched(
    int n, int m, int N, double epsilon,
    const Vec1d& c, const Vec1d& p, const Vec3d& A, const Vec2d& b,
    double time_lim, int K,
    std::string gnn_checkpoint_path, std::string gnn_device = "cuda",
    double gnn_max_depth_frac = 1.0,
    bool use_surrogate_zeta = false,
    int gnn_node_stride = 1,
    std::string gnn_backend = "native",
    bool stop_after_first_level = false);
