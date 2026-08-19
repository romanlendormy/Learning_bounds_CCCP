#pragma once
/*
 * gecode_bab_v1cld_gnn_batched.hpp — Équivalent V1CLD de
 * gecode_bab_gnn_batched.hpp/solve_gecode_bab_mkp_gnn_batched : même moteur
 * DFS + bucketing par n_g, même MKPSpaceGNNBatched (RÉUTILISÉE telle quelle
 * — elle est déjà bound-agnostique : post_gnn_bound(int) ne connaît que la
 * valeur scalaire de borne, jamais la forme de λ qui l'a produite), même
 * GNNBatchedBabResult (idem) — SEUL le backend GNN change
 * (GNNV1CLDSdEditNativeBatched au lieu de GNNSdEditNativeBatched).
 *
 * Fichier séparé de gecode_bab_gnn_batched.hpp/.cpp — AUCUNE modification du
 * driver V1LD de production (même convention que tout ce dossier).
 *
 * Seul backend supporté pour l'instant : "native" (GNNV1CLDSdEditNativeBatched,
 * validé contre train_v1CLD_sd_edit.py — cf. test_gnn_v1cld_native_batched.cpp).
 * Pas d'équivalent AOTInductor pour V1CLD (native_aoti/native_aoti_static)
 * pour l'instant — à ajouter seulement si le backend natif s'avère un
 * goulot après mesure, comme ça a été fait pour V1LD.
 */

#include "gecode_bab_gnn_batched.hpp"
#include "gnn_v1cld_sd_edit_native_batched.hpp"

GNNBatchedBabResult solve_gecode_bab_mkp_v1cld_gnn_batched(
    int n, int m, int N, double epsilon,
    const Vec1d& c, const Vec1d& p, const Vec3d& A, const Vec2d& b,
    double time_lim, int K,
    std::string gnn_checkpoint_path, std::string gnn_device = "cuda",
    double gnn_max_depth_frac = 1.0,
    bool use_surrogate_zeta = false,
    int gnn_node_stride = 1,
    std::string gnn_backend = "native");
