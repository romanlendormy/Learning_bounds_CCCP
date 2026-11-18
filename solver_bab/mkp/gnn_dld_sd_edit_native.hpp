#pragma once
/*
 * gnn_dld_sd_edit_native.hpp — Adaptateur mono-nœud (GNNBoundBackend::query())
 * autour de GNNDLDSdEditNativeBatched (query_batch, interface batchée) —
 * cf. gnn_dld_sd_edit_native_batched.hpp.
 *
 * Motivation : le moteur root-only (bound_mode=GNN_ROOT_DLD, gecode_bab.hpp,
 * symétrique de GNN_ROOT côté SLD) n'appelle que l'interface GNNBoundBackend
 * (un nœud à la fois, cf. gnn_bound_backend.hpp) — mais le seul backend natif
 * DLD existant (GNNDLDSdEditNativeBatched) implémente l'interface batchée
 * GNNBoundBackendBatch (query_batch, utilisée par gecode_bab_dld_gnn_batched.hpp).
 * Cette classe fait juste le pont : un appel à query() = un appel à
 * query_batch() avec des vecteurs de taille 1. Aucune nouvelle logique GNN.
 *
 * Utilisée UNE SEULE FOIS par résolution (racine seulement, cf. GNN_ROOT_DLD),
 * donc le surcoût de construire des std::vector de taille 1 à chaque appel
 * est négligeable.
 */

#include "gnn_bound_backend.hpp"
#include "gnn_dld_sd_edit_native_batched.hpp"
#include "../../solver_generic/types.hpp"

#include <memory>
#include <string>
#include <vector>

class GNNDLDSdEditNative : public GNNBoundBackend {
public:
    GNNDLDSdEditNative(const std::string& checkpoint_path, const std::string& device,
                         int n, int m, int N, double epsilon,
                         const Vec1d& c, const Vec1d& p, const Vec3d& A, const Vec2d& b,
                         bool use_surrogate_zeta = false);

    ~GNNDLDSdEditNative() override;

    GNNDLDSdEditNative(const GNNDLDSdEditNative&)            = delete;
    GNNDLDSdEditNative& operator=(const GNNDLDSdEditNative&) = delete;

    double query(const std::vector<int8_t>& x_mask, const Vec2d& b_res,
                 double curr_cost, int depth, GNNCallTiming& timing) override;

private:
    std::unique_ptr<GNNDLDSdEditNativeBatched> batched_;
};
