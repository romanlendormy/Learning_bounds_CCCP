#pragma once
/*
 * gnn_v1cld_sd_edit_native.hpp — Adaptateur mono-nœud (GNNBoundBackend::query())
 * autour de GNNV1CLDSdEditNativeBatched (query_batch, interface batchée) —
 * cf. gnn_v1cld_sd_edit_native_batched.hpp.
 *
 * Motivation : le moteur root-only (bound_mode=GNN_ROOT_V1CLD, gecode_bab.hpp,
 * symétrique de GNN_ROOT côté V1LD) n'appelle que l'interface GNNBoundBackend
 * (un nœud à la fois, cf. gnn_bound_backend.hpp) — mais le seul backend natif
 * V1CLD existant (GNNV1CLDSdEditNativeBatched) implémente l'interface batchée
 * GNNBoundBackendBatch (query_batch, utilisée par gecode_bab_v1cld_gnn_batched.hpp).
 * Cette classe fait juste le pont : un appel à query() = un appel à
 * query_batch() avec des vecteurs de taille 1. Aucune nouvelle logique GNN.
 *
 * Utilisée UNE SEULE FOIS par résolution (racine seulement, cf. GNN_ROOT_V1CLD),
 * donc le surcoût de construire des std::vector de taille 1 à chaque appel
 * est négligeable.
 */

#include "gnn_bound_backend.hpp"
#include "gnn_v1cld_sd_edit_native_batched.hpp"
#include "../../solver_generic/types.hpp"

#include <memory>
#include <string>
#include <vector>

class GNNV1CLDSdEditNative : public GNNBoundBackend {
public:
    GNNV1CLDSdEditNative(const std::string& checkpoint_path, const std::string& device,
                         int n, int m, int N, double epsilon,
                         const Vec1d& c, const Vec1d& p, const Vec3d& A, const Vec2d& b,
                         bool use_surrogate_zeta = false);

    ~GNNV1CLDSdEditNative() override;

    GNNV1CLDSdEditNative(const GNNV1CLDSdEditNative&)            = delete;
    GNNV1CLDSdEditNative& operator=(const GNNV1CLDSdEditNative&) = delete;

    double query(const std::vector<int8_t>& x_mask, const Vec2d& b_res,
                 double curr_cost, int depth, GNNCallTiming& timing) override;

private:
    std::unique_ptr<GNNV1CLDSdEditNativeBatched> batched_;
};
