#pragma once
/*
 * gnn_v1cld_sd_edit_native_batched.hpp — Équivalent V1CLD de
 * GNNSdEditNativeBatched (gnn_sd_edit_native_batched.hpp) : B nœuds B&B
 * bornés en un seul forward GNN, mais avec λ[i,j,k] (3D, un axe contrainte
 * de plus) au lieu de λ[i,k] (2D) — cf. gnn_sd_edit/train_v1CLD_sd_edit.py.
 *
 * Fichier séparé de gnn_sd_edit_native_batched.hpp/.cpp — AUCUNE modification
 * du backend V1LD de production (même convention que le reste du repo).
 *
 * ⚠️ Hypothèse héritée de GNNSdEditNativeBatched, PAS revalidée séparément
 * ici : appelé uniquement pendant la phase de branchement sur x (avant tout
 * branchement sur z) — z_mask est donc toujours entièrement libre (N_z = N,
 * n_fixed_z1 = 0), comme le backend V1LD le suppose déjà en dur. Si ce
 * backend est un jour appelé après un branchement z partiel, cette hypothèse
 * devra être revue (cf. build_graph de train_v1CLD_sd_edit.py, qui lui gère
 * le cas général free_z ⊊ [N]).
 */

#include "gnn_bound_backend_batch.hpp"
#include "../../solver_generic/types.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class GNNV1CLDSdEditNativeBatched : public GNNBoundBackendBatch {
public:
    GNNV1CLDSdEditNativeBatched(const std::string& checkpoint_path, const std::string& device,
                                int n, int m, int N, double epsilon,
                                const Vec1d& c, const Vec1d& p, const Vec3d& A, const Vec2d& b,
                                bool use_surrogate_zeta = false);

    ~GNNV1CLDSdEditNativeBatched() override;

    GNNV1CLDSdEditNativeBatched(const GNNV1CLDSdEditNativeBatched&)            = delete;
    GNNV1CLDSdEditNativeBatched& operator=(const GNNV1CLDSdEditNativeBatched&) = delete;

    void query_batch(const std::vector<std::vector<int8_t>>& x_masks,
                      const std::vector<double>& curr_costs,
                      const std::vector<Vec2d>& b_res_list, int depth,
                      std::vector<double>& ub_relative_out,
                      GNNBatchCallTiming& timing) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
