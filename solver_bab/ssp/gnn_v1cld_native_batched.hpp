#pragma once
/*
 * gnn_v1cld_native_batched.hpp — Borne GNN v1CLD batchée (Étape B2 du
 * batching multi-nœuds SSP), mirror structurel de
 * solver_bab/mkp/gnn_sd_edit_native_batched.hpp (MKP).
 *
 * Implémente SSPGNNBoundBackendBatch (gecode_bab_ssp_gnn_batched.hpp) : UN
 * appel GNNSspScriptableBatched::forward() par bucket flush, au lieu d'un
 * appel par nœud (GNNV1CLDNative::query, gnn_v1cld_native.hpp -- mode
 * séquentiel, toujours utilisé tel quel par ailleurs).
 *
 * Contrat imposé par le bucketing (G, N_z) côté driver (SSPSpaceGNNBatched) :
 * tous les membres d'un même appel query_batch() partagent EXACTEMENT le
 * même G et le même N_z -- vérifié en interne (lève sinon).
 *
 * pImpl : ce header ne dépend pas de <torch/script.h> (même raison que
 * gnn_v1cld_native.hpp).
 */

#include "gecode_bab_ssp_gnn_batched.hpp"
#include "../../solver_generic/ssp/ssp_loader.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct SSPGNNBatchQueryTiming {
    double  t_graph_build_s     = 0.0;  // features (par membre) + arêtes (une fois/appel)
    double  t_forward_s         = 0.0;
    double  t_zeta_theta_s      = 0.0;  // somme sur les B membres
    double  t_lagrangian_rest_s = 0.0;  // somme sur les B membres
    int64_t n_batch_calls       = 0;
    int64_t n_nodes_total       = 0;    // somme des B sur tous les appels
};

class GNNV1CLDNativeBatched : public SSPGNNBoundBackendBatch {
public:
    GNNV1CLDNativeBatched(const std::string& checkpoint_path, const std::string& device,
                          const SSPInstance& inst);
    ~GNNV1CLDNativeBatched() override;

    GNNV1CLDNativeBatched(const GNNV1CLDNativeBatched&)            = delete;
    GNNV1CLDNativeBatched& operator=(const GNNV1CLDNativeBatched&) = delete;

    void query_batch(const std::vector<std::vector<int8_t>>& x_masks,
                      const std::vector<std::vector<int8_t>>& z_masks,
                      const std::vector<double>& curr_costs,
                      const std::vector<int>& depths,
                      std::vector<double>& cost_lower_out) override;

    SSPGNNBatchQueryTiming cumulative_timing() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
