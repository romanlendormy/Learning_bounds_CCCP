#pragma once
/*
 * gnn_bound_backend_batch.hpp — Interface BATCHÉE, séparée de
 * gnn_bound_backend.hpp (n'y touche pas — aucun risque pour le backend
 * mono-nœud de production). Un appel = B nœuds B&B bornés en UN SEUL
 * forward GNN (cf. plan de batching, Étape 2 — [[project_bucketed_bab_risk_result]],
 * [[project_gnn_batch_forward_probe_result]]).
 *
 * Contrainte imposée par l'appelant (garantie par le bucketing par n_g de
 * l'Étape 1, gecode_bab_bucketed.cpp) : tous les x_masks d'un même appel
 * DOIVENT partager le même n_g (nombre d'items encore libres) — l'implémentation
 * vérifie cette hypothèse et lève une exception si elle est violée, plutôt que
 * de calculer une borne silencieusement fausse.
 */

#include "../../solver_generic/types.hpp"

#include <cstdint>
#include <vector>

struct GNNBatchCallTiming {
    double t_graph_build_s   = 0.0;
    double t_forward_s       = 0.0;
    double t_bound_compute_s = 0.0;
};

class GNNBoundBackendBatch {
public:
    virtual ~GNNBoundBackendBatch() = default;

    // x_masks : B masques (x_masks[b][k] ∈ {0,1,2}, taille n) ; curr_costs :
    // B profits déjà fixés (fp) ; b_res_list : B capacités résiduelles déjà
    // à jour (MAINTENUES INCRÉMENTALEMENT par l'appelant — cf.
    // MKPSpaceGNNBatched::b_res, gecode_bab_gnn_batched.hpp — PLUS
    // reconstruites depuis x_mask à chaque appel, cf. échange du 2026-07-17) ;
    // depth partagé par construction (même n_g pour tout le batch → même
    // depth au sens more()). Remplit ub_relative_out (même taille que
    // x_masks, ub_relative_out[b] = L1(λ) du sous-problème résiduel du
    // masque b — l'appelant calcule ub_b = curr_costs[b] + ub_relative_out[b],
    // comme GNNBoundBackend::query()).
    virtual void query_batch(const std::vector<std::vector<int8_t>>& x_masks,
                              const std::vector<double>& curr_costs,
                              const std::vector<Vec2d>& b_res_list, int depth,
                              std::vector<double>& ub_relative_out,
                              GNNBatchCallTiming& timing) = 0;
};
