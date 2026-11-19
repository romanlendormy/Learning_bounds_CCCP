#pragma once
/*
 * gnn_ccpo_native_batched.hpp — backend GNN CCPO natif (libtorch/TorchScript),
 * batché. Miroir de
 * mkp/gnn_sd_edit_native_batched.{hpp,cpp} : construction du graphe 100 % C++
 * (ccpo_graph_cpp.hpp, validé 24/24 vs ccpo_graph.py), forward = UN appel
 * TorchScript pour tout le batch (module exporté par
 * gnn_ccpo/ccpo_gnn_scriptable_batched.py).
 *
 * Header sans libtorch (pimpl) pour pouvoir être inclus depuis les TU Gecode.
 */
#include "ccpo_gnn_backend_batch.hpp"

#include <memory>
#include <string>

class CCPOGNNNativeBatched : public CCPOGNNBackendBatch {
public:
    // scripted_path : .scripted_batched.pt ; torch_threads : threads intra-op
    // du forward (attention : partage libgomp avec l'évaluation OpenMP, cf.
    // piège documenté -- le driver réaffirme omp_set_num_threads avant chaque évaluation).
    CCPOGNNNativeBatched(const std::string& scripted_path, int torch_threads,
                         const std::string& device = "cpu");  // "cpu" | "cuda"
    ~CCPOGNNNativeBatched() override;

    void query_batch(const CCPOInstanceCtx& ctx,
                     const std::vector<const CCPOQuery*>& batch,
                     std::vector<std::vector<std::vector<double>>>& lambda_out) override;

    // Chronométrage cumulé (secondes) des sous-étapes de query_batch.
    double t_graph_s = 0.0;    // features + contexte + assemblage des tenseurs
    double t_forward_s = 0.0;  // module.forward
    double t_post_s = 0.0;     // remise en forme de λ
    long long n_queries = 0, n_members = 0;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
