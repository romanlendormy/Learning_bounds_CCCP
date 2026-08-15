#pragma once
/*
 * gnn_sd_edit_native_aoti_batched_static.hpp — Borne GNN batchée, backend
 * AOTInductor à formes STATIQUES (n_g ET B tous deux figés à l'export),
 * contrairement à GNNSdEditNativeAOTIBatched (gnn_sd_edit_native_aoti_batched.hpp,
 * B dynamique — abandonné : bug compilateur CUDA/Inductor sur la garde de
 * forme dérivée B_dim*V, cf. mémoire "aoti-batched-gru-decomposed-abandoned").
 *
 * Fichier séparé (ne modifie ni ne réutilise GNNSdEditNativeAOTIBatched) —
 * même convention que la séparation déjà établie entre TorchScript et AOTI :
 * artefacts de recherche comparés indépendamment, aucune abstraction commune
 * pour ne pas risquer l'un en modifiant l'autre. Construction de
 * graphe/cache active_x/cache d'arêtes dupliqués mot pour mot depuis
 * gnn_sd_edit_native_batched.cpp — seule la section chargement/appel du
 * modèle change.
 *
 * Principe : un artefact .pt2 par n_g, compilé pour EXACTEMENT K graphes
 * (K = taille de bucket passée au constructeur, la même valeur que
 * l'appelant utilise pour flush_bucket). Un bucket partiel (B_réel < K,
 * survient au dernier flush d'une profondeur) est complété par du padding
 * (graphes factices à zéro) avant l'appel — sûr car la structure
 * bloc-diagonale des arêtes garantit qu'aucun graphe factice ne peut
 * influencer la sortie d'un graphe réel (aucune opération de pooling du
 * modèle ne mélange les lignes du batch — toutes les réductions sont
 * `.view(B, n_z, n_g, ...).mean/max(dim ∈ {1,2})`, jamais sur B lui-même ;
 * norm_ctx est un LayerNorm, pas un BatchNorm — aucune statistique inter-
 * lignes). Les sorties correspondant aux graphes factices sont ignorées.
 *
 * B_réel > K ne devrait jamais survenir (l'appelant flush dès que
 * bucket.size()==K, cf. gecode_bab_gnn_batched.cpp) — vérifié par exception
 * défensive plutôt que silencieusement tronqué.
 *
 * pImpl : ce header ne dépend pas de <torch/...>, même raison que les autres
 * variantes de ce dossier.
 */

#include "gnn_bound_backend_batch.hpp"
#include "../../solver_generic/types.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class GNNSdEditNativeAOTIStaticBatched : public GNNBoundBackendBatch {
public:
    // K : taille de bucket FIGÉE à l'export (DOIT correspondre exactement à
    // la taille de bucket utilisée par l'appelant, sans quoi tous les
    // artefacts .pt2 seraient à re-exporter).
    // use_surrogate_zeta : cf. doc sur GNNSdEditNativeBatched
    // (gnn_sd_edit_native_batched.hpp) — même sémantique, même défaut.
    GNNSdEditNativeAOTIStaticBatched(const std::string& checkpoint_path, const std::string& device,
                                      int n, int m, int N, double epsilon,
                                      const Vec1d& c, const Vec1d& p, const Vec3d& A, const Vec2d& b,
                                      int K, bool use_surrogate_zeta = false);

    ~GNNSdEditNativeAOTIStaticBatched() override;

    GNNSdEditNativeAOTIStaticBatched(const GNNSdEditNativeAOTIStaticBatched&)            = delete;
    GNNSdEditNativeAOTIStaticBatched& operator=(const GNNSdEditNativeAOTIStaticBatched&) = delete;

    void query_batch(const std::vector<std::vector<int8_t>>& x_masks,
                      const std::vector<double>& curr_costs,
                      const std::vector<Vec2d>& b_res_list, int depth,
                      std::vector<double>& ub_relative_out,
                      GNNBatchCallTiming& timing) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
