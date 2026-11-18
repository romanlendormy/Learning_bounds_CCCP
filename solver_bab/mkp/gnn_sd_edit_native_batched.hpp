#pragma once
/*
 * gnn_sd_edit_native_batched.hpp — Variante BATCHÉE de GNNSdEditNative
 * (gnn_sd_edit_native.hpp) : B nœuds B&B bornés en un seul forward GNN au
 * lieu de B appels séparés (cf. plan de batching, Étape 2d/2e).
 *
 * Fichier séparé de gnn_sd_edit_native.hpp/.cpp — AUCUNE modification du
 * backend de production (même choix de duplication que le repo pratique déjà
 * entre variantes homogène/sd_edit, cf. commentaire de gnn_sd_edit_native.hpp).
 * pImpl : ce header ne dépend pas de <torch/script.h> (inclus seulement dans
 * le .cpp), même raison que GNNSdEditNative.
 */

#include "gnn_bound_backend_batch.hpp"
#include "../../solver_generic/types.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class GNNSdEditNativeBatched : public GNNBoundBackendBatch {
public:
    // use_surrogate_zeta : ζ_i (résolu APRÈS la prédiction λ du GNN, pour
    // calculer L1(λ)) via MkpBnBSurr (borne agrégée, bnb_solver.cpp) au
    // lieu de MkpBnB (min de m relaxations séparées, historique). Même valeur
    // ζ_i exacte dans les deux cas — validé ×2.25 plus rapide dans le B&B
    // SLD séquentiel équivalent (gecode_bab.hpp), cf. mémoire
    // project_compare_sld_zeta_bnb_surrogate. Défaut false = comportement
    // historique inchangé.
    GNNSdEditNativeBatched(const std::string& checkpoint_path, const std::string& device,
                           int n, int m, int N, double epsilon,
                           const Vec1d& c, const Vec1d& p, const Vec3d& A, const Vec2d& b,
                           bool use_surrogate_zeta = false);

    ~GNNSdEditNativeBatched() override;

    GNNSdEditNativeBatched(const GNNSdEditNativeBatched&)            = delete;
    GNNSdEditNativeBatched& operator=(const GNNSdEditNativeBatched&) = delete;

    void query_batch(const std::vector<std::vector<int8_t>>& x_masks,
                      const std::vector<double>& curr_costs,
                      const std::vector<Vec2d>& b_res_list, int depth,
                      std::vector<double>& ub_relative_out,
                      GNNBatchCallTiming& timing) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
