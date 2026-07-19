#pragma once
/*
 * gnn_sd_edit_native_aoti_batched.hpp — Borne GNN batchée (cf.
 * gnn_sd_edit_native_batched.hpp), backend AOTInductor au lieu de
 * TorchScript.
 *
 * Fichier séparé de gnn_sd_edit_native_batched.hpp/.cpp — même convention
 * que la séparation TorchScript/AOTI déjà établie pour le backend NON batché
 * (cf. gnn_sd_edit_native_aoti.hpp) : artefacts de recherche comparés
 * indépendamment, aucune abstraction commune pour ne pas risquer l'un en
 * modifiant l'autre. Construction de graphe/cache active_x/cache d'arêtes
 * dupliqués mot pour mot depuis gnn_sd_edit_native_batched.cpp — seule la
 * section chargement/appel du modèle change.
 *
 * Différence structurelle avec le backend TorchScript batché : l'artefact
 * AOTInductor est COMPILÉ EN AVANCE pour n_g FIGÉ (comme le backend AOTI non
 * batché) — un artefact par valeur de n_g rencontrée, chargé via
 * torch::inductor::AOTIModelPackageLoader. B (taille du batch de nœuds), en
 * revanche, reste DYNAMIQUE À L'INTÉRIEUR de chaque artefact — feasibility
 * validée le 2026-07-17 (cf. gnn_sd_edit/probe_aoti_batched_dynamic_B.py) :
 * contrairement à n_g (~n valeurs, une par profondeur), B varie quasi
 * continûment (1..K par flush de bucket) — un export par (n_g,B) aurait été
 * rédhibitoire. Le modèle Python exporté (GNNSdEditScriptableBatchedAOTI,
 * gnn_sd_edit/gnn_sd_edit_scriptable_batched.py) dérive B de
 * context.shape[0] au lieu de le recevoir en argument scalaire séparé —
 * requis pour que torch.export/aoti_compile_and_package n'échoue pas (cf.
 * docstring de cette classe Python).
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

class GNNSdEditNativeAOTIBatched : public GNNBoundBackendBatch {
public:
    // use_surrogate_zeta : cf. doc sur GNNSdEditNativeBatched
    // (gnn_sd_edit_native_batched.hpp) — même sémantique, même défaut.
    GNNSdEditNativeAOTIBatched(const std::string& checkpoint_path, const std::string& device,
                               int n, int m, int N, double epsilon,
                               const Vec1d& c, const Vec1d& p, const Vec3d& A, const Vec2d& b,
                               bool use_surrogate_zeta = false);

    ~GNNSdEditNativeAOTIBatched() override;

    GNNSdEditNativeAOTIBatched(const GNNSdEditNativeAOTIBatched&)            = delete;
    GNNSdEditNativeAOTIBatched& operator=(const GNNSdEditNativeAOTIBatched&) = delete;

    void query_batch(const std::vector<std::vector<int8_t>>& x_masks,
                      const std::vector<double>& curr_costs,
                      const std::vector<Vec2d>& b_res_list, int depth,
                      std::vector<double>& ub_relative_out,
                      GNNBatchCallTiming& timing) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
