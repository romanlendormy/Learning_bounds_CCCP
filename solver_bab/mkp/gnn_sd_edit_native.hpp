#pragma once
/*
 * gnn_sd_edit_native.hpp — Borne GNN one-shot en-process, variante sd_edit
 * (arêtes inter-scénarios en étoile), sans IPC.
 *
 * Fichier séparé de gnn_v1ld_native.hpp/.cpp (variante "homogene", clique
 * scénario-scénario) — même s'il duplique une bonne partie de la logique,
 * la séparation est volontaire : les deux variantes restent des artefacts de
 * recherche comparés indépendamment, et on ne veut pas qu'une modification
 * de l'une (features, ζ, export) affecte accidentellement l'autre via une
 * abstraction commune.
 *
 * Différence unique avec gnn_v1ld_native.cpp : build_graph_cpp() construit
 * les arêtes scénario-scénario en étoile (ancrée au scénario 0, poids 1,
 * O(n_g·N_z) arêtes) au lieu de la clique complète (O(n_g·N_z²)) — cf.
 * gnn_sd_edit/train_v1LD_sd_edit.py:265-277 et solver_swann_darius.cpp:435-442.
 * Les features de nœuds (7m+5), lagrangian_v1ld_cpp() (topologie-agnostique,
 * ne lit que A_g/b_res_g/c_g/lam) et l'export TorchScript sont identiques à
 * la variante homogène — gnn_v1ld_scriptable.py::export() n'a aucune
 * dépendance à la topologie (il prend des tenseurs bruts), donc réutilisable
 * tel quel pour un checkpoint entraîné sur des graphes sd_edit.
 *
 * pImpl : ce header ne dépend PAS de <torch/script.h> (inclus seulement dans
 * le .cpp), même raison que gnn_v1ld_native.hpp — ne pas forcer l'include
 * LibTorch sur tous les binaires qui compilent gecode_bab.hpp.
 */

#include "gnn_bound_backend.hpp"
#include "../../solver_generic/types.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class GNNSdEditNative : public GNNBoundBackend {
public:
    GNNSdEditNative(const std::string& checkpoint_path, const std::string& device,
                     int n, int m, int N, double epsilon,
                     const Vec1d& c, const Vec1d& p, const Vec3d& A, const Vec2d& b);

    ~GNNSdEditNative() override;

    GNNSdEditNative(const GNNSdEditNative&)            = delete;
    GNNSdEditNative& operator=(const GNNSdEditNative&) = delete;

    double query(const std::vector<int8_t>& x_mask, const Vec2d& b_res,
                 double curr_cost, int depth, GNNCallTiming& timing) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
