#pragma once
/*
 * gnn_sd_edit_native_aoti.hpp — Borne GNN one-shot en-process, variante
 * sd_edit, backend AOTInductor (au lieu de TorchScript, cf.
 * gnn_sd_edit_native.hpp/.cpp).
 *
 * Fichier séparé de gnn_sd_edit_native.hpp/.cpp — même raison que la
 * séparation homogène/sd_edit déjà établie (cf. commentaire de
 * gnn_sd_edit_native.hpp) : AOTI et TorchScript restent deux artefacts de
 * recherche comparés indépendamment (gain wall-clock ×1.77-1.79 mesuré,
 * cf. mémoire "aoti-sdedit-wallclock-result"), on ne veut pas qu'une
 * modification de l'un affecte l'autre via une abstraction commune.
 * build_graph_cpp()/lagrangian_v1ld_cpp() sont donc dupliqués mot pour mot
 * depuis gnn_sd_edit_native.cpp (seule la section chargement/forward change).
 *
 * Différence structurelle majeure avec le backend TorchScript : l'export
 * TorchScript recompile en interne PAR FORME de tenseur (JIT profiling,
 * quasi gratuit, cf. gecode_bab.cpp) — un artefact AOTInductor est, lui,
 * COMPILÉ EN AVANCE pour une forme figée (n_g fixé à l'export). Ce backend
 * charge donc UNE COLLECTION d'artefacts pré-compilés, un par valeur de
 * n_g rencontrée (1..n), mis en cache sur disque par
 * gnn_sd_edit/export_aoti_sdedit.py (~15-20s de compilation par forme LA
 * PREMIÈRE FOIS, quasi instantané ensuite — cache invalidé si le
 * checkpoint est modifié, cf. ensure_aoti_exports()).
 *
 * pImpl : ce header ne dépend PAS de <torch/...>, même raison que
 * gnn_sd_edit_native.hpp.
 */

#include "gnn_bound_backend.hpp"
#include "../../solver_generic/types.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class GNNSdEditNativeAOTI : public GNNBoundBackend {
public:
    GNNSdEditNativeAOTI(const std::string& checkpoint_path, const std::string& device,
                        int n, int m, int N, double epsilon,
                        const Vec1d& c, const Vec1d& p, const Vec3d& A, const Vec2d& b);

    ~GNNSdEditNativeAOTI() override;

    GNNSdEditNativeAOTI(const GNNSdEditNativeAOTI&)            = delete;
    GNNSdEditNativeAOTI& operator=(const GNNSdEditNativeAOTI&) = delete;

    double query(const std::vector<int8_t>& x_mask, const Vec2d& b_res,
                 double curr_cost, int depth, GNNCallTiming& timing) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
