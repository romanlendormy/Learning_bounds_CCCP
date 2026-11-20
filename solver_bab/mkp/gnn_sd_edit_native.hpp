#pragma once
/*
 * gnn_sd_edit_native.hpp — Borne GNN one-shot en-process, variante sd_edit
 * (arêtes inter-scénarios en étoile), sans IPC.
 *
 * Graphe sd_edit : arêtes scénario-scénario en étoile ancrée au scénario 0
 * (poids 1, O(n_g·N_z) arêtes) — cf. gnn/mkp/train_SLD_sd_edit.py.
 * Features de nœuds (7m+5), calcul de L1(λ) (lagrangian_sld_cpp) et export
 * TorchScript (gnn/mkp/gnn_sld_scriptable.py, qui ne dépend pas de la
 * topologie du graphe).
 *
 * pImpl : ce header ne dépend PAS de <torch/script.h> (inclus seulement dans
 * le .cpp) — ne pas forcer l'include
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
