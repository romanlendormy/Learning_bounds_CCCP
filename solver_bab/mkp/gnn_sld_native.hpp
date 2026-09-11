#pragma once
/*
 * gnn_sld_native.hpp — Borne GNN one-shot en-process, sans IPC.
 *
 * Alternative à GNNSLDClient (gnn_sld_client.hpp), inspirée de
 * solver_swann_darius.cpp : le modèle est chargé UNE FOIS via
 * torch::jit::load() (TorchScript exporté par gnn_sd_edit/gnn_sld_scriptable.py,
 * qui réimplémente gnn.train_gnn_sld.GNN en torch pur — sans
 * torch_geometric, pas scriptable, cf. commentaires du .cpp) et appelé
 * directement via .forward() — PAS de réimplémentation manuelle de
 * l'architecture en C++ (contrairement à une version antérieure de ce
 * fichier). Contrairement à solver_swann_darius.cpp, pas de raffinement
 * itératif post-GNN : la borne reste strictement one-shot (contrainte du
 * projet, cf. mémoire "Contrainte borne one-shot").
 *
 * build_graph() et lagrangian_sld() (construction de graphe + calcul de la
 * borne à partir de λ) restent réimplémentés à la main en C++ — ce ne sont
 * pas le modèle. Le sous-problème ζ réutilise solve_zeta_batch()
 * (gnn/bnb_solver.cpp), déjà utilisé par lagrangian_sld côté Python.
 *
 * Validé (voir session de développement) : lam et L1(λ) calculés ici
 * correspondent à la référence Python à ~1e-4 près (bruit float32).
 *
 * pImpl : ce header ne dépend PAS de <torch/script.h> (inclus seulement dans
 * le .cpp), pour ne pas forcer le chemin d'include LibTorch sur tous les
 * binaires qui compilent gecode_bab.hpp (bab_mkp_gecode_collect, etc.) —
 * seul bab_mkp_gecode qui instancie réellement GNNSLDNative en a besoin.
 *
 * device : "cpu" ou "cuda" — cuda lève une exception si LibTorch a été lié
 * sans support CUDA ou si aucun GPU n'est visible (torch::cuda::is_available()).
 */

#include "gnn_bound_backend.hpp"
#include "../../solver_generic/types.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class GNNSLDNative : public GNNBoundBackend {
public:
    GNNSLDNative(const std::string& checkpoint_path, const std::string& device,
                  int n, int m, int N, double epsilon,
                  const Vec1d& c, const Vec1d& p, const Vec3d& A, const Vec2d& b);

    ~GNNSLDNative() override;

    GNNSLDNative(const GNNSLDNative&)            = delete;
    GNNSLDNative& operator=(const GNNSLDNative&) = delete;

    double query(const std::vector<int8_t>& x_mask, const Vec2d& b_res,
                 double curr_cost, int depth, GNNCallTiming& timing) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
