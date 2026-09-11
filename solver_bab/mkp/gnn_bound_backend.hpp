#pragma once
/*
 * gnn_bound_backend.hpp — Interface commune aux backends de borne GNN
 * one-shot (bound_mode=GNN) :
 *   - GNNSLDClient (gnn_sld_client.hpp) : process Python persistant, IPC
 *     par pipe (mesuré/isolé — cf. GNNCallTiming). Supporte gnn_variant
 *     "homogene" et "sd_edit" (délègue à build_graph côté Python).
 *   - GNNSLDNative (gnn_sld_native.hpp) : build_graph + forward + L1(λ)
 *     réimplémentés en C++/LibTorch, en-process (pas d'IPC). gnn_variant
 *     "homogene" (clique scénario-scénario).
 *   - GNNSdEditNative (gnn_sd_edit_native.hpp) : même principe que
 *     GNNSLDNative, gnn_variant "sd_edit" (étoile scénario-scénario ancrée
 *     au scénario 0 — O(n_g·N) arêtes au lieu de O(n_g·N²)). Fichier séparé
 *     de GNNSLDNative par choix (cf. commentaire du .hpp), pas une
 *     sous-classe de celui-ci.
 *
 * gecode_bab.hpp n'appelle que cette interface (via MKPSharedData::gnn_client,
 * un GNNBoundBackend*) — le choix du backend ET de la variante est fait une
 * fois, à la construction, dans solve_gecode_bab_mkp().
 */

#include "../../solver_generic/types.hpp"

#include <cstdint>
#include <vector>

struct GNNCallTiming {
    double t_graph_build_s   = 0.0;
    // Sous-décomposition de t_graph_build_s (backend natif uniquement — 0.0
    // pour le backend IPC, qui ne les peuple pas) : features = calcul des
    // tenseurs de nœuds (cap_frac, dantzig_rank, etc.), edges = génération
    // des index/poids d'arêtes item-item et scénario-scénario.
    double t_graph_features_s = 0.0;
    double t_graph_edges_s     = 0.0;
    double t_forward_s       = 0.0;
    double t_bound_compute_s = 0.0;
};

class GNNBoundBackend {
public:
    virtual ~GNNBoundBackend() = default;

    // x_mask[k] ∈ {0,1,2} (0=fixé_0, 1=fixé_1, 2=libre), b_res[i][j] = capacité
    // résiduelle du scénario i / contrainte j APRÈS déduction des items fixés
    // à 1 (déjà maintenue incrémentalement par l'appelant, cf.
    // MKPSpaceSLD::b_res dans gecode_bab.hpp — ne JAMAIS la recalculer depuis
    // x_mask ici : c'est précisément ce que more() maintient à coût O(delta)
    // pour éviter de la ressommer à chaque nœud, cf. revue niveau 1 point B).
    // curr_cost = profit déjà fixé (fp), depth = nombre de variables fixées.
    // Retourne ub_relative = L1(λ) sur le sous-problème résiduel (l'appelant
    // calcule ub = fp + ub_relative). Remplit timing avec la décomposition
    // temporelle.
    virtual double query(const std::vector<int8_t>& x_mask, const Vec2d& b_res,
                         double curr_cost, int depth, GNNCallTiming& timing) = 0;
};
