#pragma once
/*
 * gnn_dld_native.hpp — Borne GNN one-shot en-process pour DLD (SSP),
 * mirror structurel de solver_bab/mkp/gnn_sld_native.hpp (MKP).
 *
 * Différences avec MKP (cf. discussion 2026-08-18) :
 *   - Trois axes varient/interviennent (N_z scénarios, T périodes fixes,
 *     G paires (e,t) libres) au lieu d'un seul (n_g) — le modèle exporté
 *     (gnn_ssp/gnn_ssp_scriptable_batched.py) attend (n_g, T, B) en plus
 *     du triplet de tenseurs habituel.
 *   - Le sous-problème ζ/θ (SSPDLDProblemBnB, solver_generic/ssp/
 *     ssp_dld_problem_bnb.hpp) n'a PAS d'équivalent batché façon
 *     solve_zeta_batch (MKP) — appels un par un ici, en boucle simple
 *     (glouton O(E·T·Q), pas de CP-SAT — bon marché par appel, cf. décision
 *     "pas d'OpenMP pour l'instant, à ajouter si le profilage le justifie").
 *   - z (sélection de scénario) PEUT être fixé par le B&B côté SSP
 *     (contrairement à MKP SLD où z_mask est toujours libre) — z_mask est
 *     donc un paramètre réel de query(), pas une hypothèse figée.
 *
 * Étape 2 du plan (validation numérique standalone) : PAS encore branché sur
 * SSPSpaceBnB/Gecode — l'instance et l'état de nœud sont fournis directement
 * par l'appelant (cf. test_gnn_dld_native.cpp). Le branchement réel dans
 * more() est l'étape 3, différée (cf. discussion sur l'absence de masque
 * incrémental dans SSPSpaceBnB).
 *
 * pImpl : ce header ne dépend pas de <torch/script.h> (cf. même raison que
 * gnn_sld_native.hpp — isoler l'include LibTorch au .cpp).
 */

#include "ssp_gnn_bound_backend.hpp"
#include "../../solver_generic/types.hpp"
#include "../../solver_generic/ssp/ssp_loader.hpp"   // SSPInstance

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Détail cumulé (sur tous les appels query() d'une instance donnée) du temps
// passé dans chaque phase -- cf. discussion 2026-08-19, "détail du temps B&B
// pour SSP (construction graphe, calcul zeta, forward GNN)".
struct SSPGNNQueryTiming {
    double  t_graph_build_s     = 0.0;
    double  t_forward_s         = 0.0;
    double  t_zeta_theta_s      = 0.0;  // boucle OpenMP ζ/θ (lagrangian_dld_cpp) uniquement
    double  t_lagrangian_rest_s = 0.0;  // reste de lagrangian_dld_cpp (prep Λ_sum_i/cost_full + tri glouton + L1)
    int64_t n_queries           = 0;
};

class GNNDLDNative : public SSPGNNBoundBackend {
public:
    GNNDLDNative(const std::string& checkpoint_path, const std::string& device,
                    const SSPInstance& inst);

    ~GNNDLDNative() override;

    GNNDLDNative(const GNNDLDNative&)            = delete;
    GNNDLDNative& operator=(const GNNDLDNative&) = delete;

    // x_mask (E*T,) : 0..3 = symbole fixé (idx e*T+t), 4 = libre — convention
    //                 disque, cf. collect_ssp_base.hpp.
    // z_mask (N,)   : 0/1 = fixé (0=exclu, 1=sélectionné), 2 = libre.
    // Retourne cost_lower = -L1(λ) (borne inférieure sur le coût, cf.
    // convention de signe train_DLD_ssp.py::relative_gap_pct).
    double query(const std::vector<int8_t>& x_mask, const std::vector<int8_t>& z_mask,
                 double curr_cost, int depth) override;

    // Cumul depuis la construction (pas remis à zéro entre deux query()) --
    // à lire APRÈS un run complet pour le détail des temps.
    SSPGNNQueryTiming cumulative_timing() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
