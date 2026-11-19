#pragma once
/*
 * ccpo_gnn_backend_batch.hpp — interface backend GNN batché, SANS dépendance
 * Gecode (ni libtorch) : le backend réel (gnn_ccpo_native_batched.cpp) inclut
 * libtorch et ne doit PAS voir `using namespace Gecode` de gecode_bab_ccpo.hpp
 * (risque de collisions de noms) -- même séparation que MKP
 * (gnn_bound_backend_batch.hpp).
 */

#include <cstddef>
#include <vector>

// Sous-problème résiduel d'un nœud (construit par CCPOSpaceSLD::build_residual).
struct CCPOResidualData {
    std::vector<int> free_vars;
    int n_free = 0, Q_res = 0, k_res = 0;
    std::vector<double> cost_free;
    std::vector<std::vector<std::vector<double>>> Sigma_free;  // (N,n_free,n_free)
    std::vector<std::vector<double>> R_ext;                     // (N,n_free)
};

// Snapshot d'un nœud en attente de borne.
struct CCPOQuery {
    CCPOResidualData res;
    std::vector<double> b_res;        // (N,) budget résiduel par scénario (copie)
    int depth = 0;                    // profondeur DFS (définition de la collecte d'entraînement)
    double frac_fixed_active = 0.0;   // fixed_count_active / n
    double cost_fixed = 0.0;          // fp_running
};

// Constantes d'instance partagées par tous les nœuds d'un run.
struct CCPOInstanceCtx {
    int n = 0, N = 0, L = 0, U = 0, k = 0, Q = 0, min_feas = 0;
    double c_scale = 1.0, sigma_scale = 1.0, epsilon = 0.0;
    double B = 0.0;  // budget de risque d'origine (features v3)
    const std::vector<bool>* infeasible = nullptr;
};

// Contrat : tous les membres du batch ont le MÊME n_free. lambda_out[b] est
// (N, n_free) -- lignes des scénarios presolve-infaisables à 0.
class CCPOGNNBackendBatch {
public:
    virtual ~CCPOGNNBackendBatch() = default;
    virtual void query_batch(const CCPOInstanceCtx& ctx,
                             const std::vector<const CCPOQuery*>& batch,
                             std::vector<std::vector<std::vector<double>>>& lambda_out) = 0;
};

// Backend factice : λ = 0 (validation de la mécanique, sans GNN).
class CCPOZeroBackend : public CCPOGNNBackendBatch {
public:
    void query_batch(const CCPOInstanceCtx& ctx,
                     const std::vector<const CCPOQuery*>& batch,
                     std::vector<std::vector<std::vector<double>>>& lambda_out) override
    {
        lambda_out.assign(batch.size(), {});
        for (std::size_t b = 0; b < batch.size(); ++b)
            lambda_out[b].assign(ctx.N, std::vector<double>(batch[b]->res.n_free, 0.0));
    }
};
