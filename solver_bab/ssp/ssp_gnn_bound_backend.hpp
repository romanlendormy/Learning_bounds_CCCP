#pragma once
/*
 * ssp_gnn_bound_backend.hpp — Interface abstraite entre SSPSpaceBnB::more()
 * et l'implémentation concrète de la borne GNN (GNNV1CLDNative,
 * gnn_v1cld_native.hpp), mirror de solver_bab/mkp/gnn_bound_backend.hpp.
 *
 * Pourquoi cette indirection : gecode_bab_ssp.hpp (et tout binaire qui le
 * compile — bab_ssp_gecode_base_collect, bench_ssp_bab, debug_root_bound,
 * etc., aucun n'a besoin de LibTorch) ne doit dépendre QUE de ce header,
 * sans torche ni dépendance LibTorch. SSPSharedData::gnn_client est un
 * pointeur vers cette interface, pas vers GNNV1CLDNative directement — un
 * appel virtuel ne force pas le linker à résoudre l'implémentation concrète
 * pour les binaires qui ne construisent jamais de GNNV1CLDNative (gnn_client
 * reste nullptr, jamais appelé). Seul le futur binaire qui active
 * bound_mode=GNN a besoin de lier gnn_v1cld_native.cpp + LibTorch.
 */

#include <cstdint>
#include <vector>

class SSPGNNBoundBackend {
public:
    virtual ~SSPGNNBoundBackend() = default;

    // x_mask (E*T,) : 0..3 = symbole fixé, 4 = libre (convention disque).
    // z_mask (N,)   : 0/1 = fixé, 2 = libre.
    // Retourne cost_lower = -L1(λ) (borne inférieure sur le coût salarial).
    virtual double query(const std::vector<int8_t>& x_mask, const std::vector<int8_t>& z_mask,
                         double curr_cost, int depth) = 0;
};
