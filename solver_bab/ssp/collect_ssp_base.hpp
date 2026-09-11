#pragma once
/*
 * collect_ssp_base.hpp — B&B Gecode CP pur (SSP) avec collecte GNN.
 *
 * Analogue SSP de gecode_base/collect_gecode_base.hpp (MKP) : réutilise
 * SSPSpaceBnB/SSPSharedData (gecode_bab_ssp.hpp) avec use_lagrangian_bound
 * =false — BoundBrancher::commit ne rappelle alors jamais more(), donc aucun
 * coût SLD/DLD pendant la collecte (pruning CP pur uniquement), comme la
 * variante "B&B base_ref" déjà validée dans bench_ssp_bab_file.cpp. On garde
 * bound_mode=SLD (défaut de SSPSharedData) plutôt que DLD : les deux sont
 * de toute façon jamais lus (use_lagrangian_bound=false), mais leur allocation
 * est copiée à CHAQUE clone() — SLD (N×E×T×4 doubles) est T fois plus légère
 * que DLD (N×T×E×T×4) pour un effet strictement identique (nul) ici.
 *
 * Différence structurelle avec MKP : x_flat n'est pas booléen (alphabet à 4
 * symboles Σ={O=0,P=1,R=2,W=3}), donc le masque ternaire (0/1/2) de MKP
 * devient un masque à 5 valeurs : 0..3 = symbole fixé, 4 = libre. z_mask
 * reste ternaire (0/1/2), identique à MKP (z est booléen dans les deux
 * modèles).
 *
 * Format de sortie — deux fichiers CSV :
 *
 *   instances.txt (une ligne par instance résolue à optimalité)
 *     instance_id, E, T, N, epsilon, opt,
 *     initial, n_states, n_accepting, accepting[0..n_accepting-1],
 *     n_transitions, (src, sym, dst) × n_transitions,
 *     salary[0..T-1],
 *     demand[i][t]   (i=0..N-1, t=0..T-1)
 *
 *   nodes.txt (une ligne par nœud de branchement) — Format C adapté
 *     instance_id, depth, curr_cost, x_mask[0..E*T-1], z_mask[0..N-1]
 *   x_mask : 0=O,1=P,2=R,3=W fixé ; 4=libre.  z_mask : 0/1 fixé, 2=libre.
 *   (3 + E*T + N champs — sans ub_node ni incumbent, comme MKP)
 */

#include "gecode_bab_ssp.hpp"
#include "ssp_loader.hpp"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// NodeRecordSSP — état d'un nœud sans borne Lagrangienne
// ─────────────────────────────────────────────────────────────────────────────

struct NodeRecordSSP {
    int    depth;
    double curr_cost;
    std::vector<int8_t> x_mask;  // taille E*T : 0..3 = symbole fixé, 4 = libre
    std::vector<int8_t> z_mask;  // taille N   : 0/1 = fixé, 2 = libre
};

NodeRecordSSP collect_node_state_ssp(
    const SSPSpaceBnB* space,
    const SSPSharedData& data,
    int depth);

void write_instance_line_ssp(
    std::ofstream& out,
    int instance_id,
    const SSPInstance& inst,
    double opt);

// Format C : inst_id, depth, curr_cost, x_mask[E*T], z_mask[N]
void write_node_line_ssp(
    std::ofstream& out,
    int instance_id,
    const NodeRecordSSP& rec);

// ─────────────────────────────────────────────────────────────────────────────
// Résultat
// ─────────────────────────────────────────────────────────────────────────────

struct SSPCollectResult {
    double      objective;
    bool        optimal;
    bool        feasible;
    double      solve_time_s;
    int64_t     nodes;
    int64_t     fails;
    int64_t     depth;
    std::string status;
};

// ─────────────────────────────────────────────────────────────────────────────
// solve_gecode_bab_ssp_collect_base — DFS manuelle, pruning CP pur
// ─────────────────────────────────────────────────────────────────────────────

SSPCollectResult solve_gecode_bab_ssp_collect_base(
    const SSPInstance& inst,
    std::ofstream& instances_out,
    std::ofstream& nodes_out,
    int       instance_id,
    double    time_lim           = -1.0,
    int       max_depth_collect  = -1);
