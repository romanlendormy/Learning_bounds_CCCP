#pragma once
/*
 * gecode_bab_collect.hpp — B&B Gecode avec collecte de données d'entraînement GNN.
 *
 * Même algorithme que solve_gecode_bab_mkp, avec en plus la collecte à chaque
 * nœud de branchement de l'état du sous-problème courant.
 *
 * Format de sortie — deux fichiers CSV :
 *
 *   instances.txt  (une ligne par instance B&B résolue à optimalité)
 *     instance_id, m, n, N, epsilon, opt,
 *     c[0..n-1],
 *     b[k=0..N-1, j=0..m-1]  (row-major sur k),
 *     A[k=0..N-1, j=0..m-1, i=0..n-1]  (row-major sur (k,j))
 *
 *   nodes.txt  (une ligne par nœud de branchement)
 *     instance_id, depth, curr_cost, ub_node, violated_count,
 *     fix_ones[0..n-1]
 *   (b_res est omis : reconstructible depuis fix_ones + b + A de instances.txt)
 *
 * Seules les instances résolues à optimalité (res.optimal == true) sont écrites,
 * car le label opt doit être la vraie valeur optimale.
 */

#include "gecode_bab.hpp"

#include <fstream>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// NodeRecord — état du sous-problème à un nœud de branchement
// ─────────────────────────────────────────────────────────────────────────────

struct NodeRecord {
    int    depth;
    double curr_cost;       // Σ_{k ∈ F₁} c[k]  (profit des items déjà sélectionnés)
    double ub_node;         // borne supérieure Lagrangienne calculée à ce nœud
    int    violated_count;  // |{i : ∃j, b_res[i][j] < 0}|  (scénarios déjà violés)

    std::vector<int8_t> fix_ones;  // fix_ones[k] = 1 si x[k] fixé à 1,  taille n
};

// ─────────────────────────────────────────────────────────────────────────────
// collect_node_state
// ─────────────────────────────────────────────────────────────────────────────
// Extrait l'état du sous-problème depuis un espace Gecode après space->status().
// Doit être appelé uniquement quand status() a retourné SS_BRANCH.

NodeRecord collect_node_state(
    const MKPSpaceV1LD* space,
    const MKPSharedData& data,
    int depth);

// ─────────────────────────────────────────────────────────────────────────────
// Fonctions d'écriture CSV
// ─────────────────────────────────────────────────────────────────────────────

// Écrit la ligne d'instance (appelée après la résolution, quand opt est connu).
void write_instance_line(
    std::ofstream& out,
    int instance_id,
    const MKPSharedData& data,
    double opt);

// Écrit la ligne d'un nœud.
void write_node_line(
    std::ofstream& out,
    int instance_id,
    const NodeRecord& rec);

// ─────────────────────────────────────────────────────────────────────────────
// solve_gecode_bab_mkp_collect
// ─────────────────────────────────────────────────────────────────────────────
// Identique à solve_gecode_bab_mkp, mais collecte l'état de chaque nœud de
// branchement. Écrit dans les fichiers de sortie uniquement si l'instance est
// résolue à optimalité (res.optimal == true).

GecodeV1LDResult solve_gecode_bab_mkp_collect(
    int n, int m, int N, double epsilon,
    const Vec1d& c, const Vec1d& p,
    const Vec3d& A, const Vec2d& b,
    std::ofstream& instances_out,
    std::ofstream& nodes_out,
    int       instance_id,
    double    time_lim         = 300.0,
    int       T_root           = 200,
    int       T_node           = 30,
    BoundMode bound_mode       = BoundMode::V1LD,
    bool      patience_mode    = false,
    double    patience_tol     = 0.1,
    int       max_depth_collect = -1);  // -1 = collecter tous les nœuds
