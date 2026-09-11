/*
 * gecode_bab_ssp_gnn_batched.cpp — Implémentation de solve_gecode_bab_ssp_gnn_batched.
 *
 * Boucle DFS + bucketing par (G, N_z) -- mirror de
 * solver_bab/mkp/gecode_bab_gnn_batched.cpp (MKP), adapté :
 *   - SSP minimise (total_cost) au lieu de maximiser (obj) -- pruning et
 *     mise à jour d'incumbent inversés en conséquence.
 *   - Bucket indexé par la PAIRE (G, N_z), pas un seul entier n_g (cf.
 *     docstring gecode_bab_ssp_gnn_batched.hpp -- G et N_z rétrécissent
 *     indépendamment).
 *   - AUCUN incumbent glouton initial (contrairement à MKP/greedy_mkp_local) :
 *     le moteur natif Gecode::BAB<> utilisé par le mode GNN séquentiel
 *     (gecode_bab_ssp.hpp) ne pré-sème pas non plus d'incumbent -- pour rester
 *     comparable, ce driver démarre aussi best_val = +infini et découvre son
 *     premier incumbent par la recherche elle-même.
 *   - Arité du choix VARIABLE (4 pour x_flat via SimpleBrancher, 2 pour z via
 *     le brancher natif Gecode) -- utilise Choice::alternatives(), pas de 2
 *     codé en dur comme MKP (où x ET z sont tous deux binaires).
 *   - Pas d'équivalent b_res/apply_item_fixed : extract_pending() rescane
 *     tout à chaque nœud (O(E*T+N), même coût que more() aujourd'hui) --
 *     aucun état incrémental à maintenir entre commit() et l'appel batché.
 *
 *   pour chaque enfant fraîchement créé :
 *     commit() (fixe la variable, AUCUNE borne) → status() [CHECK CP PUR]
 *     extract_pending() : si G==0 ou N_z==0 (feuille en x, ou z totalement
 *       résolu par propagation avant tout branchement -- zone jamais
 *       entraînée) : traité IMMÉDIATEMENT, pas de borne GNN
 *     sinon : (x_mask, z_mask, curr_cost, depth) poussés dans pending[{G,N_z}]
 *
 *   au flush d'un bucket (taille K atteinte, ou plus rien dans `open`) :
 *     UN SEUL appel backend->query_batch(...) pour tout le bucket
 *     pour chaque membre : post_gnn_bound(lb_int) → SECOND status() [propage
 *       la borne fraîchement postée] → pruning/incumbent contre un snapshot
 *       de best_val (protocole "en aveugle", même principe que MKP)
 */

#include "gecode_bab_ssp_gnn_batched.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace {

struct PendingEntrySSP {
    std::vector<int8_t> x_mask, z_mask;
    double curr_cost;
    int depth;
    int outer_depth;
    SSPSpaceGNNBatched* space;
};

} // namespace

GNNBatchedSSPBabResult solve_gecode_bab_ssp_gnn_batched(
    int E, int T, int N, double epsilon,
    const std::vector<std::tuple<int,int,int>>& transitions,
    const std::vector<int>& accepting, int initial,
    const Vec1d& salary, const Vec2i& demand, const Vec1d& p,
    double time_lim, int K,
    SSPGNNBoundBackendBatch* backend,
    SSPBranchHeuristic branch_heuristic)
{
    if (K < 1)
        throw std::invalid_argument("solve_gecode_bab_ssp_gnn_batched : K doit être >= 1");
    if (!backend)
        throw std::invalid_argument("solve_gecode_bab_ssp_gnn_batched : backend requis (non nullptr)");

    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();
    auto time_ok = [&]() {
        if (time_lim <= 0.0) return true;
        return std::chrono::duration<double>(Clock::now() - t0).count() < time_lim;
    };

    // bound_mode/gnn_client ignorés ici (more() jamais appelé -- SimpleBrancher
    // ne le déclenche pas) ; SSPSharedData réutilisé uniquement pour les
    // données du problème + branch_order/dfa/work_set/sal_ub/min_feas.
    SSPSharedData shared(E, T, N, epsilon, transitions, accepting, initial, salary, demand, p);
    shared.branch_heuristic = branch_heuristic;

    using NodeTup = std::tuple<double, int, SSPSpaceGNNBatched*>;  // (lb, outer_depth, space)
    std::vector<NodeTup> open;
    std::map<std::pair<int,int>, std::vector<PendingEntrySSP>> pending;  // clé (G, N_z)

    int64_t nodes = 0, fails = 0, max_depth = 0, incumbent_impr = 0;
    double  best_val = std::numeric_limits<double>::infinity();
    SSPSpaceGNNBatched* best = nullptr;

    int64_t n_bucket_flushes = 0, sum_bucket_fill_at_flush = 0, max_pending_live = 0;

    auto count_pending_live = [&]() {
        int64_t total = 0;
        for (const auto& kv : pending) total += static_cast<int64_t>(kv.second.size());
        return total;
    };
    auto any_pending_nonempty = [&]() {
        for (const auto& kv : pending) if (!kv.second.empty()) return true;
        return false;
    };

    auto flush_bucket = [&](const std::pair<int,int>& key) {
        auto it = pending.find(key);
        if (it == pending.end() || it->second.empty()) return;
        auto& bucket = it->second;

        ++n_bucket_flushes;
        sum_bucket_fill_at_flush += static_cast<int64_t>(bucket.size());
        const int B = static_cast<int>(bucket.size());

        std::vector<std::vector<int8_t>> x_masks(B), z_masks(B);
        std::vector<double> curr_costs(B);
        std::vector<int>    depths(B);
        for (int i = 0; i < B; ++i) {
            x_masks[i] = bucket[i].x_mask; z_masks[i] = bucket[i].z_mask;
            curr_costs[i] = bucket[i].curr_cost; depths[i] = bucket[i].depth;
        }
        std::vector<double> cost_lower(B);
        backend->query_batch(x_masks, z_masks, curr_costs, depths, cost_lower);

        const double snapshot_best = best_val;
        double best_leaf_val = std::numeric_limits<double>::infinity();
        SSPSpaceGNNBatched* best_leaf_space = nullptr;

        for (int i = 0; i < B; ++i) {
            SSPSpaceGNNBatched* sp = bucket[i].space;

            // lb_salary : borne inférieure sur le coût salarial réel, déjà
            // dans cette convention (pas de -dual_val ici, cf. more() où la
            // conversion depuis la convention max(-salaire) a lieu -- le
            // backend renvoie directement cost_lower comme GNNDLDNative::query).
            const double lb_salary = cost_lower[i];
            const int lb_int = static_cast<int>(std::floor(lb_salary * SSPSharedData::SCALE));
            sp->post_gnn_bound(lb_int);
            SpaceStatus cst = sp->status();   // SECOND status() : propage la borne fraîchement postée

            if (cst == SS_FAILED) { ++fails; delete sp; continue; }

            double clb = sp->total_cost.min() / static_cast<double>(SSPSharedData::SCALE);

            if (cst == SS_SOLVED) {
                double val = sp->total_cost.val() / static_cast<double>(SSPSharedData::SCALE);
                if (val < best_leaf_val) {
                    if (best_leaf_space) delete best_leaf_space;
                    best_leaf_val = val; best_leaf_space = sp;
                } else delete sp;
                continue;
            }

            if (clb >= snapshot_best - 1e-6) { ++fails; delete sp; continue; }
            open.push_back({clb, bucket[i].outer_depth, sp});
        }
        bucket.clear();

        if (best_leaf_space) {
            if (best_leaf_val < best_val - 1e-9) {
                best_val = best_leaf_val; delete best; best = best_leaf_space; ++incumbent_impr;
            } else delete best_leaf_space;
        }
    };

    // Racine — aucun calcul de borne à l'initialisation, seulement la
    // propagation CP (même convention que le moteur séquentiel).
    {
        SSPSpaceGNNBatched* root = new SSPSpaceGNNBatched(&shared);
        SpaceStatus rst = root->status();
        if (rst == SS_BRANCH) {
            double rlb = root->total_cost.min() / static_cast<double>(SSPSharedData::SCALE);
            open.push_back({rlb, 0, root});
        } else if (rst == SS_SOLVED) {
            best_val = root->total_cost.val() / static_cast<double>(SSPSharedData::SCALE);
            best = root; ++incumbent_impr;
        } else {
            delete root;
        }
    }

    while ((!open.empty() || any_pending_nonempty()) && time_ok()) {
        if (open.empty()) {
            for (auto& kv : pending) flush_bucket(kv.first);
            continue;
        }

        auto [lb, outer_depth, space] = open.back(); open.pop_back();

        if (lb >= best_val - 1e-6) { ++fails; delete space; continue; }

        SpaceStatus st = space->status();

        if (st == SS_FAILED) { ++fails; delete space; continue; }

        if (st == SS_SOLVED) {
            ++nodes;
            double val = space->total_cost.val() / static_cast<double>(SSPSharedData::SCALE);
            if (val < best_val - 1e-9) {
                best_val = val; delete best; best = space; space = nullptr;
                ++incumbent_impr;
            } else { delete space; }
            continue;
        }

        ++nodes;
        if (outer_depth > max_depth) max_depth = outer_depth;

        const Gecode::Choice* c_choice = space->choice();
        const unsigned int n_alt = c_choice->alternatives();

        for (unsigned int a = 0; a < n_alt; ++a) {
            SSPSpaceGNNBatched* child = static_cast<SSPSpaceGNNBatched*>(space->clone());
            child->commit(*c_choice, a);           // fixe la variable, AUCUNE borne

            SpaceStatus cst = child->status();     // check CP pur (pas encore de borne)

            if (cst == SS_FAILED) { ++fails; delete child; continue; }

            std::vector<int8_t> x_mask, z_mask; double curr_cost; int depth, G, N_z;
            bool has_candidate = child->extract_pending(x_mask, z_mask, curr_cost, depth, G, N_z);

            if (!has_candidate) {
                // Feuille en x (total_cost déjà exact) OU z entièrement résolu
                // par propagation avant tout branchement explicite (zone
                // jamais entraînée, cf. more()) : traitement IMMÉDIAT, comme
                // more() qui retourne sans rien poster.
                double clb = child->total_cost.min() / static_cast<double>(SSPSharedData::SCALE);
                if (clb >= best_val - 1e-6) { ++fails; delete child; continue; }
                if (cst == SS_SOLVED) {
                    double val = child->total_cost.val() / static_cast<double>(SSPSharedData::SCALE);
                    if (val < best_val - 1e-9) {
                        best_val = val; delete best; best = child; child = nullptr;
                        ++incumbent_impr;
                    } else delete child;
                    continue;
                }
                open.push_back({clb, outer_depth + 1, child});
                continue;
            }

            auto& bucket = pending[{G, N_z}];
            bucket.push_back({std::move(x_mask), std::move(z_mask), curr_cost, depth, outer_depth + 1, child});
            max_pending_live = std::max(max_pending_live, count_pending_live());
            if (static_cast<int>(bucket.size()) == K)
                flush_bucket({G, N_z});
        }

        delete c_choice;
        delete space;
    }

    bool timed_out = !open.empty() || any_pending_nonempty();

    while (!open.empty()) { auto [lb, d, sp] = open.back(); open.pop_back(); delete sp; }
    for (auto& kv : pending) { for (auto& e : kv.second) delete e.space; kv.second.clear(); }

    double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();

    GNNBatchedSSPBabResult res;
    res.K = K;
    res.n_bucket_flushes = n_bucket_flushes;
    res.mean_bucket_fill_at_flush = n_bucket_flushes > 0
        ? static_cast<double>(sum_bucket_fill_at_flush) / static_cast<double>(n_bucket_flushes) : 0.0;
    res.max_pending_live = max_pending_live;

    res.solve_time_s = elapsed;
    res.nodes = nodes;
    res.fails = fails;
    res.depth = max_depth;
    res.incumbent_improvements = incumbent_impr;

    if (best) {
        res.objective = static_cast<double>(best->total_cost.val()) / SSPSharedData::SCALE;
        res.feasible = true;
        res.optimal  = !timed_out;
        res.status   = timed_out ? "FEASIBLE" : "OPTIMAL";
        delete best;
    } else {
        res.objective = std::numeric_limits<double>::quiet_NaN();
        res.feasible = false;
        res.optimal  = false;
        res.status   = timed_out ? "TIMEOUT" : "INFEASIBLE";
    }

    return res;
}
