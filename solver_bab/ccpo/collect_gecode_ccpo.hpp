#pragma once
/*
 * collect_gecode_ccpo.hpp — B&B Gecode CP pur (mode REF) avec collecte
 * d'états de nœuds pour l'entraînement du GNN.
 *
 * Sur le patron de gecode_base/collect_gecode_base.hpp (MKP) : réutilise
 * directement CCPOSpaceSLD en BoundMode::REF (aucune borne SLD calculée,
 * more() est un no-op -- cf. gecode_bab_ccpo.hpp), fait une DFS manuelle
 * identique en esprit à solve_gecode_bab_ccpo, et dump l'état de chaque
 * nœud SS_BRANCH (avant branchement) qui survit à l'élagage CP naturel de
 * Gecode (obj.max() <= best_val).
 *
 * Contrairement à MKP, on ne dump PAS x_mask+z_mask bruts en pariant sur une
 * reconstruction identique côté Python -- on dump q_mask (valeur fixée de
 * chaque actif, ou -1 si libre) UNE SEULE FOIS par nœud, et c'est tout ce
 * qui est nécessaire : le sous-problème résiduel complet (cost_free,
 * Sigma_free, B_res, R_ext, Q_res, k_res) se reconstruit entièrement à
 * partir de q_mask + les données d'instance (c_int, Sigma dense, Q, k, B),
 * exactement le calcul déjà fait par CCPOSpaceSLD::more() (cf.
 * ccpo_dataset.py::reconstruct_residual_instance, qui DOIT rester en
 * synchronisation exacte avec ce calcul, terme croisé fixé×libre inclus).
 */

#include "gecode_bab_ccpo.hpp"

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <tuple>
#include <vector>

struct NodeRecordCCPO {
    int depth;
    double curr_cost;              // fp_running : profit déjà fixé à ce nœud
    std::vector<int32_t> q_mask;   // (n,) -- valeur fixée, ou -1 si libre
};

inline NodeRecordCCPO collect_node_state_ccpo(
    const CCPOSpaceSLD* space, const CCPOSharedData& data, int depth)
{
    NodeRecordCCPO rec;
    rec.depth = depth;
    rec.q_mask.assign(data.n, -1);
    rec.curr_cost = 0.0;
    for (int i = 0; i < data.n; ++i) {
        if (space->q[i].assigned()) {
            const int v = space->q[i].val();
            rec.q_mask[i] = v;
            if (v > 0) rec.curr_cost += data.c_int[i] * v;
        }
    }
    return rec;
}

// ── Format instance : une ligne, tout ce qu'il faut pour reconstruire
// CCPOSharedData côté Python SANS refaire le presolve (infeasible_scenario
// dumpé directement -- indépendant de λ/du nœud, cf. ccpo_sld.hpp). ────────
inline void write_instance_line_ccpo(
    std::ofstream& out, int instance_id, const CCPOSharedData& data, double opt)
{
    out << std::fixed << std::setprecision(6);
    out << instance_id << ',' << data.n << ',' << data.N << ',' << data.k << ','
        << data.Q << ',' << data.L << ',' << data.U << ',' << data.B << ','
        << data.min_feas << ',' << opt;

    for (int i = 0; i < data.n; ++i) out << ',' << data.c_int[i];
    for (int j = 0; j < data.N; ++j) out << ',' << (data.infeasible_scenario[j] ? 1 : 0);

    // Sigma : triangle supérieur (i,k,val) par scénario -- même format que
    // scenarios.txt (ccpo_stochastic), pour réutiliser le même parseur.
    for (int j = 0; j < data.N; ++j)
        for (const auto& t : data.sigma[j])
            out << ',' << t[0] << ',' << t[1] << ',' << t[2];

    out << '\n';
}

inline void write_node_line_ccpo(
    std::ofstream& out, int instance_id, const NodeRecordCCPO& rec)
{
    out << std::fixed << std::setprecision(6);
    out << instance_id << ',' << rec.depth << ',' << rec.curr_cost;
    for (int32_t v : rec.q_mask) out << ',' << v;
    out << '\n';
}

// ─────────────────────────────────────────────────────────────────────────
// solve_gecode_bab_ccpo_collect — DFS CP-only (REF) + collecte, sur le
// patron de solve_gecode_bab_mkp_collect_base (gecode_base).
// ─────────────────────────────────────────────────────────────────────────
inline CCPOResult solve_gecode_bab_ccpo_collect(
    int n, int N, int k, int Q, int L, int U, long long B, int min_feas,
    const std::vector<int>& c_int,
    const std::vector<std::vector<SigmaTriplet>>& sigma,
    std::ofstream& instances_out,
    std::ofstream& nodes_out,
    int instance_id,
    double time_lim,
    int max_depth_collect)
{
    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();

    CCPOSharedData shared(n, N, k, Q, L, U, B, min_feas, c_int, sigma);
    shared.bound_mode = BoundMode::REF;

    auto time_ok = [&]() {
        if (time_lim <= 0.0) return true;
        return std::chrono::duration<double>(Clock::now() - t0).count() < time_lim;
    };

    using NodeTup = std::tuple<double, int, CCPOSpaceSLD*>;
    std::vector<NodeTup> open;

    int64_t nodes_cnt = 0, fails = 0, max_depth_seen = 0, incumbent_impr = 0;
    double best_val = -1.0;
    bool found_any = false;
    CCPOSpaceSLD* best = nullptr;

    std::vector<NodeRecordCCPO> records;

    {
        CCPOSpaceSLD* root = new CCPOSpaceSLD(&shared);
        SpaceStatus rst = root->status();
        if (rst == SS_BRANCH) {
            open.push_back({static_cast<double>(root->obj.max()), 0, root});
        } else if (rst == SS_SOLVED) {
            best_val = root->obj.val(); best = root; found_any = true; ++incumbent_impr;
        } else {
            delete root;
        }
    }

    while (!open.empty() && time_ok()) {
        auto [ub, depth, space] = open.back(); open.pop_back();

        if (found_any && ub <= best_val + 1e-6) { ++fails; delete space; continue; }

        SpaceStatus st = space->status();
        if (st == SS_FAILED) { ++fails; delete space; continue; }

        if (st == SS_SOLVED) {
            ++nodes_cnt;
            const double val = space->obj.val();
            if (!found_any || val > best_val + 1e-9) {
                best_val = val; delete best; best = space; space = nullptr;
                found_any = true; ++incumbent_impr;
            } else { delete space; }
            continue;
        }

        // SS_BRANCH -- collecter avant branchement.
        ++nodes_cnt;
        if (depth > max_depth_seen) max_depth_seen = depth;
        if (max_depth_collect < 0 || depth <= max_depth_collect)
            records.push_back(collect_node_state_ccpo(space, shared, depth));

        const Gecode::Choice* ch = space->choice();
        for (int a = 0; a < 2; ++a) {
            CCPOSpaceSLD* child = static_cast<CCPOSpaceSLD*>(space->clone());
            child->commit(*ch, a);
            SpaceStatus cst = child->status();

            if (cst == SS_FAILED) { ++fails; delete child; continue; }

            const double cub = static_cast<double>(child->obj.max());
            if (found_any && cub <= best_val + 1e-6) { ++fails; delete child; continue; }

            if (cst == SS_SOLVED) {
                const double val = child->obj.val();
                if (!found_any || val > best_val + 1e-9) {
                    best_val = val; delete best; best = child; child = nullptr;
                    found_any = true; ++incumbent_impr;
                } else { delete child; }
                continue;
            }
            open.push_back({cub, depth + 1, child});
        }
        delete ch;
        delete space;
    }

    const bool timed_out = !open.empty();
    while (!open.empty()) { auto [ub, d, sp] = open.back(); open.pop_back(); delete sp; }

    const double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();

    CCPOResult res;
    res.solve_time_s = elapsed;
    res.nodes = nodes_cnt;
    res.fails = fails;
    res.depth = max_depth_seen;
    res.incumbent_improvements = incumbent_impr;

    if (found_any) {
        res.objective = best_val;
        res.feasible = true;
        res.optimal = !timed_out;
        res.status = timed_out ? "FEASIBLE" : "OPTIMAL";
        delete best;
    } else {
        res.feasible = false;
        res.optimal = false;
        res.status = timed_out ? "TIMEOUT" : "INFEASIBLE";
    }

    if (res.optimal) {
        write_instance_line_ccpo(instances_out, instance_id, shared, res.objective);
        for (const auto& rec : records)
            write_node_line_ccpo(nodes_out, instance_id, rec);
    }

    return res;
}
