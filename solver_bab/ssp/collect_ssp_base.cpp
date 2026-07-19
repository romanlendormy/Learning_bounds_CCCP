/*
 * collect_ssp_base.cpp — B&B Gecode CP pur (SSP) avec collecte GNN.
 */

#include "collect_ssp_base.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <limits>
#include <tuple>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// collect_node_state_ssp
// ─────────────────────────────────────────────────────────────────────────────

NodeRecordSSP collect_node_state_ssp(
    const SSPSpaceBnB* space,
    const SSPSharedData& data,
    int depth)
{
    NodeRecordSSP rec;
    rec.depth = depth;

    const int E = data.E, T = data.T;
    rec.x_mask.assign(E * T, 4);
    rec.curr_cost = 0.0;

    for (int e = 0; e < E; ++e) {
        for (int t = 0; t < T; ++t) {
            const IntVar& v = space->x_flat[e * T + t];
            if (v.assigned()) {
                int val = v.val();
                rec.x_mask[e * T + t] = static_cast<int8_t>(val);
                if (val == SSP_BAB_SIGMA_W)
                    rec.curr_cost += data.salary[t];
            }
        }
    }

    rec.z_mask.assign(data.N, 2);
    for (int i = 0; i < data.N; ++i)
        if (space->z[i].assigned())
            rec.z_mask[i] = static_cast<int8_t>(space->z[i].val());

    return rec;
}

// ─────────────────────────────────────────────────────────────────────────────
// write_instance_line_ssp
// ─────────────────────────────────────────────────────────────────────────────

void write_instance_line_ssp(
    std::ofstream& out,
    int instance_id,
    const SSPInstance& inst,
    double opt)
{
    int n_states = inst.initial + 1;
    for (auto& [src, sym, dst] : inst.transitions)
        n_states = std::max({n_states, src + 1, dst + 1});
    for (int q : inst.accepting)
        n_states = std::max(n_states, q + 1);

    out << std::fixed << std::setprecision(6);
    out << instance_id << ',' << inst.E << ',' << inst.T << ',' << inst.N << ','
        << inst.epsilon << ',' << opt;

    out << ',' << inst.initial;
    out << ',' << n_states;
    out << ',' << inst.accepting.size();
    for (int q : inst.accepting) out << ',' << q;

    out << ',' << inst.transitions.size();
    for (auto& [src, sym, dst] : inst.transitions)
        out << ',' << src << ',' << sym << ',' << dst;

    for (int t = 0; t < inst.T; ++t)
        out << ',' << inst.salary[t];

    for (int i = 0; i < inst.N; ++i)
        for (int t = 0; t < inst.T; ++t)
            out << ',' << inst.demand[i][t];

    out << '\n';
}

// ─────────────────────────────────────────────────────────────────────────────
// write_node_line_ssp  (Format C : 3 + E*T + N champs)
// ─────────────────────────────────────────────────────────────────────────────

void write_node_line_ssp(
    std::ofstream& out,
    int instance_id,
    const NodeRecordSSP& rec)
{
    out << std::fixed << std::setprecision(6);
    out << instance_id << ',' << rec.depth << ',' << rec.curr_cost;

    for (int8_t v : rec.x_mask) out << ',' << (int)v;
    for (int8_t v : rec.z_mask) out << ',' << (int)v;

    out << '\n';
}

// ─────────────────────────────────────────────────────────────────────────────
// solve_gecode_bab_ssp_collect_base
// ─────────────────────────────────────────────────────────────────────────────

SSPCollectResult solve_gecode_bab_ssp_collect_base(
    const SSPInstance& inst,
    std::ofstream& instances_out,
    std::ofstream& nodes_out,
    int       instance_id,
    double    time_lim,
    int       max_depth_collect)
{
    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();

    SSPSharedData shared(inst.E, inst.T, inst.N, inst.epsilon,
                         inst.transitions, inst.accepting, inst.initial,
                         inst.salary, inst.demand, inst.p);
    shared.use_lagrangian_bound = false;
    shared.branch_heuristic     = SSPBranchHeuristic::SIZE_MIN;
    shared.time_lim             = time_lim;
    shared.t_start               = t0;

    auto time_ok = [&]() {
        if (time_lim <= 0.0) return true;
        return std::chrono::duration<double>(Clock::now() - t0).count() < time_lim;
    };

    using NodeTup = std::tuple<double, int, SSPSpaceBnB*>;  // (lb, depth, space)
    std::vector<NodeTup> open;

    int64_t nodes = 0, fails = 0, max_depth = 0;
    double  best_val = std::numeric_limits<double>::infinity();
    SSPSpaceBnB* best = nullptr;

    std::vector<NodeRecordSSP> records;

    {
        SSPSpaceBnB* root = new SSPSpaceBnB(&shared);
        SpaceStatus rst = root->status();
        if (rst == SS_BRANCH) {
            double lb = root->total_cost.min() / (double)SSPSharedData::SCALE;
            open.push_back({lb, 0, root});
        } else if (rst == SS_SOLVED) {
            best_val = root->total_cost.val() / (double)SSPSharedData::SCALE;
            best = root;
        } else {
            delete root;
        }
    }

    while (!open.empty() && time_ok()) {
        auto [lb, depth, space] = open.back(); open.pop_back();

        if (lb >= best_val - 1e-6) { ++fails; delete space; continue; }

        SpaceStatus st = space->status();

        if (st == SS_FAILED) { ++fails; delete space; continue; }

        if (st == SS_SOLVED) {
            ++nodes;
            double val = space->total_cost.val() / (double)SSPSharedData::SCALE;
            if (val < best_val - 1e-9) {
                best_val = val; delete best; best = space; space = nullptr;
            } else {
                delete space;
            }
            continue;
        }

        // SS_BRANCH — collecter l'état avant branchement
        ++nodes;
        if (depth > max_depth) max_depth = depth;

        if (max_depth_collect < 0 || depth <= max_depth_collect)
            records.push_back(collect_node_state_ssp(space, shared, depth));

        const Gecode::Choice* ch = space->choice();
        unsigned int n_alt = ch->alternatives();

        for (unsigned int a = 0; a < n_alt; ++a) {
            SSPSpaceBnB* child = static_cast<SSPSpaceBnB*>(space->clone());
            child->commit(*ch, a);
            SpaceStatus cst = child->status();

            if (cst == SS_FAILED) { ++fails; delete child; continue; }

            double clb = child->total_cost.min() / (double)SSPSharedData::SCALE;
            if (clb >= best_val - 1e-6) { ++fails; delete child; continue; }

            if (cst == SS_SOLVED) {
                double val = child->total_cost.val() / (double)SSPSharedData::SCALE;
                if (val < best_val - 1e-9) {
                    best_val = val; delete best; best = child; child = nullptr;
                } else {
                    delete child;
                }
                continue;
            }

            open.push_back({clb, depth + 1, child});
        }

        delete ch;
        delete space;
    }

    bool timed_out = !open.empty();

    while (!open.empty()) {
        auto [lb, d, sp] = open.back(); open.pop_back();
        delete sp;
    }

    double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();

    SSPCollectResult res;
    res.solve_time_s = elapsed;
    res.nodes        = nodes;
    res.fails        = fails;
    res.depth        = max_depth;

    if (best) {
        res.objective = best->total_cost.val() / (double)SSPSharedData::SCALE;
        res.feasible  = true;
        res.optimal   = !timed_out;
        res.status    = timed_out ? "FEASIBLE" : "OPTIMAL";
        delete best;
    } else {
        res.objective = std::numeric_limits<double>::quiet_NaN();
        res.feasible  = false;
        res.optimal   = false;
        res.status    = timed_out ? "TIMEOUT" : "INFEASIBLE";
    }

    if (res.optimal) {
        write_instance_line_ssp(instances_out, instance_id, inst, res.objective);
        for (const auto& rec : records)
            write_node_line_ssp(nodes_out, instance_id, rec);
    }

    return res;
}
