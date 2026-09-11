/*
 * collect_gecode_base.cpp — B&B Gecode CP pur avec collecte GNN.
 */

#include "collect_gecode_base.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <limits>
#include <numeric>
#include <tuple>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// collect_node_state_base
// ─────────────────────────────────────────────────────────────────────────────

NodeRecordBase collect_node_state_base(
    const MKPSpaceBase* space,
    const MKPSharedData& data,
    int depth)
{
    NodeRecordBase rec;
    rec.depth = depth;

    rec.x_mask.assign(data.n, 2);
    rec.curr_cost = 0.0;
    for (int k = 0; k < data.n; ++k) {
        if (space->x[k].assigned()) {
            rec.x_mask[k] = static_cast<int8_t>(space->x[k].val());
            if (rec.x_mask[k] == 1) rec.curr_cost += data.c[k];
        }
    }

    rec.z_mask.assign(data.N, 2);
    for (int i = 0; i < data.N; ++i)
        if (space->z[i].assigned())
            rec.z_mask[i] = static_cast<int8_t>(space->z[i].val());

    return rec;
}

// ─────────────────────────────────────────────────────────────────────────────
// write_instance_line_base  (format identique à gecode_bab_collect)
// ─────────────────────────────────────────────────────────────────────────────

void write_instance_line_base(
    std::ofstream& out,
    int instance_id,
    const MKPSharedData& data,
    double opt)
{
    out << std::fixed << std::setprecision(6);
    out << instance_id << ',' << data.m << ',' << data.n << ','
        << data.N << ',' << data.epsilon << ',' << opt;

    for (int k = 0; k < data.n; ++k)
        out << ',' << data.c[k];

    for (int i = 0; i < data.N; ++i)
        for (int j = 0; j < data.m; ++j)
            out << ',' << data.b[i][j];

    for (int i = 0; i < data.N; ++i)
        for (int j = 0; j < data.m; ++j)
            for (int k = 0; k < data.n; ++k)
                out << ',' << data.A[i][j][k];

    out << '\n';
}

// ─────────────────────────────────────────────────────────────────────────────
// write_node_line_base  (Format C : 3 + n + N champs, sans ub_node/incumbent)
// ─────────────────────────────────────────────────────────────────────────────

void write_node_line_base(
    std::ofstream& out,
    int instance_id,
    const NodeRecordBase& rec)
{
    out << std::fixed << std::setprecision(6);
    out << instance_id << ',' << rec.depth << ',' << rec.curr_cost;

    for (int8_t v : rec.x_mask) out << ',' << (int)v;
    for (int8_t v : rec.z_mask) out << ',' << (int)v;

    out << '\n';
}

// ─────────────────────────────────────────────────────────────────────────────
// solve_gecode_bab_mkp_collect_base
// ─────────────────────────────────────────────────────────────────────────────

GecodeSLDResult solve_gecode_bab_mkp_collect_base(
    int n, int m, int N, double epsilon,
    const Vec1d& c, const Vec1d& p,
    const Vec3d& A, const Vec2d& b,
    std::ofstream& instances_out,
    std::ofstream& nodes_out,
    int       instance_id,
    double    time_lim,
    int       max_depth_collect)
{
    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();

    auto [greedy_val, greedy_x] = greedy_mkp(c, p, A, b, epsilon);
    int greedy_int = static_cast<int>(std::floor(greedy_val * MKPSharedData::SCALE));

    MKPSharedData shared(n, m, N, epsilon, c, p, A, b, /*T_node=*/0);
    shared.initial_lb_int = greedy_int;

    auto time_ok = [&]() {
        if (time_lim <= 0.0) return true;
        return std::chrono::duration<double>(Clock::now() - t0).count() < time_lim;
    };

    using NodeTup = std::tuple<double, int, MKPSpaceBase*>;
    std::vector<NodeTup> open;

    int64_t nodes          = 0;
    int64_t fails          = 0;
    int64_t max_depth      = 0;
    int64_t incumbent_impr = 0;
    double  best_val       = greedy_val;
    MKPSpaceBase* best     = nullptr;

    std::vector<NodeRecordBase> records;

    {
        MKPSpaceBase* root = new MKPSpaceBase(&shared);
        SpaceStatus rst = root->status();
        if (rst == SS_BRANCH) {
            double rub = root->obj.max() / (double)MKPSharedData::SCALE;
            open.push_back({rub, 0, root});
        } else if (rst == SS_SOLVED) {
            best_val = root->obj.val() / (double)MKPSharedData::SCALE;
            best = root; ++incumbent_impr;
        } else {
            delete root;
        }
    }

    while (!open.empty() && time_ok()) {
        auto [ub, depth, space] = open.back(); open.pop_back();

        if (ub <= best_val + 1e-6) { ++fails; delete space; continue; }

        SpaceStatus st = space->status();

        if (st == SS_FAILED) { ++fails; delete space; continue; }

        if (st == SS_SOLVED) {
            ++nodes;
            double val = space->obj.val() / (double)MKPSharedData::SCALE;
            if (val > best_val + 1e-9) {
                best_val = val; delete best; best = space; space = nullptr;
                ++incumbent_impr;
            } else { delete space; }
            continue;
        }

        // SS_BRANCH — collecter l'état avant branchement
        ++nodes;
        if (depth > max_depth) max_depth = depth;

        if (max_depth_collect < 0 || depth <= max_depth_collect)
            records.push_back(collect_node_state_base(space, shared, depth));

        const Gecode::Choice* ch = space->choice();

        for (int a = 0; a < 2; ++a) {
            MKPSpaceBase* child = static_cast<MKPSpaceBase*>(space->clone());
            child->commit(*ch, a);
            SpaceStatus cst = child->status();

            if (cst == SS_FAILED) { ++fails; delete child; continue; }

            double cub = child->obj.max() / (double)MKPSharedData::SCALE;
            if (cub <= best_val + 1e-6) { ++fails; delete child; continue; }

            if (cst == SS_SOLVED) {
                double val = child->obj.val() / (double)MKPSharedData::SCALE;
                if (val > best_val + 1e-9) {
                    best_val = val; delete best; best = child; child = nullptr;
                    ++incumbent_impr;
                } else { delete child; }
                continue;
            }

            open.push_back({cub, depth + 1, child});
        }

        delete ch;
        delete space;
    }

    bool timed_out = !open.empty();

    while (!open.empty()) {
        auto [ub, d, sp] = open.back(); open.pop_back();
        delete sp;
    }

    double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();

    GecodeSLDResult res;
    res.greedy_value          = greedy_val;
    res.best_open_ub          = 0.0;
    res.solve_time_s          = elapsed;
    res.nodes                 = nodes;
    res.fails                 = fails;
    res.fails_from_bound      = 0;
    res.fails_from_cp         = fails;
    res.depth                 = max_depth;
    res.incumbent_improvements = incumbent_impr;
    res.more_calls            = 0;
    res.ub_tightened          = 0;
    res.root_bound_time_s     = 0.0;
    res.t_bound_total_s       = 0.0;
    res.t_bab_pure_s          = elapsed;

    if (best) {
        res.objective = static_cast<double>(best->obj.val()) / shared.SCALE;
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
        write_instance_line_base(instances_out, instance_id, shared, res.objective);
        for (const auto& rec : records)
            write_node_line_base(nodes_out, instance_id, rec);
    }

    return res;
}
