/*
 * gecode_bab_collect.cpp — B&B Gecode avec collecte de données GNN.
 */

#include "gecode_bab_collect.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <limits>
#include <numeric>
#include <tuple>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// collect_node_state
// ─────────────────────────────────────────────────────────────────────────────

NodeRecord collect_node_state(
    const MKPSpaceV1LD* space,
    const MKPSharedData& data,
    int depth)
{
    NodeRecord rec;
    rec.depth = depth;
    rec.fix_ones.assign(data.n, 0);

    // Variables fixées à 1 (par commit explicite ou propagation CP)
    std::vector<int> fixed_one;
    fixed_one.reserve(data.n);
    for (int k = 0; k < data.n; ++k) {
        if (space->x[k].assigned() && space->x[k].val() == 1) {
            rec.fix_ones[k] = 1;
            fixed_one.push_back(k);
        }
    }

    // Profit accumulé
    rec.fix_bound = 0.0;
    for (int k : fixed_one) rec.fix_bound += data.c[k];

    // Capacités résiduelles et scénarios violés
    rec.b_res.assign(data.N, Vec1d(data.m, 0.0));
    rec.violated_count = 0;
    for (int i = 0; i < data.N; ++i) {
        bool violated = false;
        for (int j = 0; j < data.m; ++j) {
            double consumed = 0.0;
            for (int k : fixed_one) consumed += data.A[i][j][k];
            rec.b_res[i][j] = data.b[i][j] - consumed;
            if (rec.b_res[i][j] < -1e-9) violated = true;
        }
        if (violated) ++rec.violated_count;
    }

    return rec;
}

// ─────────────────────────────────────────────────────────────────────────────
// write_instance_line
// ─────────────────────────────────────────────────────────────────────────────

void write_instance_line(
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
// write_node_line
// ─────────────────────────────────────────────────────────────────────────────

void write_node_line(
    std::ofstream& out,
    int instance_id,
    const NodeRecord& rec)
{
    out << std::fixed << std::setprecision(6);
    out << instance_id << ',' << rec.depth << ','
        << rec.fix_bound << ',' << rec.violated_count;

    for (int v : rec.fix_ones)
        out << ',' << (int)v;

    for (const auto& row : rec.b_res)
        for (double val : row)
            out << ',' << val;

    out << '\n';
}

// ─────────────────────────────────────────────────────────────────────────────
// solve_gecode_bab_mkp_collect
// ─────────────────────────────────────────────────────────────────────────────

GecodeV1LDResult solve_gecode_bab_mkp_collect(
    int n, int m, int N, double epsilon,
    const Vec1d& c, const Vec1d& p,
    const Vec3d& A, const Vec2d& b,
    std::ofstream& instances_out,
    std::ofstream& nodes_out,
    int       instance_id,
    double    time_lim,
    int       T_root,
    int       T_node,
    BoundMode bound_mode,
    bool      patience_mode,
    double    patience_tol,
    int       max_depth_collect)
{
    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();

    auto [greedy_val, greedy_x] = greedy_mkp(c, p, A, b, epsilon);
    int greedy_int = static_cast<int>(std::floor(greedy_val * MKPSharedData::SCALE));

    MKPSharedData shared(n, m, N, epsilon, c, p, A, b, T_node);
    shared.T_root         = T_root;
    shared.bound_mode     = bound_mode;
    shared.patience_mode  = patience_mode;
    shared.patience_tol   = patience_tol;
    shared.initial_lb_int = greedy_int;

    auto time_ok = [&]() {
        if (time_lim <= 0.0) return true;
        return std::chrono::duration<double>(Clock::now() - t0).count() < time_lim;
    };

    using NodeTup = std::tuple<double, int, MKPSpaceV1LD*>;
    std::vector<NodeTup> open;

    int64_t nodes          = 0;
    int64_t fails          = 0;
    int64_t max_depth      = 0;
    int64_t incumbent_impr = 0;
    double  best_val       = greedy_val;
    MKPSpaceV1LD* best     = nullptr;

    // Buffer de collecte — rempli pendant le B&B, écrit à la fin
    std::vector<NodeRecord> records;

    {
        MKPSpaceV1LD* root = new MKPSpaceV1LD(&shared);
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

        // SS_BRANCH — collecter l'état de ce nœud avant branchement
        ++nodes;
        if (depth > max_depth) max_depth = depth;

        if (max_depth_collect < 0 || depth <= max_depth_collect)
            records.push_back(collect_node_state(space, shared, depth));

        const Gecode::Choice* ch = space->choice();

        for (int a = 0; a < 2; ++a) {
            MKPSpaceV1LD* child = static_cast<MKPSpaceV1LD*>(space->clone());
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

    GecodeV1LDResult res;
    res.greedy_value     = greedy_val;
    res.best_open_ub     = shared.best_open_ub;
    res.solve_time_s     = elapsed;
    res.nodes            = nodes;
    res.fails            = fails;
    res.fails_from_bound = static_cast<int64_t>(shared.fails_from_bound);
    res.fails_from_cp    = fails - res.fails_from_bound;
    res.depth            = max_depth;
    res.incumbent_improvements = incumbent_impr;
    res.more_calls       = static_cast<int64_t>(shared.more_calls);
    res.ub_tightened       = static_cast<int64_t>(shared.ub_tightened);
    res.root_bound_time_s  = shared.root_bound_time_s;
    res.t_bound_total_s    = shared.t_bound_total_s;
    res.t_bab_pure_s       = res.solve_time_s - shared.t_bound_total_s;

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

    // Écriture dans les fichiers uniquement si l'optimalité est prouvée
    if (res.optimal) {
        write_instance_line(instances_out, instance_id, shared, res.objective);
        for (const auto& rec : records)
            write_node_line(nodes_out, instance_id, rec);
    }

    return res;
}
