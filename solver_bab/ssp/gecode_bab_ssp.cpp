/*
 * gecode_bab_ssp.cpp — Implémentation de solve_gecode_bab_ssp.
 *
 * Piloté par Gecode::BAB<> natif, comme solve_gecode_bab_mkp (solver_bab/mkp/
 * gecode_bab.cpp) pour MKP — pas de DFS manuelle (cf. commentaire en tête de
 * gecode_bab_ssp.hpp sur pourquoi une DFS maison a été abandonnée).
 */

#include "gecode_bab_ssp.hpp"

#include <limits>
#include <tuple>
#include <vector>

GecodeSSPBabResult solve_gecode_bab_ssp(
    int E, int T, int N, double epsilon,
    const std::vector<std::tuple<int,int,int>>& transitions,
    const std::vector<int>& accepting, int initial,
    const Vec1d& salary, const Vec2i& demand, Vec1d p,
    double       time_lim,
    int          T_root,
    int          T_node,
    SSPBoundMode bound_mode,
    double       rel_tol,
    int          tol_window,
    bool         use_lagrangian_bound)
{
    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();

    SSPSharedData shared(E, T, N, epsilon, transitions, accepting, initial,
                         salary, demand, std::move(p));
    shared.T_root               = T_root;
    shared.T_node               = T_node;
    shared.bound_mode           = bound_mode;
    shared.rel_tol              = rel_tol;
    shared.tol_window           = tol_window;
    shared.time_lim             = time_lim;
    shared.t_start              = t0;
    shared.use_lagrangian_bound = use_lagrangian_bound;

    SSPSpaceBnB* model = new SSPSpaceBnB(&shared);

    Search::Options opts;
    Search::TimeStop* stop = nullptr;
    if (time_lim > 0.0) {
        stop      = new Search::TimeStop(static_cast<unsigned long long>(time_lim * 1000));
        opts.stop = stop;
    }

    BAB<SSPSpaceBnB> engine(model, opts);
    delete model;

    SSPSpaceBnB* best = nullptr;
    int64_t inc_impr = 0;
    while (SSPSpaceBnB* sol = engine.next()) { delete best; best = sol; ++inc_impr; }

    bool timed_out = stop && engine.stopped();
    auto stats = engine.statistics();
    delete stop;

    double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();

    GecodeSSPBabResult res;
    res.solve_time_s           = elapsed;
    res.nodes                  = static_cast<int64_t>(stats.node);
    res.fails                  = static_cast<int64_t>(stats.fail);
    res.depth                  = static_cast<int64_t>(stats.depth);
    res.incumbent_improvements = inc_impr;
    res.more_calls             = shared.more_calls;
    res.root_bound_time_s      = shared.root_bound_time_s;
    res.t_bound_total_s        = shared.t_bound_total_s;
    res.t_bab_pure_s           = res.solve_time_s - shared.t_bound_total_s;

    if (best) {
        res.objective = static_cast<double>(best->total_cost.val()) / SSPSharedData::SCALE;
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
    return res;
}
