/*
 * kp_gecode_bab.cpp — Implémentation de greedy_kp et solve_gecode_bab_kp.
 *
 * Adaptation de solver_bab/mkp/gecode_bab.cpp au cas m=1. Moteur DFS identique
 * (pile LIFO gérée à la main via status()/choice()/commit()/clone()) — cf.
 * gecode_bab.cpp pour la discussion complète de la séquence par nœud.
 */

#include "kp_gecode_bab.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>
#include <tuple>
#include <vector>

// ═════════════════════════════════════════════════════════════════════════════
// greedy_kp
// ═════════════════════════════════════════════════════════════════════════════

std::pair<double, std::vector<int8_t>> greedy_kp(
    const Vec1d& c, const Vec1d& p, const Vec2d& A, const Vec1d& b, double epsilon)
{
    const int n = static_cast<int>(c.size());
    const int N = static_cast<int>(p.size());

    // Tri des items par ratio c_k / poids_surrogate décroissant.
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int bk) {
        double wa = 0.0, wb = 0.0;
        for (int i = 0; i < N; ++i)
            if (b[i] > 1e-9) {
                wa += p[i] * A[i][a]  / b[i];
                wb += p[i] * A[i][bk] / b[i];
            }
        double ra = (wa > 1e-12) ? c[a]  / wa  : c[a]  * 1e12;
        double rb = (wb > 1e-12) ? c[bk] / wb  : c[bk] * 1e12;
        return ra > rb;
    });

    // Vérification de faisabilité : P(Σ A_i[k] x_k > b_i) ≤ ε
    auto feasible = [&](const std::vector<int8_t>& x) {
        double pv = 0.0;
        for (int i = 0; i < N; ++i) {
            double load = 0.0;
            for (int k = 0; k < n; ++k)
                if (x[k] == 1) load += A[i][k];
            if (load > b[i] + 1e-9) pv += p[i];
            if (pv > epsilon + 1e-9) return false;
        }
        return true;
    };

    std::vector<int8_t> x(n, 0);
    double val = 0.0;
    for (int k : order) {
        if (c[k] <= 0.0) continue;
        x[k] = 1;
        if (!feasible(x)) x[k] = 0;
        else              val  += c[k];
    }
    return {val, x};
}

// ═════════════════════════════════════════════════════════════════════════════
// solve_gecode_bab_kp
// ═════════════════════════════════════════════════════════════════════════════

GecodeKPResult solve_gecode_bab_kp(
    int n, int N, double epsilon,
    const Vec1d& c, const Vec1d& p,
    const Vec2d& A, const Vec1d& b,
    double      time_lim,
    int         T_root,
    int         T_node,
    KPBoundMode bound_mode,
    bool        patience_mode,
    double      patience_tol)
{
    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();

    // ── 1. Solution gloutonne — incumbent initial ─────────────────────────
    auto [greedy_val, greedy_x] = greedy_kp(c, p, A, b, epsilon);
    int greedy_int = static_cast<int>(std::floor(greedy_val * KPSharedData::SCALE));

    // ── 2. Données partagées ────────────────────────────────────────────────
    KPSharedData shared(n, N, epsilon, c, p, A, b, T_node);
    shared.T_root         = T_root;
    shared.bound_mode     = bound_mode;
    shared.patience_mode  = patience_mode;
    shared.patience_tol   = patience_tol;
    shared.initial_lb_int = greedy_int;

    // ── 3. DFS B&B avec Gecode Spaces (pile LIFO maison) ───────────────────
    auto time_ok = [&]() {
        if (time_lim <= 0.0) return true;
        return std::chrono::duration<double>(Clock::now() - t0).count() < time_lim;
    };

    using NodeTup = std::tuple<double, int, KPSpaceV1LD*>;  // (ub, depth, space)
    std::vector<NodeTup> open;

    int64_t nodes          = 0;
    int64_t fails          = 0;
    int64_t max_depth      = 0;
    int64_t incumbent_impr = 0;
    double  best_val       = greedy_val;
    KPSpaceV1LD* best      = nullptr;

    {
        KPSpaceV1LD* root = new KPSpaceV1LD(&shared);
        SpaceStatus rst = root->status();
        if (rst == SS_BRANCH) {
            double rub = root->obj.max() / (double)KPSharedData::SCALE;
            open.push_back({rub, 0, root});
        } else if (rst == SS_SOLVED) {
            best_val = root->obj.val() / (double)KPSharedData::SCALE;
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
            double val = space->obj.val() / (double)KPSharedData::SCALE;
            if (val > best_val + 1e-9) {
                best_val = val; delete best; best = space; space = nullptr;
                ++incumbent_impr;
            } else { delete space; }
            continue;
        }

        ++nodes;
        if (depth > max_depth) max_depth = depth;

        const Gecode::Choice* c_choice = space->choice();

        for (int a = 0; a < 2; ++a) {
            KPSpaceV1LD* child = static_cast<KPSpaceV1LD*>(space->clone());
            child->commit(*c_choice, a);
            SpaceStatus cst = child->status();

            if (cst == SS_FAILED) { ++fails; delete child; continue; }

            double cub = child->obj.max() / (double)KPSharedData::SCALE;
            if (cub <= best_val + 1e-6) { ++fails; delete child; continue; }

            if (cst == SS_SOLVED) {
                double val = child->obj.val() / (double)KPSharedData::SCALE;
                if (val > best_val + 1e-9) {
                    best_val = val; delete best; best = child; child = nullptr;
                    ++incumbent_impr;
                } else { delete child; }
                continue;
            }

            open.push_back({cub, depth + 1, child});
        }

        delete c_choice;
        delete space;
    }

    bool timed_out = !open.empty();

    while (!open.empty()) {
        auto [ub, d, sp] = open.back(); open.pop_back();
        delete sp;
    }

    double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();

    // ── 4. Construction du résultat ─────────────────────────────────────────
    GecodeKPResult res;
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
    res.ub_tightened     = static_cast<int64_t>(shared.ub_tightened);
    res.root_bound_time_s = shared.root_bound_time_s;
    res.root_bound_value  = shared.root_bound_value;
    res.t_bound_total_s   = shared.t_bound_total_s;
    res.t_bab_pure_s      = res.solve_time_s - shared.t_bound_total_s;
    res.clone_count       = shared.clone_count;
    res.t_clone_state_s   = shared.t_clone_state_s;
    res.t_clone_total_s   = shared.t_clone_total_s;

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

    return res;
}
