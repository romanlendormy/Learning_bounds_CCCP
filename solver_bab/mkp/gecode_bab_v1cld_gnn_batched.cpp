/*
 * gecode_bab_v1cld_gnn_batched.cpp — Implémentation de
 * solve_gecode_bab_mkp_v1cld_gnn_batched. Port QUASI-VERBATIM de
 * gecode_bab_gnn_batched.cpp (V1LD) — même boucle DFS + bucketing par n_g,
 * même MKPSpaceGNNBatched/PendingEntry/greedy_mkp_local (réutilisés/copiés
 * à l'identique) — SEULE différence : le backend instancié
 * (GNNV1CLDSdEditNativeBatched) et, en conséquence, l'absence des variantes
 * AOTInductor (cf. doc de header).
 */

#include "gecode_bab_v1cld_gnn_batched.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <vector>

// ── greedy_mkp — copie locale (même convention que gecode_bab_gnn_batched.cpp) ──
namespace {

std::pair<double, std::vector<int8_t>> greedy_mkp_local(
    const Vec1d& c, const Vec1d& p, const Vec3d& A, const Vec2d& b, double epsilon)
{
    const int n = static_cast<int>(c.size());
    const int N = static_cast<int>(p.size());
    const int m = static_cast<int>(b[0].size());

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int bk) {
        double wa = 0.0, wb = 0.0;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < m; ++j)
                if (b[i][j] > 1e-9) {
                    wa += p[i] * A[i][j][a]  / b[i][j];
                    wb += p[i] * A[i][j][bk] / b[i][j];
                }
        double ra = (wa > 1e-12) ? c[a]  / wa  : c[a]  * 1e12;
        double rb = (wb > 1e-12) ? c[bk] / wb  : c[bk] * 1e12;
        return ra > rb;
    });

    auto feasible = [&](const std::vector<int8_t>& xv) {
        double pv = 0.0;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < m; ++j) {
                double load = 0.0;
                for (int k = 0; k < n; ++k)
                    if (xv[k] == 1) load += A[i][j][k];
                if (load > b[i][j] + 1e-9) { pv += p[i]; break; }
            }
            if (pv > epsilon + 1e-9) return false;
        }
        return true;
    };

    std::vector<int8_t> xv(n, 0);
    double val = 0.0;
    for (int k : order) {
        if (c[k] <= 0.0) continue;
        xv[k] = 1;
        if (!feasible(xv)) xv[k] = 0;
        else               val  += c[k];
    }
    return {val, xv};
}

struct PendingEntry {
    std::vector<int8_t> x_mask;
    double               fp;
    Vec2d                b_res;
    int                   outer_depth;
    MKPSpaceGNNBatched*   space;
};

// Signal d'arrêt anticipé (2026-08-24) : utilisé par stop_after_first_level
// (cf. header) pour interrompre le B&B dès que first_level_bound_value est
// capturé, sans attendre la fin de la résolution — permet de mesurer le
// "gap à la racine" en un temps constant (indépendant de n/N) au lieu du
// temps de résolution complet, en réutilisant l'optimum déjà connu (ref) côté
// appelant plutôt que res.objective (jamais atteint dans ce mode).
struct FirstLevelCapturedSignal {};

} // namespace

GNNBatchedBabResult solve_gecode_bab_mkp_v1cld_gnn_batched(
    int n, int m, int N, double epsilon,
    const Vec1d& c, const Vec1d& p, const Vec3d& A, const Vec2d& b,
    double time_lim, int K,
    std::string gnn_checkpoint_path, std::string gnn_device,
    double gnn_max_depth_frac,
    bool use_surrogate_zeta,
    int gnn_node_stride,
    std::string gnn_backend,
    bool stop_after_first_level)
{
    if (K < 1)
        throw std::invalid_argument("solve_gecode_bab_mkp_v1cld_gnn_batched : K doit être >= 1");
    if (gnn_node_stride < 1)
        throw std::invalid_argument("solve_gecode_bab_mkp_v1cld_gnn_batched : gnn_node_stride doit être >= 1");

    const int max_depth_cutoff = static_cast<int>(std::floor(gnn_max_depth_frac * n));
    int64_t nodes_bypassed_depth_cutoff = 0;
    int64_t gnn_call_counter = 0;
    int64_t nodes_bypassed_stride = 0;

    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();

    auto [greedy_val, greedy_x] = greedy_mkp_local(c, p, A, b, epsilon);
    int greedy_int = static_cast<int>(std::floor(greedy_val * MKPSharedData::SCALE));

    MKPSharedData shared(n, m, N, epsilon, c, p, A, b);
    shared.initial_lb_int = greedy_int;

    std::unique_ptr<GNNBoundBackendBatch> gnn_client;
    if (gnn_backend == "native")
        gnn_client = std::make_unique<GNNV1CLDSdEditNativeBatched>(
            gnn_checkpoint_path, gnn_device, n, m, N, epsilon, c, p, A, b, use_surrogate_zeta);
    else
        throw std::invalid_argument(
            "solve_gecode_bab_mkp_v1cld_gnn_batched : gnn_backend inconnu (\"" + gnn_backend +
            "\") — seul \"native\" est supporté pour V1CLD pour l'instant (cf. doc de header)");

    auto time_ok = [&]() {
        if (time_lim <= 0.0) return true;
        return std::chrono::duration<double>(Clock::now() - t0).count() < time_lim;
    };

    using NodeTup = std::tuple<double, int, MKPSpaceGNNBatched*>;
    std::vector<NodeTup> open;
    std::vector<std::vector<PendingEntry>> pending(n + 1);

    int64_t nodes = 0, fails = 0, max_depth = 0, incumbent_impr = 0;
    int64_t dbg_zstats_total_xnodes = 0, dbg_zstats_nodes_with_fixed_z = 0;
    int64_t dbg_zstats_sum_fixed = 0, dbg_zstats_sum_fixed_z1 = 0, dbg_zstats_max_fixed = 0;
    double  best_val = greedy_val;
    MKPSpaceGNNBatched* best = nullptr;

    int64_t n_bucket_flushes = 0, sum_bucket_fill_at_flush = 0, max_pending_live = 0;
    double t_gnn_graph_build_s = 0.0, t_gnn_forward_s = 0.0, t_gnn_bound_compute_s = 0.0;

    double  first_level_bound_value     = std::numeric_limits<double>::quiet_NaN();
    int64_t first_level_bound_n_members = 0;
    bool    first_level_captured        = false;

    auto count_pending_live = [&]() {
        int64_t total = 0;
        for (const auto& bucket : pending) total += static_cast<int64_t>(bucket.size());
        return total;
    };
    auto any_pending_nonempty = [&]() {
        for (const auto& bucket : pending) if (!bucket.empty()) return true;
        return false;
    };

    auto flush_bucket = [&](int d) {
        auto& bucket = pending[d];
        if (bucket.empty()) return;

        ++n_bucket_flushes;
        sum_bucket_fill_at_flush += static_cast<int64_t>(bucket.size());
        const int B = static_cast<int>(bucket.size());

        std::vector<std::vector<int8_t>> x_masks(B);
        std::vector<double> curr_costs(B);
        std::vector<Vec2d> b_res_list(B);
        for (int i = 0; i < B; ++i) {
            x_masks[i] = bucket[i].x_mask; curr_costs[i] = bucket[i].fp; b_res_list[i] = bucket[i].b_res;
        }

        int gnn_depth = n - d;
        std::vector<double> ub_relative;
        GNNBatchCallTiming timing;
        gnn_client->query_batch(x_masks, curr_costs, b_res_list, gnn_depth, ub_relative, timing);
        t_gnn_graph_build_s   += timing.t_graph_build_s;
        t_gnn_forward_s       += timing.t_forward_s;
        t_gnn_bound_compute_s += timing.t_bound_compute_s;

        const double snapshot_best = best_val;
        double best_leaf_val = -std::numeric_limits<double>::infinity();
        MKPSpaceGNNBatched* best_leaf_space = nullptr;

        const bool capture_first_level = (gnn_depth == 1 && !first_level_captured);
        double first_level_max_ub = -std::numeric_limits<double>::infinity();

        for (int i = 0; i < B; ++i) {
            MKPSpaceGNNBatched* sp = bucket[i].space;
            double ub = bucket[i].fp + ub_relative[i];
            if (capture_first_level) first_level_max_ub = std::max(first_level_max_ub, ub);

            if (std::getenv("GNN_TRACE_DEBUG")) {
                const int nn = static_cast<int>(x_masks[i].size());
                std::string mstr; mstr.reserve(nn);
                for (int k = 0; k < nn; ++k) mstr += static_cast<char>('0' + x_masks[i][k]);
                std::fprintf(stderr, "TRACE,%s,%d,%.6f,%.6f,%.6f\n",
                             mstr.c_str(), gnn_depth, curr_costs[i], ub_relative[i], ub);
            }

            int ub_int = static_cast<int>(std::ceil(ub * MKPSharedData::SCALE));
            sp->post_gnn_bound(ub_int);
            SpaceStatus cst = sp->status();

            if (cst == SS_FAILED) { ++fails; delete sp; continue; }

            double cub = sp->obj.max() / (double)MKPSharedData::SCALE;

            if (cst == SS_SOLVED) {
                double val = sp->obj.val() / (double)MKPSharedData::SCALE;
                if (val > best_leaf_val) {
                    if (best_leaf_space) delete best_leaf_space;
                    best_leaf_val = val; best_leaf_space = sp;
                } else delete sp;
                continue;
            }

            if (cub <= snapshot_best + 1e-6) { ++fails; delete sp; continue; }
            open.push_back({cub, bucket[i].outer_depth, sp});
        }
        bucket.clear();

        if (capture_first_level && B > 0) {
            first_level_bound_value     = first_level_max_ub;
            first_level_bound_n_members = B;
            first_level_captured        = true;
            if (stop_after_first_level) throw FirstLevelCapturedSignal{};
        }

        if (best_leaf_space) {
            if (best_leaf_val > best_val + 1e-9) {
                best_val = best_leaf_val; delete best; best = best_leaf_space; ++incumbent_impr;
            } else delete best_leaf_space;
        }
    };

    bool stopped_after_first_level = false;

    try {
    {
        MKPSpaceGNNBatched* root = new MKPSpaceGNNBatched(&shared);
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

    while ((!open.empty() || any_pending_nonempty()) && time_ok()) {
        if (open.empty()) {
            for (int d = 0; d <= n; ++d) flush_bucket(d);
            continue;
        }

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

        ++nodes;
        if (depth > max_depth) max_depth = depth;

        const Gecode::Choice* c_choice = space->choice();
        const bool parent_has_free_x = space->has_free_x();

        for (int a = 0; a < 2; ++a) {
            MKPSpaceGNNBatched* child = static_cast<MKPSpaceGNNBatched*>(space->clone());
            child->commit(*c_choice, a);

            if (parent_has_free_x) {
                const auto& pv = static_cast<const MKPSpaceGNNBatched::SimpleBrancher::PosVal&>(*c_choice);
                int val = (a == 0) ? 1 : 0;
                if (val == 1) child->apply_item_fixed(pv.pos);
            }

            SpaceStatus cst = child->status();

            if (cst == SS_FAILED) { ++fails; delete child; continue; }

            std::vector<int8_t> x_mask; double fp;
            child->extract_pending(x_mask, fp);
            int n_free = 0;
            for (int8_t v : x_mask) if (v == 2) ++n_free;
            int ng_depth = n_free;

            // ── Diagnostic (lecture seule, aucun impact sur la borne/pruning) :
            // fréquence à laquelle des z sont déjà résolus par la propagation CP
            // PENDANT le x-branching (extract_pending ne les extrait pas —
            // question soulevée par l'utilisateur, cf. discussion). ──
            if (ng_depth > 0) {
                int nz1 = 0, nz0 = 0;
                for (int zi = 0; zi < child->data->N; ++zi)
                    if (child->z[zi].assigned()) { if (child->z[zi].val() == 1) ++nz1; else ++nz0; }
                ++dbg_zstats_total_xnodes;
                if (nz1 + nz0 > 0) ++dbg_zstats_nodes_with_fixed_z;
                dbg_zstats_sum_fixed    += (nz1 + nz0);
                dbg_zstats_sum_fixed_z1 += nz1;
                dbg_zstats_max_fixed = std::max(dbg_zstats_max_fixed, (int64_t)(nz1 + nz0));
            }

            const int depth_here = n - ng_depth;
            const bool depth_or_zbranch_bypass = (ng_depth == 0 || depth_here > max_depth_cutoff);

            bool stride_bypass = false;
            if (!depth_or_zbranch_bypass) {
                stride_bypass = (gnn_call_counter % gnn_node_stride) != 0;
                ++gnn_call_counter;
            }

            if (depth_or_zbranch_bypass || stride_bypass) {
                if (ng_depth > 0) {
                    if (stride_bypass) ++nodes_bypassed_stride;
                    else                ++nodes_bypassed_depth_cutoff;
                }
                double cub = child->obj.max() / (double)MKPSharedData::SCALE;
                if (cub <= best_val + 1e-6) { ++fails; delete child; continue; }
                if (cst == SS_SOLVED) {
                    if (cub > best_val + 1e-9) {
                        best_val = cub; delete best; best = child; child = nullptr;
                        ++incumbent_impr;
                    } else delete child;
                    continue;
                }
                open.push_back({cub, depth + 1, child});
                continue;
            }

            pending[ng_depth].push_back({std::move(x_mask), fp, child->b_res, depth + 1, child});
            max_pending_live = std::max(max_pending_live, count_pending_live());
            if (static_cast<int>(pending[ng_depth].size()) == K)
                flush_bucket(ng_depth);
        }

        delete c_choice;
        delete space;
    }
    } catch (const FirstLevelCapturedSignal&) {
        stopped_after_first_level = true;
    }

    std::fprintf(stderr,
        "[zstats] noeuds x-branching=%lld  avec>=1 z fixe=%lld (%.2f%%)  "
        "moyenne z fixes/noeud=%.3f  moyenne z=1 fixes/noeud=%.3f  max z fixes sur 1 noeud=%lld\n",
        (long long)dbg_zstats_total_xnodes, (long long)dbg_zstats_nodes_with_fixed_z,
        dbg_zstats_total_xnodes > 0 ? 100.0 * dbg_zstats_nodes_with_fixed_z / dbg_zstats_total_xnodes : 0.0,
        dbg_zstats_total_xnodes > 0 ? (double)dbg_zstats_sum_fixed / dbg_zstats_total_xnodes : 0.0,
        dbg_zstats_total_xnodes > 0 ? (double)dbg_zstats_sum_fixed_z1 / dbg_zstats_total_xnodes : 0.0,
        (long long)dbg_zstats_max_fixed);

    bool timed_out = !open.empty() || any_pending_nonempty();

    while (!open.empty()) { auto [ub, d, sp] = open.back(); open.pop_back(); delete sp; }
    for (auto& bucket : pending) { for (auto& e : bucket) delete e.space; bucket.clear(); }

    double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();

    GNNBatchedBabResult res;
    res.K = K;
    res.n_bucket_flushes = n_bucket_flushes;
    res.mean_bucket_fill_at_flush = n_bucket_flushes > 0
        ? static_cast<double>(sum_bucket_fill_at_flush) / static_cast<double>(n_bucket_flushes) : 0.0;
    res.max_pending_live = max_pending_live;
    res.nodes_bypassed_depth_cutoff = nodes_bypassed_depth_cutoff;
    res.nodes_bypassed_stride = nodes_bypassed_stride;
    res.first_level_bound_value     = first_level_bound_value;
    res.first_level_bound_n_members = first_level_bound_n_members;

    res.greedy_value = greedy_val;
    res.solve_time_s = elapsed;
    res.nodes = nodes;
    res.fails = fails;
    res.depth = max_depth;
    res.incumbent_improvements = incumbent_impr;
    res.t_gnn_graph_build_s = t_gnn_graph_build_s;
    res.t_gnn_forward_s     = t_gnn_forward_s;
    res.t_gnn_bound_compute_s = t_gnn_bound_compute_s;
    res.t_bound_total_s = t_gnn_graph_build_s + t_gnn_forward_s + t_gnn_bound_compute_s;
    res.t_bab_pure_s = res.solve_time_s - res.t_bound_total_s;

    if (best) {
        res.objective = static_cast<double>(best->obj.val()) / MKPSharedData::SCALE;
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
    // stop_after_first_level : res.objective ci-dessus n'est PAS l'optimum (à
    // usage diagnostique uniquement — meilleur incumbent trouvé avant l'arrêt
    // anticipé, cf. header) ; comparer first_level_bound_value à l'optimum
    // connu par ailleurs (ex. résultat ref), pas à res.objective.
    if (stopped_after_first_level) res.status = "FIRST_LEVEL_ONLY";

    return res;
}
