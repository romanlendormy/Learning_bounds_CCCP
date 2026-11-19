/*
 * gecode_bab_ccpo_gnn_batched.cpp — cf. gecode_bab_ccpo_gnn_batched.hpp.
 *
 * Boucle DFS dérivée de solve_gecode_bab_ccpo (gecode_bab_ccpo.cpp), avec
 * bucketing par n_free et flush batché (mirroir de
 * solve_gecode_bab_mkp_gnn_batched, côté MKP).
 */

#include "gecode_bab_ccpo_gnn_batched.hpp"

#include <omp.h>

#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>

namespace {

// Mêmes définitions que gnn_ccpo/train_ccpo_gnn.py::instance_scales (le GNN
// a été entraîné avec ces échelles -- DOIT rester synchronisé).
void compute_scales(const CCPOSharedData& d, double& c_scale, double& sigma_scale) {
    int cmax = 0;
    for (int v : d.c_int) cmax = std::max(cmax, std::abs(v));
    c_scale = std::max(static_cast<double>(cmax), 1.0);

    double acc = 0.0;
    int n_el = 0;
    for (int j = 0; j < d.N; ++j) {
        if (d.infeasible_scenario[j]) continue;
        double s = 0.0;
        for (int i = 0; i < d.n; ++i) s += d.sigma_dense[j][i][i];
        acc += s / d.n;
        ++n_el;
    }
    sigma_scale = (n_el == 0) ? 1.0 : std::max(acc / n_el, 1e-6);
}

}  // namespace

CCPOGNNBatchedResult solve_gecode_bab_ccpo_gnn_batched(
    int n, int N, int k, int Q, int L, int U, long long B, int min_feas,
    const std::vector<int>& c_int,
    const std::vector<std::vector<SigmaTriplet>>& sigma,
    double time_lim,
    int K,
    CCPOGNNBackendBatch& backend,
    int omp_threads)
{
    using Clock = std::chrono::steady_clock;
    const auto t0 = Clock::now();

    CCPOSharedData shared(n, N, k, Q, L, U, B, min_feas, c_int, sigma);
    shared.bound_mode = BoundMode::GNN_BATCHED;

    CCPOGNNBatchedResult out;

    if (shared.n_infeasible > N - min_feas) {
        out.feasible = false; out.optimal = false; out.status = "INFEASIBLE";
        out.solve_time_s = std::chrono::duration<double>(Clock::now() - t0).count();
        return out;
    }

    GreedyCCPOResult greedy = greedy_ccpo(n, N, k, Q, L, U, c_int, sigma, B, min_feas);
    if (greedy.feasible) shared.initial_lb_int = static_cast<int>(greedy.value);

    CCPOInstanceCtx ctx;
    ctx.n = n; ctx.N = N; ctx.L = L; ctx.U = U; ctx.k = k; ctx.Q = Q; ctx.min_feas = min_feas;
    compute_scales(shared, ctx.c_scale, ctx.sigma_scale);
    ctx.epsilon = 1.0 - static_cast<double>(min_feas) / N;
    ctx.B = static_cast<double>(B);
    ctx.infeasible = &shared.infeasible_scenario;

    auto time_ok = [&]() {
        if (time_lim <= 0.0) return true;
        return std::chrono::duration<double>(Clock::now() - t0).count() < time_lim;
    };

    using NodeTup = std::tuple<double, int, CCPOSpaceSLD*>;
    std::vector<NodeTup> open;

    struct Pending {
        CCPOSpaceSLD* space;
        int depth;
        CCPOQuery query;
    };
    std::vector<std::vector<Pending>> pending(n + 1);
    int64_t pending_live = 0;

    int64_t nodes = 0, fails = 0, max_depth = 0, incumbent_impr = 0;
    int64_t n_flushes = 0, fill_sum = 0;
    double  first_level_ub = -std::numeric_limits<double>::infinity();   // borne du 1er niveau (cf. CCPOGNNBatchedResult)
    int64_t first_level_n = 0;
    bool   has_incumbent = greedy.feasible;
    double best_val      = greedy.feasible ? greedy.value : -1.0;
    CCPOSpaceSLD* best = nullptr;

    {
        CCPOSpaceSLD* root = new CCPOSpaceSLD(&shared);
        SpaceStatus rst = root->status();
        if (rst == SS_BRANCH) {
            open.push_back({static_cast<double>(root->obj.max()), 0, root});
        } else if (rst == SS_SOLVED) {
            best_val = root->obj.val(); has_incumbent = true;
            best = root; ++incumbent_impr;
        } else {
            delete root;
        }
    }

    // Traite un enfant déjà borné (CP + borne éventuellement postée) : même
    // logique de fin que solve_gecode_bab_ccpo.
    auto admit_child = [&](CCPOSpaceSLD* child, int depth, SpaceStatus cst) {
        const double cub = child->obj.max();
        if (has_incumbent && cub <= best_val + 1e-6) { ++fails; delete child; return; }
        if (cst == SS_SOLVED) {
            const double val = child->obj.val();
            if (!has_incumbent || val > best_val + 1e-9) {
                best_val = val; has_incumbent = true;
                delete best; best = child; ++incumbent_impr;
            } else {
                delete child;
            }
            return;
        }
        open.push_back({cub, depth, child});
    };

    auto flush_bucket = [&](int d) {
        std::vector<Pending>& bucket = pending[d];
        if (bucket.empty()) return;
        const int B_sz = static_cast<int>(bucket.size());
        ++n_flushes; fill_sum += B_sz;

        std::vector<const CCPOQuery*> batch;
        batch.reserve(B_sz);
        for (auto& p : bucket) batch.push_back(&p.query);

        std::vector<std::vector<std::vector<double>>> lambdas;
        const auto tb0 = Clock::now();
        backend.query_batch(ctx, batch, lambdas);
        out.t_backend_s += std::chrono::duration<double>(Clock::now() - tb0).count();

        for (int b = 0; b < B_sz; ++b) {
            Pending& p = bucket[b];
            CCPOSpaceSLD* child = p.space;
            const CCPOQuery& qy = p.query;
            --pending_live;

            // Déjà dominé par un incumbent trouvé entre-temps : inutile de payer zeta_solve.
            if (has_incumbent && child->obj.max() <= best_val + 1e-6) {
                ++fails; delete child; continue;
            }

            std::vector<int> order(qy.res.n_free);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(),
                      [&](int a, int bb) { return qy.res.cost_free[a] > qy.res.cost_free[bb]; });

            // PIÈGE documenté (cf. test_zeta_omp_speedup.py) : torch/PyG peuvent
            // réinitialiser le thread count OpenMP à 1 -- réaffirmé juste avant
            // CHAQUE évaluation, pas une seule fois au démarrage.
            omp_set_num_threads(omp_threads);
            // Héritage de l'infaisabilité (cf. CCPOZetaConfig::inherit_infeas) : un scénario j dont z_j est déjà fixé à 0
            // dans l'espace Gecode (preuve d'infaisabilité d'un ancêtre -- zeta_solve exact, indépendante de λ, valable
            // pour tout descendant -- ou propagation CP du risque) est traité comme infaisable : zeta_solve n'est pas rappelé.
            const bool inherit = ccpo_zeta_config().inherit_infeas != 0;
            std::vector<bool> infeas_node = shared.infeasible_scenario;
            if (inherit)
                for (int j = 0; j < N; ++j)
                    if (child->z[j].assigned() && child->z[j].val() == 0) infeas_node[j] = true;
            const auto te0 = Clock::now();
            LagrangianEval ev;
            if (ccpo_zeta_config().lazy) {
                // évaluation pilotée par l'élagage. Le nœud est élagué ssi ub_int = ⌊cost_fixed + L + 1e-6⌋ <
                // tau_int, tau_int = max(obj.min, best_val + 1) (cf. ci-dessous et admit_child : obj.max ≤ best_val + 1e-6),
                // c.-à-d. ssi L < tau = tau_int − cost_fixed − 1e-6.
                double tau_int = static_cast<double>(child->obj.min());
                if (has_incumbent) tau_int = std::max(tau_int, std::floor(best_val + 1e-6) + 1.0);
                const double tau = tau_int - qy.cost_fixed - 1e-6;
                ev = evaluate_ccpo_lagrangian_lazy_omp(
                    qy.res.n_free, N, qy.res.k_res, qy.res.Q_res, L, U, qy.res.cost_free,
                    qy.res.Sigma_free, qy.b_res, min_feas, infeas_node,
                    lambdas[b], order, qy.res.R_ext, tau);
                if (ccpo_zeta_config().lazy_check) {
                    LagrangianEval ex = evaluate_ccpo_lagrangian_omp(
                        qy.res.n_free, N, qy.res.k_res, qy.res.Q_res, L, U, qy.res.cost_free,
                        qy.res.Sigma_free, qy.b_res, min_feas, infeas_node,
                        lambdas[b], order, qy.res.R_ext);
                    const bool prune_ex = !ex.feasible || ex.L < tau, prune_lz = !ev.feasible || ev.L < tau;
                    const bool bad = (prune_ex != prune_lz) || (ex.feasible && ev.feasible && ev.L < ex.L - 1e-7 * std::max(1.0, std::abs(ex.L)));
                    ccpo_zeta_stats().lazy_check_n++;
                    if (bad) {
                        ccpo_zeta_stats().lazy_check_bad++;
                        std::cerr << "[lazy_check] ÉCART tau=" << tau << " exact(feas=" << ex.feasible << ",L=" << ex.L
                                  << ") lazy(feas=" << ev.feasible << ",L=" << ev.L << ")\n";
                    }
                }
            } else {
                ev = evaluate_ccpo_lagrangian_omp(
                    qy.res.n_free, N, qy.res.k_res, qy.res.Q_res, L, U, qy.res.cost_free,
                    qy.res.Sigma_free, qy.b_res, min_feas, infeas_node,
                    lambdas[b], order, qy.res.R_ext);
            }
            const double te = std::chrono::duration<double>(Clock::now() - te0).count();
            out.t_eval_s += te;
            shared.t_bound_total_s += te;

            SpaceStatus cst;
            if (!ev.feasible) {
                shared.fails_from_bound++;
                ++fails; delete child; continue;
            }
            // Preuves d'infaisabilité de ce nœud -> z_j = 0 dans l'espace (hérité par clone() par tous les descendants).
            if (inherit)
                for (int j = 0; j < N && j < static_cast<int>(ev.infeasible_found.size()); ++j)
                    if (ev.infeasible_found[j] && !(child->z[j].assigned() && child->z[j].val() == 0)) {
                        rel(*child, child->z[j] == 0);
                        ++out.infeas_inherited;
                    }
            const double ub = qy.cost_fixed + ev.L;
            if (p.depth == 1) { first_level_ub = std::max(first_level_ub, ub); ++first_level_n; }
            const int ub_int = static_cast<int>(std::floor(ub + 1e-6));
            if (ub_int < child->obj.max()) {
                shared.ub_tightened++;
                if (ub_int < child->obj.min()) shared.fails_from_bound++;
                rel(*child, child->obj <= ub_int);
            }
            cst = child->status();  // SECOND status() : propage la borne fraîchement postée
            if (cst == SS_FAILED) { ++fails; delete child; continue; }
            admit_child(child, p.depth, cst);
        }
        bucket.clear();
    };

    while ((!open.empty() || pending_live > 0) && time_ok()) {
        if (open.empty()) {
            for (int d = 0; d <= n; ++d) flush_bucket(d);
            continue;
        }

        auto [ub, depth, space] = open.back(); open.pop_back();

        if (has_incumbent && ub <= best_val + 1e-6) { ++fails; delete space; continue; }

        SpaceStatus st = space->status();
        if (st == SS_FAILED) { ++fails; delete space; continue; }

        if (st == SS_SOLVED) {
            ++nodes;
            double val = space->obj.val();
            if (!has_incumbent || val > best_val + 1e-9) {
                best_val = val; has_incumbent = true;
                delete best; best = space; space = nullptr;
                ++incumbent_impr;
            } else {
                delete space;
            }
            continue;
        }

        ++nodes;
        if (depth > max_depth) max_depth = depth;

        const Gecode::Choice* c = space->choice();

        for (int a = 0; a < 2; ++a) {
            CCPOSpaceSLD* child = static_cast<CCPOSpaceSLD*>(space->clone());
            child->commit(*c, a);
            SpaceStatus cst = child->status();  // propagation CP pure d'abord
            if (cst == SS_FAILED) { ++fails; delete child; continue; }

            Pending pd;
            pd.space = child; pd.depth = depth + 1;
            if (!child->build_residual(pd.query.res)) {
                // Plus aucune variable libre sur q : pas de borne lagrangienne
                // (objectif exact) -- comme more()==false dans le driver REF/SLD.
                ++out.nodes_bypassed_leaf;
                admit_child(child, depth + 1, cst);
                continue;
            }
            pd.query.b_res = child->b_res;
            pd.query.depth = depth + 1;
            pd.query.frac_fixed_active = static_cast<double>(child->fixed_count_active_running) / n;
            pd.query.cost_fixed = child->fp_running;

            const int nf = pd.query.res.n_free;
            pending[nf].push_back(std::move(pd));
            ++pending_live;
            out.max_pending_live = std::max<int64_t>(out.max_pending_live, pending_live);
            if (static_cast<int>(pending[nf].size()) >= K) flush_bucket(nf);
        }

        delete c;
        delete space;
    }

    const bool timed_out = !open.empty() || pending_live > 0;
    while (!open.empty()) { auto [ub, d, sp] = open.back(); open.pop_back(); delete sp; }
    for (auto& bucket : pending) {
        for (auto& p : bucket) delete p.space;
        bucket.clear();
    }

    out.greedy_value = greedy.feasible ? greedy.value : std::numeric_limits<double>::quiet_NaN();
    out.solve_time_s = std::chrono::duration<double>(Clock::now() - t0).count();
    out.nodes = nodes; out.fails = fails; out.depth = max_depth;
    out.incumbent_improvements = incumbent_impr;
    out.fails_from_bound = shared.fails_from_bound;
    out.fails_from_cp = fails - shared.fails_from_bound;
    out.more_calls = shared.more_calls;
    out.ub_tightened = shared.ub_tightened;
    out.t_bound_total_s = shared.t_bound_total_s;
    out.n_bucket_flushes = n_flushes;
    out.mean_bucket_fill = n_flushes ? static_cast<double>(fill_sum) / n_flushes : 0.0;
    out.first_level_children = first_level_n;
    out.first_level_bound = first_level_n ? first_level_ub : std::numeric_limits<double>::quiet_NaN();

    if (best) {
        out.objective = best->obj.val();
        out.feasible = true;
        out.optimal = !timed_out;
        out.status = timed_out ? "FEASIBLE" : "OPTIMAL";
        delete best;
    } else if (greedy.feasible) {
        // Incumbent glouton jamais amélioré : il EST la solution finale (cf. gecode_bab_ccpo.cpp).
        out.objective = greedy.value;
        out.feasible = true;
        out.optimal = !timed_out;
        out.status = timed_out ? "FEASIBLE" : "OPTIMAL";
    } else {
        out.feasible = false; out.optimal = false;
        out.status = timed_out ? "TIMEOUT" : "INFEASIBLE";
    }
    return out;
}
