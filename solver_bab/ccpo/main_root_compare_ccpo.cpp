/*
 * main_root_compare_ccpo.cpp — CCPO : GNN un-coup vs sous-gradient À LA RACINE (protocole t*, mirroir de
 * test_paper/mkp/trillium/gnn_vs_subgrad côté MKP), en DEUX phases pour ne réserver un GPU que le temps du forward :
 *
 * Sur le sous-problème résiduel de la racine (après la propagation CP de la racine, aucune variable branchée) :
 *   phase "gnn" (nœud GPU) : λ_GNN = backend.query_batch (graphe C++ + forward TorchScript) -- EXACTEMENT le chemin d'un
 *       nœud du driver batché (gecode_bab_ccpo_gnn_batched.cpp::flush_bucket), batch de taille 1 -- suivi de L(λ_GNN) par
 *       evaluate_ccpo_lagrangian_omp. Un appel de pré-chauffage (JIT TorchScript, allocations CUDA) puis `reps` appels
 *       chronométrés -> médianes. λ_GNN est écrit dans <lambda_out> (texte, %.17g) pour la phase suivante.
 *   phase "subgrad" (nœud CPU) : relit λ_GNN, RÉ-ÉVALUE L(λ_GNN) sur ce nœud (`reps` appels chronométrés -> médiane : le coût
 *       d'évaluation d'un appel GNN mesuré sur le MÊME CPU que le sous-gradient ; la borne doit coïncider avec celle de la
 *       phase gnn), puis sous-gradient : réplique de solve_ccpo_sld (ccpo_sld.hpp : cold start λ=0, pas 1/sqrt(t+1),
 *       meilleur itéré, sondage theta_solve, héritage d'infaisabilité selon CCPO_ZETA_INHERIT), chronométré par
 *       itération et arrêté dès que best_L(t) <= L(λ_GNN). t* et son temps sont MESURÉS (pas extrapolés). Le critère
 *       d'arrêt rel_tol/tol_window de production n'est PAS appliqué (il censurerait t* par la convergence lente) :
 *       l'itération où il se serait déclenché est rapportée (conv_iter, -1 sinon). Censure : T_max ou time_lim.
 * Les bornes sont rapportées à l'échelle de l'objectif (cost_fixed + L).
 *
 * Usage :
 *   bab_ccpo_rootcmp gnn     <épisode> <scripted.pt> <cpu|cuda> <omp_threads> <torch_threads> <lambda_out> [reps=5]
 *     stdout : n,N,n_free,gnn_bound,gnn_time_s,gnn_graph_s,gnn_forward_s,gnn_eval_s
 *   bab_ccpo_rootcmp subgrad <épisode> <lambda_in> <omp_threads> [T_max=1000] [time_lim_s=3600] [reps=5]
 *     stdout : n,N,n_free,gnn_bound,gnn_eval_cpu_s,subgrad_first,subgrad_best,t_star,t_star_time_s,iters_run,
 *              subgrad_time_s,stop_reason,conv_iter,last_impr_iter,last_impr_time_s
 *     (last_impr_iter : dernière itération où la meilleure borne a strictement diminué -- si t* est censuré, dit si le
 *      sous-gradient stagnait depuis longtemps ou progressait encore à la coupure)
 *     stderr : trace "SUBGRAD_TRACE,<t>,<best_bound>,<elapsed_s>"
 * Réglage de zeta_solve (hybride SCIP, héritage) : variables CCPO_ZETA_* comme bab_ccpo.
 */

#include "gecode_bab_ccpo.hpp"
#include "gnn_ccpo_native_batched.hpp"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

#ifdef CCPO_WITH_SCIP
void ccpo_scip_install_hook();   // scip_zeta.cpp
#endif

namespace {

struct Inst {
    int n, N, k, Q, L, U, eps_num, eps_den;
    long long B;
    std::vector<int> c_int;
    std::vector<std::vector<SigmaTriplet>> sigma;
};

// Même lecteur que main_gecode_ccpo.cpp::load_ccpo_episode (format generators/ccpo).
Inst load_episode(const std::string& dir) {
    Inst in;
    std::ifstream f(dir + "/instance.txt");
    if (!f) throw std::runtime_error("impossible d'ouvrir " + dir + "/instance.txt");
    f >> in.n >> in.N >> in.k >> in.Q >> in.L >> in.U >> in.eps_num >> in.eps_den >> in.B;
    in.c_int.resize(in.n);
    for (int i = 0; i < in.n; ++i) f >> in.c_int[i];
    if (!f) throw std::runtime_error(dir + "/instance.txt : format invalide");
    std::ifstream g(dir + "/scenarios.txt");
    if (!g) throw std::runtime_error("impossible d'ouvrir " + dir + "/scenarios.txt");
    const int nt = in.n * (in.n + 1) / 2;
    in.sigma.assign(in.N, {});
    for (int j = 0; j < in.N; ++j) {
        int id; long long psi;
        g >> id;
        if (!g || id != j + 1) throw std::runtime_error(dir + "/scenarios.txt : bloc " + std::to_string(j + 1) + " attendu");
        for (int i = 0; i < in.n; ++i) g >> psi;
        in.sigma[j].reserve(nt);
        for (int t = 0; t < nt; ++t) { int r, c, v; g >> r >> c >> v; in.sigma[j].push_back({r - 1, c - 1, v}); }
    }
    if (!g) throw std::runtime_error(dir + "/scenarios.txt : format invalide");
    return in;
}

// Mêmes échelles que gecode_bab_ccpo_gnn_batched.cpp::compute_scales (= train_ccpo_gnn.py::instance_scales).
void compute_scales(const CCPOSharedData& d, double& c_scale, double& sigma_scale) {
    int cmax = 0;
    for (int v : d.c_int) cmax = std::max(cmax, std::abs(v));
    c_scale = std::max(static_cast<double>(cmax), 1.0);
    double acc = 0.0; int n_el = 0;
    for (int j = 0; j < d.N; ++j) {
        if (d.infeasible_scenario[j]) continue;
        double s = 0.0;
        for (int i = 0; i < d.n; ++i) s += d.sigma_dense[j][i][i];
        acc += s / d.n; ++n_el;
    }
    sigma_scale = (n_el == 0) ? 1.0 : std::max(acc / n_el, 1e-6);
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t m = v.size();
    return m % 2 ? v[m / 2] : 0.5 * (v[m / 2 - 1] + v[m / 2]);
}

using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); }

// Racine : instance, données partagées, sous-problème résiduel après propagation CP, ordre des coûts.
struct Root {
    Inst in;
    int min_feas = 0;
    std::unique_ptr<CCPOSharedData> shared;
    CCPOSpaceSLD* space = nullptr;
    CCPOQuery qy;
    std::vector<int> order;
    ~Root() { delete space; }

    LagrangianEval eval(const std::vector<std::vector<double>>& lambda, const std::vector<bool>& infeas, int omp) const {
        omp_set_num_threads(omp);   // cf. piège torch/libgomp documenté dans le driver batché
        return evaluate_ccpo_lagrangian_omp(qy.res.n_free, in.N, qy.res.k_res, qy.res.Q_res, in.L, in.U, qy.res.cost_free,
                                            qy.res.Sigma_free, qy.b_res, min_feas, infeas, lambda, order, qy.res.R_ext);
    }
};

// Retourne 0 si la racine est exploitable, sinon un code d'erreur (message sur stderr).
int build_root(const std::string& dir, Root& r) {
    r.in = load_episode(dir);
    const long long num = static_cast<long long>(r.in.eps_den - r.in.eps_num) * r.in.N;
    r.min_feas = static_cast<int>((num + r.in.eps_den - 1) / r.in.eps_den);   // ⌈(1-ε)N⌉, cf. saa_threshold
    r.shared = std::make_unique<CCPOSharedData>(r.in.n, r.in.N, r.in.k, r.in.Q, r.in.L, r.in.U, r.in.B, r.min_feas,
                                                r.in.c_int, r.in.sigma);
    r.shared->bound_mode = BoundMode::GNN_BATCHED;
    r.space = new CCPOSpaceSLD(r.shared.get());
    if (r.space->status() != SS_BRANCH) { std::cerr << "racine non branchante (résolue ou infaisable par CP)\n"; return 2; }
    if (!r.space->build_residual(r.qy.res)) { std::cerr << "racine sans variable libre\n"; return 2; }
    r.qy.b_res = r.space->b_res;
    r.qy.depth = 0;
    r.qy.frac_fixed_active = static_cast<double>(r.space->fixed_count_active_running) / r.in.n;
    r.qy.cost_fixed = r.space->fp_running;
    const int nf = r.qy.res.n_free;
    r.order.resize(nf);
    std::iota(r.order.begin(), r.order.end(), 0);
    std::sort(r.order.begin(), r.order.end(), [&](int a, int b) { return r.qy.res.cost_free[a] > r.qy.res.cost_free[b]; });
    return 0;
}

int phase_gnn(int argc, char** argv) {
    if (argc < 8) { std::cerr << "Usage: bab_ccpo_rootcmp gnn <épisode> <scripted.pt> <cpu|cuda> <omp> <torch_threads> <lambda_out> [reps=5]\n"; return 1; }
    const std::string dir = argv[2], scripted = argv[3], device = argv[4], lambda_out = argv[7];
    const int omp = std::atoi(argv[5]), torch_threads = std::atoi(argv[6]);
    const int reps = argc > 8 ? std::max(1, std::atoi(argv[8])) : 5;
    Root r;
    if (int rc = build_root(dir, r)) return rc;
    const int N = r.in.N, nf = r.qy.res.n_free;

    CCPOInstanceCtx ctx;
    ctx.n = r.in.n; ctx.N = N; ctx.L = r.in.L; ctx.U = r.in.U; ctx.k = r.in.k; ctx.Q = r.in.Q; ctx.min_feas = r.min_feas;
    compute_scales(*r.shared, ctx.c_scale, ctx.sigma_scale);
    ctx.epsilon = 1.0 - static_cast<double>(r.min_feas) / N;
    ctx.B = static_cast<double>(r.in.B);
    ctx.infeasible = &r.shared->infeasible_scenario;

    CCPOGNNNativeBatched backend(scripted, torch_threads, device);
    const std::vector<const CCPOQuery*> batch{&r.qy};
    std::vector<std::vector<double>> lambda_gnn;
    double gnn_bound = NAN;
    std::vector<double> t_tot, t_graph, t_fwd, t_eval;
    for (int k = 0; k <= reps; ++k) {
        const double g0 = backend.t_graph_s, f0 = backend.t_forward_s, p0 = backend.t_post_s;
        std::vector<std::vector<std::vector<double>>> lambdas;
        const auto a0 = Clock::now();
        backend.query_batch(ctx, batch, lambdas);
        const auto e0 = Clock::now();
        LagrangianEval ev = r.eval(lambdas[0], r.shared->infeasible_scenario, omp);
        const double te = secs(e0), tt = secs(a0);
        if (!ev.feasible) { std::cerr << "racine prouvée infaisable par L(λ)\n"; return 2; }
        gnn_bound = r.qy.cost_fixed + ev.L;
        lambda_gnn = std::move(lambdas[0]);
        if (k == 0) continue;   // pré-chauffage exclu
        t_tot.push_back(tt); t_eval.push_back(te);
        t_graph.push_back(backend.t_graph_s - g0 + backend.t_post_s - p0); t_fwd.push_back(backend.t_forward_s - f0);
    }

    FILE* f = std::fopen(lambda_out.c_str(), "w");
    if (!f) { std::cerr << "impossible d'écrire " << lambda_out << "\n"; return 1; }
    std::fprintf(f, "%d %d %.17g\n", N, nf, gnn_bound);   // en-tête : N n_free borne (contrôle en phase subgrad)
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < nf; ++i) std::fprintf(f, i ? " %.17g" : "%.17g", lambda_gnn[j][i]);
        std::fputc('\n', f);
    }
    std::fclose(f);
    std::printf("%d,%d,%d,%.17g,%.6g,%.6g,%.6g,%.6g\n", r.in.n, N, nf, gnn_bound,
                median(t_tot), median(t_graph), median(t_fwd), median(t_eval));
    return 0;
}

int phase_subgrad(int argc, char** argv) {
    if (argc < 5) { std::cerr << "Usage: bab_ccpo_rootcmp subgrad <épisode> <lambda_in> <omp> [T_max=1000] [time_lim_s=3600] [reps=5]\n"; return 1; }
    const std::string dir = argv[2], lambda_in = argv[3];
    const int omp = std::atoi(argv[4]);
    const int T_max = argc > 5 ? std::atoi(argv[5]) : 1000;
    const double time_lim = argc > 6 ? std::atof(argv[6]) : 3600.0;
    const int reps = argc > 7 ? std::max(1, std::atoi(argv[7])) : 5;
    Root r;
    if (int rc = build_root(dir, r)) return rc;
    const int N = r.in.N, nf = r.qy.res.n_free;

    // λ_GNN de la phase gnn + contrôle de cohérence (même racine, même borne une fois ré-évaluée ici)
    std::ifstream f(lambda_in);
    int fN = -1, fnf = -1; double gnn_bound_file = NAN;
    f >> fN >> fnf >> gnn_bound_file;
    if (!f || fN != N || fnf != nf) { std::cerr << lambda_in << " : en-tête incohérent (N, n_free) attendu " << N << " " << nf << "\n"; return 1; }
    std::vector<std::vector<double>> lambda_gnn(N, std::vector<double>(nf));
    for (auto& row : lambda_gnn) for (double& v : row) f >> v;
    if (!f) { std::cerr << lambda_in << " : λ tronqué\n"; return 1; }

    double gnn_bound = NAN;
    std::vector<double> t_eval;
    for (int k = 0; k <= reps; ++k) {
        const auto e0 = Clock::now();
        LagrangianEval ev = r.eval(lambda_gnn, r.shared->infeasible_scenario, omp);
        const double te = secs(e0);
        if (!ev.feasible) { std::cerr << "racine prouvée infaisable par L(λ)\n"; return 2; }
        gnn_bound = r.qy.cost_fixed + ev.L;
        if (k > 0) t_eval.push_back(te);   // le 1er appel (caches froids) est exclu, comme le pré-chauffage de la phase gnn
    }
    if (std::abs(gnn_bound - gnn_bound_file) > 1e-6 * std::max(1.0, std::abs(gnn_bound_file)))
        std::cerr << "# ATTENTION : L(λ_GNN) ré-évaluée = " << gnn_bound << " != phase gnn " << gnn_bound_file << "\n";

    std::string stop = "t_max";
    int t_star = -1, iters = 0, conv_iter = -1, last_impr = 0;
    double t_star_time = NAN, sub_first = NAN, best_L = INFINITY, last_impr_time = NAN;
    const auto s0 = Clock::now();
    {
        ThetaResult probe = theta_solve(nf, r.qy.res.Q_res, r.qy.res.k_res, r.in.L, r.in.U, r.qy.res.cost_free);
        if (!probe.feasible) { std::cerr << "racine structurellement infaisable\n"; return 2; }
    }
    std::vector<std::vector<double>> lambda(N, std::vector<double>(nf, 0.0));
    std::vector<bool> infeas_local = r.shared->infeasible_scenario;
    const bool inherit = ccpo_zeta_config().inherit_infeas != 0;
    std::vector<double> hist;
    for (int it = 0; it < T_max; ++it) {
        if (time_lim > 0 && secs(s0) >= time_lim) { stop = "time_limit"; break; }
        LagrangianEval ev = r.eval(lambda, infeas_local, omp);
        iters = it + 1;
        if (!ev.feasible) { stop = "infeasible"; break; }
        if (inherit)
            for (int j = 0; j < N && j < static_cast<int>(ev.infeasible_found.size()); ++j)
                if (ev.infeasible_found[j]) infeas_local[j] = true;
        const double el = secs(s0);
        if (ev.L < best_L) { best_L = ev.L; last_impr = iters; last_impr_time = el; }   // dernière amélioration STRICTE de la meilleure borne
        const double best_bound = r.qy.cost_fixed + best_L;
        if (it == 0) sub_first = r.qy.cost_fixed + ev.L;
        std::fprintf(stderr, "SUBGRAD_TRACE,%d,%.6f,%.6f\n", iters, best_bound, el);
        if (best_bound <= gnn_bound + 1e-9) { t_star = iters; t_star_time = el; stop = "reached"; break; }
        // Critère rel_tol de production (fenêtre glissante sur best_L) : NOTÉ mais pas appliqué (conv_iter).
        hist.push_back(best_L);
        if (conv_iter < 0 && static_cast<int>(hist.size()) > r.shared->tol_window) {
            const double old = hist[hist.size() - 1 - r.shared->tol_window];
            if (std::abs(old - best_L) / std::max(1.0, std::abs(old)) < r.shared->rel_tol) conv_iter = iters;
        }
        const double step = 1.0 / std::sqrt(static_cast<double>(it + 1));
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < nf; ++i) lambda[j][i] -= step * ev.subgrad[j][i];
    }
    const double sub_time = secs(s0);
    std::printf("%d,%d,%d,%.17g,%.6g,%.6f,%.6f,%d,%.6g,%d,%.6g,%s,%d,%d,%.6g\n",
                r.in.n, N, nf, gnn_bound, median(t_eval), sub_first, r.qy.cost_fixed + best_L, t_star, t_star_time,
                iters, sub_time, stop.c_str(), conv_iter, last_impr, last_impr_time);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef CCPO_WITH_SCIP
    ccpo_scip_install_hook();
#endif
    const std::string phase = argc > 1 ? argv[1] : "";
    if (phase != "gnn" && phase != "subgrad") {
        std::cerr << "Usage: bab_ccpo_rootcmp gnn|subgrad ... (cf. en-tête de main_root_compare_ccpo.cpp)\n";
        return 1;
    }
    std::cerr << "# phase=" << phase << " zeta_cfg warm=" << ccpo_zeta_config().warm_mode << " branch=" << ccpo_zeta_config().branching
              << " inherit=" << ccpo_zeta_config().inherit_infeas << " scip=" << ccpo_zeta_config().scip_mode
              << " scip_nodes=" << ccpo_zeta_config().scip_nodes << " scip_nf0=" << ccpo_zeta_config().scip_nf0
              << " scip_hook=" << (ccpo_scip_hook() ? 1 : 0) << "\n";
    try {
        return phase == "gnn" ? phase_gnn(argc, argv) : phase_subgrad(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Erreur : " << e.what() << "\n";
        return 1;
    }
}
