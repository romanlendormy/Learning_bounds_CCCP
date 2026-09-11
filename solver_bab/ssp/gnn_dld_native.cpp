/*
 * gnn_dld_native.cpp — Implémentation (cf. gnn_dld_native.hpp).
 *
 * Mirror structurel de solver_bab/mkp/gnn_sld_native.cpp : TorchScript
 * (torch::jit::load + .forward()), pas de réimplémentation de l'architecture
 * GNN en C++. build_graph_cpp()/lagrangian_dld_cpp() réimplémentent la
 * construction de graphe et le calcul de borne à la main (port fidèle de
 * gnn_ssp/train_DLD_ssp.py::build_graph / lagrangian_dld_ssp) — ce ne
 * sont pas le modèle.
 *
 * ζ/θ via SSPDLDProblemBnB::solve_theta/solve_zeta appelés en boucle
 * parallélisée OpenMP (cf. lagrangian_dld_cpp) — pas de solve_zeta_batch
 * dédié côté SSP (contrairement à MKP), mais chaque (i,j) est indépendant,
 * donc un simple #pragma omp parallel for suffit. Reste dominant en pratique
 * (~99.9% du temps d'un more() sur l'instance de test E4T12N20, cf.
 * discussion 2026-08-19) malgré la parallélisation — le glouton lui-même
 * (pas sa construction de graphe) est le vrai goulot restant.
 */

#include "gnn_dld_native.hpp"
#include "../../solver_generic/ssp/ssp_dld_problem_bnb.hpp"
#include "../../solver_generic/ssp/ssp_problem.hpp"   // SSP_SIGMA_SIZE, SSP_SIGMA_W

#include <torch/script.h>
#include <torch/cuda.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <fstream>
#include <mutex>
#include <numeric>
#include <spawn.h>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>

extern char** environ;

namespace {

// ── build_graph (port de train_DLD_ssp.py::build_graph) ───────────────────
struct BuiltGraph {
    torch::Tensor X, edge_index, edge_weight, context;
    std::vector<int> free_et;   // (G,) idx global e*T+t
    std::vector<int> free_z;    // (N_z,) idx global scénario
    int G = 0, N_z = 0, n_fixed_z1 = 0;
    double c_scale = 1.0;
};

// Arêtes item/scénario (clique intra-groupe 1/G, étoile inter-groupes 1.0) —
// même algorithme que build_edges_vectorized (gnn_sld_native.cpp, MKP) : ne
// dépend que du NOMBRE de groupes et de leur TAILLE, jamais de leur sens
// (n_groups ici = N_z*T, "item" = G, cf. docstring module DLD SSP côté
// Python : "on RÉUTILISE tel quel le pooling/les arêtes de SLD").
// build_edges — depuis le 2026-08-20, ne construit plus QUE les arêtes
// étoile inter-groupe. Les arêtes clique intra-groupe (poids uniforme 1/G)
// sont retirées : le modèle (GatedGraphConvSspClique, cf.
// gnn_ssp/gnn_ssp_scriptable_batched.py) calcule maintenant leur agrégation
// par formule fermée -- (group_sum - m)/G -- identité algébrique exacte vu
// le poids uniforme, validée numériquement (gnn_ssp/validate_clique_mean_equivalence.py,
// écart 0.000e+00 sur tous les cas testés), pas une approximation. Gain
// double : O(G) au lieu de O(G²) pour cette part du forward, ET plus
// besoin de matérialiser jusqu'à G²·n_groups arêtes (source directe des OOM
// GPU observés à grand N/T, cf. discussion 2026-08-20).
std::pair<torch::Tensor, torch::Tensor> build_edges(int G, int n_groups, torch::Device device) {
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    torch::Tensor ss_src = torch::empty({0}, opts_i64), ss_dst = torch::empty({0}, opts_i64),
                  ss_w   = torch::empty({0}, opts_f32);
    if (n_groups >= 2) {
        auto k_all = torch::arange(G, opts_i64);
        auto ii = torch::arange(1, n_groups, opts_i64);
        auto fwd_src = (0 * G + k_all).unsqueeze(0).expand({ii.size(0), G}).flatten();
        auto fwd_dst = (ii.unsqueeze(1) * G + k_all.unsqueeze(0)).flatten();
        ss_src = torch::cat({fwd_src, fwd_dst});
        ss_dst = torch::cat({fwd_dst, fwd_src});
        ss_w   = torch::full({ss_src.size(0)}, 1.0f, opts_f32);
    }

    auto edge_index  = torch::stack({ss_src, ss_dst});
    auto edge_weight = ss_w;
    return {edge_index, edge_weight};
}

BuiltGraph build_graph_cpp(int E, int T, int N, const Vec1d& salary, const Vec2i& demand,
                           const std::vector<int8_t>& x_mask, const std::vector<int8_t>& z_mask,
                           double curr_cost, int depth, double epsilon, torch::Device device)
{
    std::vector<int> free_et;
    for (int idx = 0; idx < E * T; ++idx)
        if (x_mask[idx] == SSP_SIGMA_SIZE) free_et.push_back(idx);
    std::vector<int> free_z;
    int n_fixed_z1 = 0;
    for (int i = 0; i < N; ++i) {
        if (z_mask[i] == 2) free_z.push_back(i);
        else if (z_mask[i] == 1) ++n_fixed_z1;
    }
    int G = static_cast<int>(free_et.size());
    int N_z = static_cast<int>(free_z.size());
    int n_groups = N_z * T;

    std::vector<int> e_idx(G), t_idx(G);
    for (int k = 0; k < G; ++k) { e_idx[k] = free_et[k] / T; t_idx[k] = free_et[k] % T; }

    double s_max = 1e-9; for (double s : salary) s_max = std::max(s_max, s);
    double d_max = 1.0;
    for (int i = 0; i < N; ++i) for (int t = 0; t < T; ++t) d_max = std::max(d_max, (double)demand[i][t]);

    std::vector<double> y_committed(T, 0.0);
    for (int e = 0; e < E; ++e)
        for (int t = 0; t < T; ++t)
            if (x_mask[e * T + t] == SSP_SIGMA_W) y_committed[t] += 1.0;

    double c_scale = curr_cost;
    for (int k = 0; k < G; ++k) c_scale += salary[t_idx[k]];
    c_scale = std::max(c_scale, 1.0);

    std::vector<float> salary_g(G), e_norm_g(G), t_norm_g(G), y_committed_g(G);
    for (int k = 0; k < G; ++k) {
        salary_g[k]      = static_cast<float>(salary[t_idx[k]] / s_max);
        e_norm_g[k]       = static_cast<float>(e_idx[k] / static_cast<double>(std::max(E - 1, 1)));
        t_norm_g[k]       = static_cast<float>(t_idx[k] / static_cast<double>(std::max(T - 1, 1)));
        y_committed_g[k]  = static_cast<float>(y_committed[t_idx[k]] / std::max(E, 1));
    }
    std::vector<float> j_norm(T);
    for (int j = 0; j < T; ++j) j_norm[j] = static_cast<float>(j / static_cast<double>(std::max(T - 1, 1)));

    const int node_dim = 8;
    std::vector<float> X(static_cast<size_t>(N_z) * T * G * node_dim);
    std::vector<double> scen_shortfall_mean(N_z, 0.0);

    for (int i_local = 0; i_local < N_z; ++i_local) {
        int i_glob = free_z[i_local];
        double shortfall_sum = 0.0;
        for (int k = 0; k < G; ++k) {
            int t = t_idx[k];
            double d_val = static_cast<double>(demand[i_glob][t]);
            double d_ig = d_val / d_max;
            double shortfall = std::max(0.0, d_val - y_committed[t]) / std::max(E, 1);
            shortfall_sum += shortfall;
            for (int j = 0; j < T; ++j) {
                size_t row = (static_cast<size_t>(i_local) * T + j) * G + k;
                float* out = &X[row * node_dim];
                out[0] = salary_g[k];
                out[1] = static_cast<float>(d_ig);
                out[2] = y_committed_g[k];
                out[3] = static_cast<float>(shortfall);
                out[4] = e_norm_g[k];
                out[5] = t_norm_g[k];
                out[6] = j_norm[j];
                out[7] = (t == j) ? 1.0f : 0.0f;
            }
        }
        scen_shortfall_mean[i_local] = shortfall_sum / std::max(G, 1);
    }

    BuiltGraph g;
    g.X = torch::from_blob(X.data(), {static_cast<int64_t>(N_z) * T * G, node_dim}, torch::kFloat32).clone();

    auto edges = build_edges(G, n_groups, device);
    g.edge_index = edges.first;
    g.edge_weight = edges.second;

    double depth_norm = static_cast<double>(depth) / std::max(E * T, 1);
    double frac_fixed = 1.0 - static_cast<double>(G) / std::max(E * T, 1);
    double tightest = 1.0;
    if (N_z > 0) {
        tightest = 1e300;
        for (int i_local = 0; i_local < N_z; ++i_local)
            tightest = std::min(tightest, 1.0 - scen_shortfall_mean[i_local]);
    }

    std::vector<float> ctx = {
        static_cast<float>(depth_norm), static_cast<float>(frac_fixed), static_cast<float>(tightest),
        static_cast<float>(curr_cost / c_scale), static_cast<float>(epsilon),
        static_cast<float>(static_cast<double>(n_fixed_z1) / N),
        static_cast<float>(std::log(std::max(G, 1))), static_cast<float>(std::log(std::max(n_groups, 1))),
        static_cast<float>(static_cast<double>(n_groups) / std::max(G, 1))
    };
    g.context = torch::from_blob(ctx.data(), {1, 9}, torch::kFloat32).clone();

    g.free_et = free_et; g.free_z = free_z;
    g.G = G; g.N_z = N_z; g.n_fixed_z1 = n_fixed_z1; g.c_scale = c_scale;
    return g;
}

// ── lagrangian_dld (port de train_DLD_ssp.py::lagrangian_dld_ssp) ─────
// x_mask complet (E*T, convention disque 0..3/4) requis ici pour reconstruire
// le VRAI masque (E,T) attendu par SSPDLDProblemBnB (symboles imposés aux
// positions fixées, pas seulement "libre/fixé") — BuiltGraph ne garde que
// free_et (positions libres), insuffisant pour ça.
double lagrangian_dld_cpp(const torch::Tensor& lam_raw, const BuiltGraph& g,
                            const SSPInstance& inst, const std::vector<int8_t>& x_mask, int T,
                            double& t_zeta_theta_s)
{
    int N_z = g.N_z, G = g.G, N = inst.N, E = inst.E;
    double p_i = 1.0 / N;
    double p_ij = p_i / T;

    // λ mis à l'échelle par c_scale/(N*T) — cf. FIX 2026-08-17, train_DLD_ssp.py:728.
    auto lam = (lam_raw * g.c_scale / (static_cast<double>(N) * T))
                   .view({N_z, T, G, SSP_SIGMA_SIZE}).to(torch::kFloat64).contiguous();
    auto lam_acc = lam.accessor<double, 4>();

    // Λ_sum_i[i,k,σ] = Σ_j λ[i,j,k,σ]  — PAR SCÉNARIO (axe période seulement),
    // formule corrigée (cf. lagrangian_dld_ssp, FIX centrage par scénario).
    std::vector<double> Lambda_sum_i(static_cast<size_t>(N_z) * G * SSP_SIGMA_SIZE, 0.0);
    for (int i = 0; i < N_z; ++i)
        for (int j = 0; j < T; ++j)
            for (int k = 0; k < G; ++k)
                for (int s = 0; s < SSP_SIGMA_SIZE; ++s)
                    Lambda_sum_i[(static_cast<size_t>(i) * G + k) * SSP_SIGMA_SIZE + s] += lam_acc[i][j][k][s];

    // Sous-problème SSPDLDProblemBnB restreint aux scénarios libres, même
    // convention que get_ssp_dld_problem (train_DLD_ssp.py) : demande
    // restreinte à free_z (réindexée localement), p uniforme 1/N_z (PAS
    // 1/N — mirror exact du code Python), masque (E,T) reconstruit depuis
    // x_mask complet (-1 = libre, sinon symbole imposé).
    Vec2i demand_free(N_z, Vec1i(T));
    for (int i = 0; i < N_z; ++i)
        for (int t = 0; t < T; ++t)
            demand_free[i][t] = inst.demand[g.free_z[i]][t];
    Vec1d p_local(N_z, 1.0 / std::max(N_z, 1));

    Vec2i mask(E, Vec1i(T, -1));
    for (int e = 0; e < E; ++e)
        for (int t = 0; t < T; ++t) {
            int8_t v = x_mask[e * T + t];
            mask[e][t] = (v == SSP_SIGMA_SIZE) ? -1 : static_cast<int>(v);
        }

    SSPDLDProblemBnB prob(inst.transitions, inst.accepting, inst.initial,
                          inst.salary, demand_free, p_local, inst.epsilon, E, mask);

    int n_vars = prob.n_vars();   // E*T*4
    Vec1d c_base = prob.c();      // coût de base (-salary[t] si σ=W, 0 sinon), taille n_vars

    // flat_free_idx[k*4+s] = free_et[k]*4+s — indices globaux (e,t,σ) libres
    std::vector<int> flat_free_idx(static_cast<size_t>(G) * SSP_SIGMA_SIZE);
    for (int k = 0; k < G; ++k)
        for (int s = 0; s < SSP_SIGMA_SIZE; ++s)
            flat_free_idx[k * SSP_SIGMA_SIZE + s] = g.free_et[k] * SSP_SIGMA_SIZE + s;

    int n_groups = N_z * T;
    std::vector<Vec1d> cost_full(n_groups, c_base);
    for (int i = 0; i < N_z; ++i) {
        for (int j = 0; j < T; ++j) {
            int grp = i * T + j;
            for (int k = 0; k < G; ++k) {
                for (int s = 0; s < SSP_SIGMA_SIZE; ++s) {
                    double lam_ijks = lam_acc[i][j][k][s];
                    double Lsum = Lambda_sum_i[(static_cast<size_t>(i) * G + k) * SSP_SIGMA_SIZE + s];
                    double correction = lam_ijks / p_ij - Lsum / p_i;
                    cost_full[grp][flat_free_idx[k * SSP_SIGMA_SIZE + s]] += correction;
                }
            }
        }
    }

    // Boucle ζ/θ parallélisée (OpenMP) — mirror de solve_zeta_batch côté MKP
    // (gnn/bnb_solver.cpp). Chaque grp=(i,j) est indépendant (lit cost_full[grp],
    // écrit dans des cases disjointes de theta/zeta/x_theta/x_zeta, tous
    // pré-dimensionnés avant la boucle -- pas de réallocation concurrente).
    // solve_theta/solve_zeta sont const sur prob (aucun état mutable observé
    // dans SSPDLDProblem/SSPDLDProblemBnB, cf. ssp_dld_problem.hpp) -- sûr en
    // lecture concurrente. Motivé par le profilage de l'étape 3 (job
    // 19235895) : t_bound_total_s = 99.9% du temps total, cette boucle
    // séquentielle (jusqu'ici délibérément non parallélisée, cf. discussion
    // 2026-08-18 "pas d'OpenMP pour l'instant, à ajouter si le profilage le
    // justifie") était le goulot mesuré.
    // schedule(static) : testé contre dynamic/guided (2026-08-19), aucune
    // différence mesurable -- le plafonnement du scaling (×2.19 avec 4
    // threads, pas ×4) ne vient pas d'un déséquilibre de charge entre (i,j).
    std::vector<double> theta(N_z * T), zeta(N_z * T);
    std::vector<Vec1d> x_theta(n_groups), x_zeta(n_groups);
    auto t_zt0 = std::chrono::steady_clock::now();
    #pragma omp parallel for schedule(static)
    for (int grp = 0; grp < n_groups; ++grp) {
        int i = grp / T, j = grp % T;
        auto [v_t, xt] = prob.solve_theta(j, cost_full[grp]);
        auto [v_z, xz] = prob.solve_zeta(i, j, cost_full[grp]);
        theta[grp] = v_t; zeta[grp] = v_z;
        x_theta[grp] = std::move(xt); x_zeta[grp] = std::move(xz);
    }
    t_zeta_theta_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_zt0).count();

    Vec1d Delta_i(N_z, 0.0);
    for (int i = 0; i < N_z; ++i) {
        double sum_delta = 0.0;
        for (int j = 0; j < T; ++j) {
            int grp = i * T + j;
            sum_delta += std::max(0.0, theta[grp] - zeta[grp]);
        }
        Delta_i[i] = sum_delta / T;
    }

    double required_prob = std::max(0.0, (1.0 - inst.epsilon) - g.n_fixed_z1 * p_i);
    std::vector<int> order(N_z);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return Delta_i[a] < Delta_i[b]; });

    Vec1d y_star(N_z, 0.0);
    double cumprob = 0.0;
    for (int idx : order) {
        if (required_prob <= 0.0) break;
        y_star[idx] = 1.0;
        cumprob += p_i;
        if (cumprob >= required_prob - 1e-9) break;
    }

    // L1 linéaire en λ à x*,y* fixés (théorème de l'enveloppe) — cf.
    // lagrangian_dld_ssp(D) train_DLD_ssp.py:812-823.
    Vec1d theta_lam_ij(n_groups, 0.0), delta_lam_ij(n_groups, 0.0);
    for (int grp = 0; grp < n_groups; ++grp) {
        double th = 0.0, de = 0.0;
        for (int v = 0; v < n_vars; ++v) {
            th += cost_full[grp][v] * x_theta[grp][v];
            de += cost_full[grp][v] * (x_theta[grp][v] - x_zeta[grp][v]);
        }
        theta_lam_ij[grp] = th; delta_lam_ij[grp] = de;
    }
    Vec1d delta_lam_i(N_z, 0.0);
    for (int i = 0; i < N_z; ++i) {
        double sum = 0.0;
        for (int j = 0; j < T; ++j) sum += delta_lam_ij[i * T + j];
        delta_lam_i[i] = sum / T;
    }

    double L1_theta = 0.0;
    for (int grp = 0; grp < n_groups; ++grp) L1_theta += p_ij * theta_lam_ij[grp];
    double val_lam = 0.0;
    for (int i = 0; i < N_z; ++i) val_lam += p_i * y_star[i] * delta_lam_i[i];

    return L1_theta - val_lam;   // = L1(λ)
}

// ── Export TorchScript one-shot (fork python, mis en cache) ─────────────────
// Mirror de ensure_scripted_export (gnn_sld_native.cpp) — cf. ce fichier
// pour la justification de posix_spawnp vs fork().
bool file_exists(const std::string& p) { struct stat st; return ::stat(p.c_str(), &st) == 0; }
double file_mtime(const std::string& p) { struct stat st; if (::stat(p.c_str(), &st) != 0) return -1.0; return static_cast<double>(st.st_mtime); }

std::string ensure_scripted_export(const std::string& checkpoint_path, const std::string& python_exe,
                                    const std::string& repo_root) {
    std::string scripted_path = checkpoint_path + ".scripted_batched.pt";
    if (file_exists(scripted_path) && file_mtime(scripted_path) >= file_mtime(checkpoint_path))
        return scripted_path;

    std::string script = repo_root + "/gnn_ssp/gnn_ssp_scriptable_batched.py";
    std::vector<char*> argv = {
        const_cast<char*>(python_exe.c_str()), const_cast<char*>(script.c_str()),
        const_cast<char*>("--checkpoint"), const_cast<char*>(checkpoint_path.c_str()),
        const_cast<char*>("--out"), const_cast<char*>(scripted_path.c_str()),
        const_cast<char*>("--device"), const_cast<char*>("cpu"),
        nullptr
    };
    pid_t pid = 0;
    int rc = ::posix_spawnp(&pid, python_exe.c_str(), nullptr, nullptr, argv.data(), environ);
    if (rc != 0)
        throw std::runtime_error("GNNDLDNative: posix_spawnp a échoué: " + std::string(std::strerror(rc)));
    int status = 0;
    ::waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
        throw std::runtime_error("GNNDLDNative: gnn_ssp_scriptable_batched.py a échoué (checkpoint invalide ?)");
    return scripted_path;
}

} // namespace

// ── Impl ──────────────────────────────────────────────────────────────────────
struct GNNDLDNative::Impl {
    SSPInstance inst;
    torch::Device device{torch::kCPU};
    torch::jit::script::Module module;
    SSPGNNQueryTiming cum_timing;
};

GNNDLDNative::GNNDLDNative(const std::string& checkpoint_path, const std::string& device,
                                 const SSPInstance& inst)
    : impl_(std::make_unique<Impl>())
{
    impl_->inst = inst;

    if (device == "cuda") {
        if (!torch::cuda::is_available())
            throw std::invalid_argument("GNNDLDNative: device=cuda demandé mais CUDA indisponible");
        impl_->device = torch::Device(torch::kCUDA);
    } else if (device == "cpu") {
        impl_->device = torch::Device(torch::kCPU);
    } else {
        throw std::invalid_argument("GNNDLDNative: device inconnu (\"" + device + "\"), attendu cpu|cuda");
    }

    // Repo root déduit de la position de ce .cpp (solver_bab/ssp/) — deux
    // niveaux au-dessus. Surchargeable via variable d'env si besoin plus tard.
    std::string repo_root = "..";
    if (const char* env = std::getenv("SSP_REPO_ROOT")) repo_root = env;

    std::string scripted_path = ensure_scripted_export(checkpoint_path, "python3", repo_root);
    impl_->module = torch::jit::load(scripted_path, impl_->device);
    impl_->module.eval();
}

GNNDLDNative::~GNNDLDNative() = default;

double GNNDLDNative::query(const std::vector<int8_t>& x_mask, const std::vector<int8_t>& z_mask,
                              double curr_cost, int depth)
{
    using Clock = std::chrono::steady_clock;
    const SSPInstance& inst = impl_->inst;

    auto t0 = Clock::now();
    BuiltGraph g = build_graph_cpp(inst.E, inst.T, inst.N, inst.salary, inst.demand,
                                   x_mask, z_mask, curr_cost, depth, inst.epsilon, impl_->device);
    auto t1 = Clock::now();
    if (g.G == 0 || g.N_z == 0)
        throw std::runtime_error("GNNDLDNative::query: nœud sans paire (e,t) ou scénario libre "
                                  "(G=0 ou N_z=0) — pas de borne GNN à calculer ici");

    torch::NoGradGuard no_grad;
    torch::Tensor X = g.X, edge_index = g.edge_index, edge_weight = g.edge_weight, context = g.context;
    if (impl_->device.is_cuda()) {
        X = X.to(impl_->device); edge_index = edge_index.to(impl_->device);
        edge_weight = edge_weight.to(impl_->device); context = context.to(impl_->device);
    }

    std::vector<torch::jit::IValue> inputs = {
        X, edge_index, edge_weight, context,
        static_cast<int64_t>(g.G), static_cast<int64_t>(inst.T), static_cast<int64_t>(1)
    };
    torch::Tensor lam_raw = impl_->module.forward(inputs).toTensor();
    if (impl_->device.is_cuda()) lam_raw = lam_raw.to(torch::kCPU);
    auto t2 = Clock::now();

    // Trace CSV optionnelle (G,N_z,depth,t_forward_ms,wall_ts_epoch_s) par appel
    // -- diagnostic 2026-08-19 : vérifier si le coût moyen du forward mesuré en
    // vrai B&B (agrégé) est cohérent avec le coût par FORME mesuré en isolation
    // (banc Python), ou s'il y a un surcoût propre à l'intégration C++/plus
    // vraie boucle B&B (allocation répétée, etc.) que le banc isolé ne verrait pas.
    // wall_ts_epoch_s (ajouté 2026-08-20) : timestamp horloge murale (pas le
    // Clock steady_clock utilisé pour t1/t2) pour recoller avec un monitoring
    // externe (charge système, cgroup) échantillonné en temps réel.
    if (const char* trace_path = std::getenv("GNN_TRACE_SHAPE")) {
        static std::mutex trace_mutex;
        std::lock_guard<std::mutex> lock(trace_mutex);
        std::ofstream f(trace_path, std::ios::app);
        double wall_ts_s = std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        f << g.G << "," << g.N_z << "," << depth << ","
          << std::chrono::duration<double, std::milli>(t2 - t1).count() << ","
          << wall_ts_s << "\n";
    }

    double t_zeta_theta_s = 0.0;
    double L1 = lagrangian_dld_cpp(lam_raw, g, inst, x_mask, inst.T, t_zeta_theta_s);
    auto t3 = Clock::now();

    auto& ct = impl_->cum_timing;
    ct.t_graph_build_s     += std::chrono::duration<double>(t1 - t0).count();
    ct.t_forward_s         += std::chrono::duration<double>(t2 - t1).count();
    ct.t_zeta_theta_s      += t_zeta_theta_s;
    ct.t_lagrangian_rest_s += std::chrono::duration<double>(t3 - t2).count() - t_zeta_theta_s;
    ct.n_queries           += 1;

    return -L1;   // cost_lower
}

SSPGNNQueryTiming GNNDLDNative::cumulative_timing() const { return impl_->cum_timing; }
