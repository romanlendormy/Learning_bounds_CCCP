/*
 * gnn_dld_native_batched.cpp — Implémentation (cf. gnn_dld_native_batched.hpp).
 *
 * Différence structurelle avec gnn_dld_native.cpp::build_graph_cpp() : la
 * construction est scindée en deux, exactement pour permettre le batching
 * (même principe que gnn_sd_edit_native_batched.cpp, MKP) :
 *   - build_ssp_features_cpp() : tout ce qui dépend du masque/coût du membre b
 *     (features de nœuds X, contexte 9-dim) -- appelé UNE FOIS PAR MEMBRE (B fois).
 *   - build_edges() : la topologie d'arêtes, qui ne dépend QUE de (G, n_groups=N_z*T)
 *     -- donc IDENTIQUE pour tous les membres d'un même batch (même (G,N_z)
 *     garanti par l'appelant, cf. contrat SSPGNNBoundBackendBatch) -- appelée
 *     UNE SEULE FOIS par appel query_batch(), puis dupliquée B fois avec offset
 *     (bloc-diagonal, cf. gnn_ssp_scriptable_batched.py).
 *
 * lagrangian_dld_cpp (calcul de ζ/θ/L1(λ)) reste appelé B fois séparément
 * (une fois par membre, sur son propre masque) -- pas batchable simplement,
 * et de toute façon PAS le goulot ciblé par cette variante (cf. discussion
 * 2026-08-19 : le forward GNN domine à 95%, ζ/θ pèse <1%).
 *
 * build_edges/lagrangian_dld_cpp DUPLIQUÉES depuis gnn_dld_native.cpp
 * (pas partagées) -- même choix que gnn_sd_edit_native_batched.cpp côté MKP.
 */

#include "gnn_dld_native_batched.hpp"
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
#include <map>
#include <numeric>
#include <spawn.h>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>

extern char** environ;

namespace {

// ── Features seules (sans arêtes) — port de build_graph_cpp
// (gnn_dld_native.cpp) minus la construction edge_index/edge_weight. ──────
struct SSPBuiltFeatures {
    std::vector<float> X_flat;   // (N_z*T*G*node_dim,) row-major
    std::vector<float> ctx;      // (9,)
    std::vector<int> free_et, free_z;
    int G = 0, N_z = 0, n_fixed_z1 = 0;
    double c_scale = 1.0;
};

SSPBuiltFeatures build_ssp_features_cpp(int E, int T, int N, const Vec1d& salary, const Vec2i& demand,
                                        const std::vector<int8_t>& x_mask, const std::vector<int8_t>& z_mask,
                                        double curr_cost, int depth, double epsilon)
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
    SSPBuiltFeatures f;
    f.X_flat.assign(static_cast<size_t>(N_z) * T * G * node_dim, 0.0f);
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
                float* out = &f.X_flat[row * node_dim];
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

    double depth_norm = static_cast<double>(depth) / std::max(E * T, 1);
    double frac_fixed = 1.0 - static_cast<double>(G) / std::max(E * T, 1);
    double tightest = 1.0;
    if (N_z > 0) {
        tightest = 1e300;
        for (int i_local = 0; i_local < N_z; ++i_local)
            tightest = std::min(tightest, 1.0 - scen_shortfall_mean[i_local]);
    }

    f.ctx = {
        static_cast<float>(depth_norm), static_cast<float>(frac_fixed), static_cast<float>(tightest),
        static_cast<float>(curr_cost / c_scale), static_cast<float>(epsilon),
        static_cast<float>(static_cast<double>(n_fixed_z1) / N),
        static_cast<float>(std::log(std::max(G, 1))), static_cast<float>(std::log(std::max(n_groups, 1))),
        static_cast<float>(static_cast<double>(n_groups) / std::max(G, 1))
    };

    f.free_et = free_et; f.free_z = free_z;
    f.G = G; f.N_z = N_z; f.n_fixed_z1 = n_fixed_z1; f.c_scale = c_scale;
    return f;
}

// ── Arêtes (étoile inter-groupes 1.0 SEULEMENT depuis le 2026-08-20) —
// DUPLIQUÉE de gnn_dld_native.cpp (ne dépend que de (G, n_groups), jamais
// du masque). Les arêtes clique intra-groupe sont retirées : le modèle
// (GatedGraphConvSspClique) calcule leur agrégation par formule fermée
// (identité exacte, poids uniforme 1/G) -- cf. commentaire détaillé dans
// gnn_dld_native.cpp::build_edges et gnn_ssp/gnn_ssp_scriptable_batched.py. ──
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

// ── lagrangian_dld (calcul ζ/θ/L1(λ)) — DUPLIQUÉE de gnn_dld_native.cpp,
// adaptée pour lire un lam DÉJÀ tranché pour ce membre (lam_raw : V*4 éléments,
// V = N_z*T*G) plutôt que le tenseur complet du batch. ──────────────────────
double lagrangian_dld_cpp(const torch::Tensor& lam_raw, const SSPBuiltFeatures& f,
                            const SSPInstance& inst, const std::vector<int8_t>& x_mask, int T,
                            double& t_zeta_theta_s)
{
    int N_z = f.N_z, G = f.G, N = inst.N, E = inst.E;
    double p_i = 1.0 / N;
    double p_ij = p_i / T;

    auto lam = (lam_raw * f.c_scale / (static_cast<double>(N) * T))
                   .view({N_z, T, G, SSP_SIGMA_SIZE}).to(torch::kFloat64).contiguous();
    auto lam_acc = lam.accessor<double, 4>();

    std::vector<double> Lambda_sum_i(static_cast<size_t>(N_z) * G * SSP_SIGMA_SIZE, 0.0);
    for (int i = 0; i < N_z; ++i)
        for (int j = 0; j < T; ++j)
            for (int k = 0; k < G; ++k)
                for (int s = 0; s < SSP_SIGMA_SIZE; ++s)
                    Lambda_sum_i[(static_cast<size_t>(i) * G + k) * SSP_SIGMA_SIZE + s] += lam_acc[i][j][k][s];

    Vec2i demand_free(N_z, Vec1i(T));
    for (int i = 0; i < N_z; ++i)
        for (int t = 0; t < T; ++t)
            demand_free[i][t] = inst.demand[f.free_z[i]][t];
    Vec1d p_local(N_z, 1.0 / std::max(N_z, 1));

    Vec2i mask(E, Vec1i(T, -1));
    for (int e = 0; e < E; ++e)
        for (int t = 0; t < T; ++t) {
            int8_t v = x_mask[e * T + t];
            mask[e][t] = (v == SSP_SIGMA_SIZE) ? -1 : static_cast<int>(v);
        }

    SSPDLDProblemBnB prob(inst.transitions, inst.accepting, inst.initial,
                          inst.salary, demand_free, p_local, inst.epsilon, E, mask);

    int n_vars = prob.n_vars();
    Vec1d c_base = prob.c();

    std::vector<int> flat_free_idx(static_cast<size_t>(G) * SSP_SIGMA_SIZE);
    for (int k = 0; k < G; ++k)
        for (int s = 0; s < SSP_SIGMA_SIZE; ++s)
            flat_free_idx[k * SSP_SIGMA_SIZE + s] = f.free_et[k] * SSP_SIGMA_SIZE + s;

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

    double required_prob = std::max(0.0, (1.0 - inst.epsilon) - f.n_fixed_z1 * p_i);
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

    return L1_theta - val_lam;
}

// ── Export TorchScript one-shot — DUPLIQUÉE de gnn_dld_native.cpp. ────────
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
        throw std::runtime_error("GNNDLDNativeBatched: posix_spawnp a échoué: " + std::string(std::strerror(rc)));
    int status = 0;
    ::waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
        throw std::runtime_error("GNNDLDNativeBatched: gnn_ssp_scriptable_batched.py a échoué (checkpoint invalide ?)");
    return scripted_path;
}

} // namespace

// ── Impl ──────────────────────────────────────────────────────────────────────
struct GNNDLDNativeBatched::Impl {
    SSPInstance inst;
    torch::Device device{torch::kCPU};
    torch::jit::script::Module module;
    SSPGNNBatchQueryTiming cum_timing;
};

GNNDLDNativeBatched::GNNDLDNativeBatched(const std::string& checkpoint_path, const std::string& device,
                                             const SSPInstance& inst)
    : impl_(std::make_unique<Impl>())
{
    impl_->inst = inst;

    if (device == "cuda") {
        if (!torch::cuda::is_available())
            throw std::invalid_argument("GNNDLDNativeBatched: device=cuda demandé mais CUDA indisponible");
        impl_->device = torch::Device(torch::kCUDA);
    } else if (device == "cpu") {
        impl_->device = torch::Device(torch::kCPU);
    } else {
        throw std::invalid_argument("GNNDLDNativeBatched: device inconnu (\"" + device + "\"), attendu cpu|cuda");
    }

    std::string repo_root = "..";
    if (const char* env = std::getenv("SSP_REPO_ROOT")) repo_root = env;

    std::string scripted_path = ensure_scripted_export(checkpoint_path, "python3", repo_root);
    impl_->module = torch::jit::load(scripted_path, impl_->device);
    impl_->module.eval();
}

GNNDLDNativeBatched::~GNNDLDNativeBatched() = default;

void GNNDLDNativeBatched::query_batch(
    const std::vector<std::vector<int8_t>>& x_masks, const std::vector<std::vector<int8_t>>& z_masks,
    const std::vector<double>& curr_costs, const std::vector<int>& depths,
    std::vector<double>& cost_lower_out)
{
    using Clock = std::chrono::steady_clock;
    const SSPInstance& inst = impl_->inst;
    const int B = static_cast<int>(x_masks.size());
    cost_lower_out.assign(B, 0.0);
    if (B == 0) return;
    if (static_cast<int>(z_masks.size()) != B || static_cast<int>(curr_costs.size()) != B ||
        static_cast<int>(depths.size()) != B)
        throw std::invalid_argument("GNNDLDNativeBatched::query_batch : tailles de vecteurs incohérentes");

    auto t0 = Clock::now();

    std::vector<SSPBuiltFeatures> feats;
    feats.reserve(B);
    for (int b = 0; b < B; ++b)
        feats.push_back(build_ssp_features_cpp(inst.E, inst.T, inst.N, inst.salary, inst.demand,
                                                 x_masks[b], z_masks[b], curr_costs[b], depths[b], inst.epsilon));

    const int G = feats[0].G, N_z = feats[0].N_z, T = inst.T;
    for (int b = 1; b < B; ++b)
        if (feats[b].G != G || feats[b].N_z != N_z)
            throw std::invalid_argument(
                "GNNDLDNativeBatched::query_batch : tous les membres du batch doivent partager "
                "le même (G, N_z) -- contrat imposé par le bucketing (G,N_z) côté driver, violé au membre " +
                std::to_string(b));
    if (G == 0 || N_z == 0)
        throw std::runtime_error("GNNDLDNativeBatched::query_batch : G==0 ou N_z==0 -- pas de borne GNN ici");

    const int node_dim = 8;
    const int64_t n_groups = static_cast<int64_t>(N_z) * T;
    const int64_t V = n_groups * G;   // nœuds par graphe

    auto [edge_index_1, edge_weight_1] = build_edges(G, static_cast<int>(n_groups), impl_->device);
    const int64_t Ne = edge_index_1.size(1);

    torch::Tensor offsets = torch::arange(
        B, torch::TensorOptions().dtype(torch::kInt64).device(impl_->device)) * V;
    torch::Tensor ei_rep = edge_index_1.unsqueeze(0).expand({B, 2, Ne}) + offsets.view({B, 1, 1});
    torch::Tensor edge_index_B = ei_rep.permute({1, 0, 2}).reshape({2, B * Ne}).contiguous();
    torch::Tensor edge_weight_B = edge_weight_1.unsqueeze(0).expand({B, Ne}).reshape({B * Ne}).contiguous();

    torch::Tensor X = torch::empty({static_cast<int64_t>(B) * V, node_dim}, torch::kFloat32);
    torch::Tensor ctx = torch::empty({B, 9}, torch::kFloat32);
    for (int b = 0; b < B; ++b) {
        std::memcpy(X.data_ptr<float>() + static_cast<int64_t>(b) * V * node_dim,
                    feats[b].X_flat.data(), sizeof(float) * V * node_dim);
        std::memcpy(ctx.data_ptr<float>() + static_cast<int64_t>(b) * 9,
                    feats[b].ctx.data(), sizeof(float) * 9);
    }

    auto t1 = Clock::now();

    torch::NoGradGuard no_grad;
    torch::Tensor X_dev = X, ei_dev = edge_index_B, ew_dev = edge_weight_B, ctx_dev = ctx;
    if (impl_->device.is_cuda()) {
        X_dev = X_dev.to(impl_->device); ei_dev = ei_dev.to(impl_->device);
        ew_dev = ew_dev.to(impl_->device); ctx_dev = ctx_dev.to(impl_->device);
    }

    std::vector<torch::jit::IValue> inputs = {
        X_dev, ei_dev, ew_dev, ctx_dev,
        static_cast<int64_t>(G), static_cast<int64_t>(T), static_cast<int64_t>(B)
    };
    torch::Tensor h_out = impl_->module.forward(inputs).toTensor();
    if (impl_->device.is_cuda()) h_out = h_out.to(torch::kCPU);
    auto t2 = Clock::now();

    const int64_t per_member = V * SSP_SIGMA_SIZE;
    torch::Tensor h_out_2d = h_out.view({B, per_member});

    double t_zeta_theta_sum = 0.0, t_rest_sum = 0.0;
    for (int b = 0; b < B; ++b) {
        auto tb0 = Clock::now();
        double t_zt = 0.0;
        double L1 = lagrangian_dld_cpp(h_out_2d[b], feats[b], inst, x_masks[b], T, t_zt);
        cost_lower_out[b] = -L1;
        auto tb1 = Clock::now();
        t_zeta_theta_sum += t_zt;
        t_rest_sum += std::chrono::duration<double>(tb1 - tb0).count() - t_zt;
    }
    auto t3 = Clock::now();
    (void)t3;

    auto& ct = impl_->cum_timing;
    ct.t_graph_build_s     += std::chrono::duration<double>(t1 - t0).count();
    ct.t_forward_s         += std::chrono::duration<double>(t2 - t1).count();
    ct.t_zeta_theta_s      += t_zeta_theta_sum;
    ct.t_lagrangian_rest_s += t_rest_sum;
    ct.n_batch_calls       += 1;
    ct.n_nodes_total       += B;
}

SSPGNNBatchQueryTiming GNNDLDNativeBatched::cumulative_timing() const { return impl_->cum_timing; }
