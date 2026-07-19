/*
 * gnn_sd_edit_native_aoti.cpp — Implémentation (cf. gnn_sd_edit_native_aoti.hpp).
 *
 * build_graph_cpp()/lagrangian_v1ld_cpp() dupliqués mot pour mot depuis
 * gnn_sd_edit_native.cpp (cf. .hpp pour la justification de la duplication).
 * SEULE différence : le chargement/appel du modèle utilise
 * torch::inductor::AOTIModelPackageLoader (artefacts .pt2 pré-compilés, un
 * par n_g) au lieu de torch::jit::script::Module (TorchScript, une seule
 * forme, recompilée en interne par le JIT profiling executor).
 */

#include "gnn_sd_edit_native_aoti.hpp"

#include <torch/script.h>
#include <torch/cuda.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

extern "C" void solve_zeta_batch(int N_z, int m, int n_g, const double* cost,
                                  const double* A, const double* b_res, double* x_out);

namespace {

// ── build_graph (IDENTIQUE à gnn_sd_edit_native.cpp) ────────────────────────
struct BuiltGraph {
    torch::Tensor X, edge_index, edge_weight, context;
    Vec3d A_g; Vec2d b_res_g; Vec1d c_g;
    int n_g, N_z, N, n_fixed_z1;
    double c_scale;
};

BuiltGraph build_graph_cpp(int n, int m, int N, const Vec1d& c, const Vec3d& A, const Vec2d& b,
                           const std::vector<int8_t>& x_mask, const Vec2d& b_res,
                           double curr_cost, int depth, double epsilon)
{
    std::vector<int> active_x;
    for (int k = 0; k < n; ++k) if (x_mask[k] == 2) active_x.push_back(k);
    int n_g = static_cast<int>(active_x.size());
    int N_z = N, n_fixed_z1 = 0;

    const Vec2d& b_res_z = b_res;
    const Vec2d& b_z     = b;
    Vec1d c_g(n_g);
    for (int kf = 0; kf < n_g; ++kf) c_g[kf] = c[active_x[kf]];

    Vec3d A_g(N_z, Vec2d(m, Vec1d(n_g)));
    for (int i = 0; i < N_z; ++i)
        for (int j = 0; j < m; ++j)
            for (int kf = 0; kf < n_g; ++kf)
                A_g[i][j][kf] = A[i][j][active_x[kf]];

    double c_g_sum = 0.0; for (double v : c_g) c_g_sum += v;
    double c_scale = std::max(curr_cost + c_g_sum, 1.0);

    Vec2d cap_frac(N_z, Vec1d(m));
    for (int i = 0; i < N_z; ++i)
        for (int j = 0; j < m; ++j) cap_frac[i][j] = b_res_z[i][j] / std::max(b_z[i][j], 1e-9);

    Vec1d scen_min(N_z), scen_mean(N_z), scen_std(N_z), scen_sat(N_z);
    for (int i = 0; i < N_z; ++i) {
        double mn = cap_frac[i][0], sum = 0.0;
        for (int j = 0; j < m; ++j) { mn = std::min(mn, cap_frac[i][j]); sum += cap_frac[i][j]; }
        double mean = sum / m, var = 0.0; int sat = 0;
        for (int j = 0; j < m; ++j) {
            double d = cap_frac[i][j] - mean; var += d * d;
            if (cap_frac[i][j] < 0.1) ++sat;
        }
        scen_min[i] = mn; scen_mean[i] = mean; scen_std[i] = std::sqrt(var / m);
        scen_sat[i] = static_cast<double>(sat) / m;
    }

    Vec3d f_A(N_z, Vec2d(n_g, Vec1d(m)));
    for (int i = 0; i < N_z; ++i)
        for (int kf = 0; kf < n_g; ++kf)
            for (int j = 0; j < m; ++j)
                f_A[i][kf][j] = A_g[i][j][kf] / std::max(b_z[i][j], 1e-9);

    Vec2d A_mean_scen(n_g, Vec1d(m, 0.0)), A_std_scen(n_g, Vec1d(m, 0.0));
    for (int kf = 0; kf < n_g; ++kf)
        for (int j = 0; j < m; ++j) {
            double sum = 0.0; for (int i = 0; i < N_z; ++i) sum += f_A[i][kf][j];
            double mean = sum / N_z, var = 0.0;
            for (int i = 0; i < N_z; ++i) { double d = f_A[i][kf][j] - mean; var += d * d; }
            A_mean_scen[kf][j] = mean; A_std_scen[kf][j] = std::sqrt(var / N_z);
        }

    Vec3d dantzig_rank(N_z, Vec2d(m, Vec1d(n_g)));
    double rank_norm = std::max(n_g - 1, 1);
    for (int i = 0; i < N_z; ++i)
        for (int j = 0; j < m; ++j) {
            std::vector<int> order(n_g); std::iota(order.begin(), order.end(), 0);
            std::vector<double> eff(n_g);
            for (int kf = 0; kf < n_g; ++kf) eff[kf] = c_g[kf] / std::max(A_g[i][j][kf], 1e-9);
            std::sort(order.begin(), order.end(), [&](int a, int b2) { return eff[a] > eff[b2]; });
            for (int r = 0; r < n_g; ++r) dantzig_rank[i][j][order[r]] = r / rank_norm;
        }

    int node_dim = 7 * m + 5;
    std::vector<float> X(static_cast<size_t>(N_z) * n_g * node_dim);
    double tightest = 1e300;
    for (int i = 0; i < N_z; ++i) {
        for (int kf = 0; kf < n_g; ++kf) {
            size_t row = static_cast<size_t>(i) * n_g + kf;
            float* out = &X[row * node_dim];
            int off = 0;
            double f_c = c_g[kf] / c_scale;
            out[off++] = static_cast<float>(f_c);
            for (int j = 0; j < m; ++j) out[off++] = static_cast<float>(f_A[i][kf][j]);
            for (int j = 0; j < m; ++j)
                out[off++] = static_cast<float>(std::min(A_g[i][j][kf] / std::max(b_res_z[i][j], 1e-9), 10.0));
            for (int j = 0; j < m; ++j)
                out[off++] = static_cast<float>(std::log1p(f_c / std::max(f_A[i][kf][j], 1e-9)));
            for (int j = 0; j < m; ++j) out[off++] = static_cast<float>(dantzig_rank[i][j][kf]);
            for (int j = 0; j < m; ++j) out[off++] = static_cast<float>(cap_frac[i][j]);
            out[off++] = static_cast<float>(scen_min[i]);
            out[off++] = static_cast<float>(scen_mean[i]);
            for (int j = 0; j < m; ++j) out[off++] = static_cast<float>(A_mean_scen[kf][j]);
            for (int j = 0; j < m; ++j) out[off++] = static_cast<float>(A_std_scen[kf][j]);
            out[off++] = static_cast<float>(scen_std[i]);
            out[off++] = static_cast<float>(scen_sat[i]);
        }
        for (int j = 0; j < m; ++j) tightest = std::min(tightest, cap_frac[i][j]);
    }

    std::vector<int64_t> src, dst; std::vector<float> ew;
    src.reserve(static_cast<size_t>(N_z) * n_g * (n_g - 1) + 2 * static_cast<size_t>(n_g) * std::max(N_z - 1, 0));
    dst.reserve(src.capacity()); ew.reserve(src.capacity());
    for (int i = 0; i < N_z; ++i)
        for (int k1 = 0; k1 < n_g; ++k1)
            for (int k2 = k1 + 1; k2 < n_g; ++k2) {
                int64_t a = static_cast<int64_t>(i) * n_g + k1, b2 = static_cast<int64_t>(i) * n_g + k2;
                src.push_back(a); dst.push_back(b2); ew.push_back(1.0f / n_g);
                src.push_back(b2); dst.push_back(a); ew.push_back(1.0f / n_g);
            }
    for (int kf = 0; kf < n_g; ++kf) {
        int64_t anchor = static_cast<int64_t>(0) * n_g + kf;
        for (int i2 = 1; i2 < N_z; ++i2) {
            int64_t spoke = static_cast<int64_t>(i2) * n_g + kf;
            src.push_back(anchor); dst.push_back(spoke); ew.push_back(1.0f);
            src.push_back(spoke); dst.push_back(anchor); ew.push_back(1.0f);
        }
    }

    BuiltGraph g;
    g.X = torch::from_blob(X.data(), {static_cast<int64_t>(N_z) * n_g, node_dim}, torch::kFloat32).clone();
    g.edge_index = torch::stack({
        torch::from_blob(src.data(), {(int64_t)src.size()}, torch::kInt64).clone(),
        torch::from_blob(dst.data(), {(int64_t)dst.size()}, torch::kInt64).clone()});
    g.edge_weight = torch::from_blob(ew.data(), {(int64_t)ew.size()}, torch::kFloat32).clone();

    double depth_norm = static_cast<double>(depth) / std::max(n, 1);
    double frac_fixed = 1.0 - static_cast<double>(n_g) / n;
    std::vector<float> ctx = {
        static_cast<float>(depth_norm), static_cast<float>(frac_fixed), static_cast<float>(tightest),
        static_cast<float>(curr_cost / c_scale), static_cast<float>(epsilon),
        static_cast<float>(static_cast<double>(n_fixed_z1) / N),
        static_cast<float>(std::log(n_g)), static_cast<float>(std::log(N_z)),
        static_cast<float>(static_cast<double>(N_z) / n_g)
    };
    g.context = torch::from_blob(ctx.data(), {9}, torch::kFloat32).clone();
    g.A_g = A_g; g.b_res_g = b_res_z; g.c_g = c_g;
    g.n_g = n_g; g.N_z = N_z; g.N = N; g.n_fixed_z1 = n_fixed_z1; g.c_scale = c_scale;
    return g;
}

// ── lagrangian_v1ld (IDENTIQUE à gnn_sd_edit_native.cpp) ────────────────────
double lagrangian_v1ld_cpp(const torch::Tensor& lam, const BuiltGraph& g, double epsilon) {
    int N_z = g.N_z, n_g = g.n_g, m = static_cast<int>(g.A_g[0].size());
    auto lam_2d = lam.view({N_z, n_g}).to(torch::kFloat64).contiguous();
    auto lam_acc = lam_2d.accessor<double, 2>();
    double p_unif = 1.0 / g.N;

    Vec1d Lambda_sum(n_g, 0.0);
    for (int i = 0; i < N_z; ++i) for (int k = 0; k < n_g; ++k) Lambda_sum[k] += lam_acc[i][k];

    Vec2d cost(N_z, Vec1d(n_g));
    for (int i = 0; i < N_z; ++i)
        for (int k = 0; k < n_g; ++k)
            cost[i][k] = g.c_g[k] + lam_acc[i][k] / p_unif - Lambda_sum[k];

    std::vector<double> cost_flat(static_cast<size_t>(N_z) * n_g);
    std::vector<double> A_flat(static_cast<size_t>(N_z) * m * n_g);
    std::vector<double> b_flat(static_cast<size_t>(N_z) * m);
    for (int i = 0; i < N_z; ++i) {
        for (int k = 0; k < n_g; ++k) cost_flat[i * n_g + k] = cost[i][k];
        for (int j = 0; j < m; ++j) {
            b_flat[i * m + j] = g.b_res_g[i][j];
            for (int k = 0; k < n_g; ++k) A_flat[(size_t)i * m * n_g + j * n_g + k] = g.A_g[i][j][k];
        }
    }
    std::vector<double> x_zeta(static_cast<size_t>(N_z) * n_g, 0.0);
    solve_zeta_batch(N_z, m, n_g, cost_flat.data(), A_flat.data(), b_flat.data(), x_zeta.data());

    Vec1d theta(N_z), zeta(N_z), Delta(N_z);
    for (int i = 0; i < N_z; ++i) {
        double th = 0.0, ze = 0.0;
        for (int k = 0; k < n_g; ++k) {
            if (cost[i][k] > 0) th += cost[i][k];
            ze += cost[i][k] * x_zeta[i * n_g + k];
        }
        theta[i] = th; zeta[i] = ze;
        Delta[i] = std::max(0.0, th - ze);
    }

    double required_prob = std::max(0.0, (1.0 - epsilon) - g.n_fixed_z1 * p_unif);
    std::vector<int> order(N_z); std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b2) { return Delta[a] < Delta[b2]; });

    Vec1d y_star(N_z, 0.0);
    double cumprob = 0.0;
    for (int idx : order) {
        if (required_prob <= 0.0) break;
        y_star[idx] = 1.0;
        cumprob += p_unif;
        if (cumprob >= required_prob - 1e-9) break;
    }

    double L1_theta = 0.0, val_lam = 0.0;
    for (int i = 0; i < N_z; ++i) {
        L1_theta += p_unif * theta[i];
        val_lam  += p_unif * y_star[i] * (theta[i] - zeta[i]);
    }
    return L1_theta - val_lam;
}

// ── Export AOTInductor (fork python, cache disque par n_g) ──────────────────
bool file_exists(const std::string& p) { struct stat st; return ::stat(p.c_str(), &st) == 0; }
double file_mtime(const std::string& p) { struct stat st; if (::stat(p.c_str(), &st) != 0) return -1.0; return static_cast<double>(st.st_mtime); }

// Répertoire de cache — DOIT matcher cache_dir_for() dans export_aoti_sdedit.py.
std::string aoti_cache_dir(const std::string& checkpoint_path, int N) {
    return checkpoint_path + ".aoti_N" + std::to_string(N);
}

// S'assure que les artefacts .pt2 pour n_g=1..n existent et sont à jour
// (mtime >= checkpoint) ; sinon lance export_aoti_sdedit.py qui exporte
// UNIQUEMENT les formes manquantes/périmées (~15-20s/forme la première
// fois, quasi instantané ensuite). Cf. GNNSdEditNative::ensure_scripted_export
// pour le mécanisme posix_spawn analogue (TorchScript, une seule forme).
void ensure_aoti_exports(const std::string& checkpoint_path, const std::string& device,
                         int n, int N, const std::string& python_exe)
{
    std::string cache_dir = aoti_cache_dir(checkpoint_path, N);
    double ckpt_mtime = file_mtime(checkpoint_path);

    bool all_fresh = true;
    for (int n_g = 1; n_g <= n; ++n_g) {
        std::string pkg = cache_dir + "/ng" + std::to_string(n_g) + ".pt2";
        if (!file_exists(pkg) || file_mtime(pkg) < ckpt_mtime) { all_fresh = false; break; }
    }
    if (all_fresh) return;

    std::string script = "gnn_sd_edit/export_aoti_sdedit.py";
    std::vector<std::string> args_str = {
        python_exe, script,
        "--checkpoint", checkpoint_path,
        "--n-z", std::to_string(N),
        "--device", device,
        "--n-g",
    };
    for (int n_g = 1; n_g <= n; ++n_g) args_str.push_back(std::to_string(n_g));

    std::vector<char*> argv;
    argv.reserve(args_str.size() + 1);
    for (auto& s : args_str) argv.push_back(const_cast<char*>(s.c_str()));
    argv.push_back(nullptr);

    pid_t pid = 0;
    int rc = ::posix_spawnp(&pid, python_exe.c_str(), nullptr, nullptr, argv.data(), environ);
    if (rc != 0)
        throw std::runtime_error("GNNSdEditNativeAOTI: posix_spawnp a échoué: " + std::string(std::strerror(rc)));
    int status = 0;
    ::waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
        throw std::runtime_error("GNNSdEditNativeAOTI: export_aoti_sdedit.py a échoué (checkpoint invalide ?)");
}

} // namespace

// ── Impl ──────────────────────────────────────────────────────────────────────
struct GNNSdEditNativeAOTI::Impl {
    int n, m, N;
    double epsilon;
    Vec1d c; Vec3d A; Vec2d b;
    torch::Device device{torch::kCPU};
    std::map<int, std::unique_ptr<torch::inductor::AOTIModelPackageLoader>> loaders;  // clé = n_g
};

GNNSdEditNativeAOTI::GNNSdEditNativeAOTI(const std::string& checkpoint_path, const std::string& device,
                                         int n, int m, int N, double epsilon,
                                         const Vec1d& c, const Vec1d& p, const Vec3d& A, const Vec2d& b)
    : impl_(std::make_unique<Impl>())
{
    const double p_unif = 1.0 / static_cast<double>(N);
    for (double pi : p)
        if (std::abs(pi - p_unif) > 1e-9)
            throw std::invalid_argument(
                "GNNSdEditNativeAOTI : p non-uniforme — lagrangian_v1ld suppose des "
                "scénarios équiprobables (p_i = 1/N)");

    if (device == "cuda") {
        if (!torch::cuda::is_available())
            throw std::invalid_argument("GNNSdEditNativeAOTI: device=cuda demandé mais CUDA indisponible");
        impl_->device = torch::Device(torch::kCUDA);
    } else if (device == "cpu") {
        impl_->device = torch::Device(torch::kCPU);
    } else {
        throw std::invalid_argument("GNNSdEditNativeAOTI: device inconnu (\"" + device + "\"), attendu cpu|cuda");
    }

    impl_->n = n; impl_->m = m; impl_->N = N; impl_->epsilon = epsilon;
    impl_->c = c; impl_->A = A; impl_->b = b;

    ensure_aoti_exports(checkpoint_path, device, n, N, "python3");

    std::string cache_dir = aoti_cache_dir(checkpoint_path, N);
    for (int n_g = 1; n_g <= n; ++n_g) {
        std::string pkg = cache_dir + "/ng" + std::to_string(n_g) + ".pt2";
        impl_->loaders[n_g] = std::make_unique<torch::inductor::AOTIModelPackageLoader>(pkg);
    }
}

GNNSdEditNativeAOTI::~GNNSdEditNativeAOTI() = default;

double GNNSdEditNativeAOTI::query(const std::vector<int8_t>& x_mask, const Vec2d& b_res,
                                   double curr_cost, int depth, GNNCallTiming& timing)
{
    using Clock = std::chrono::steady_clock;

    auto t0 = Clock::now();
    // b_res est déjà à jour (maintenu incrémentalement par l'appelant, cf.
    // MKPSpaceV1LD::b_res dans gecode_bab.hpp) — ne PLUS le recalculer ici
    // depuis x_mask (cf. revue niveau 1 point B).
    BuiltGraph g = build_graph_cpp(impl_->n, impl_->m, impl_->N, impl_->c, impl_->A, impl_->b,
                                   x_mask, b_res, curr_cost, depth, impl_->epsilon);
    auto t1 = Clock::now();

    torch::NoGradGuard no_grad;
    torch::Tensor X = g.X, edge_index = g.edge_index, edge_weight = g.edge_weight, context = g.context;
    if (impl_->device.is_cuda()) {
        X = X.to(impl_->device);
        edge_index = edge_index.to(impl_->device);
        edge_weight = edge_weight.to(impl_->device);
        context = context.to(impl_->device);
    }

    auto it = impl_->loaders.find(g.n_g);
    if (it == impl_->loaders.end())
        throw std::runtime_error("GNNSdEditNativeAOTI: aucun artefact AOTInductor pour n_g=" +
                                  std::to_string(g.n_g) + " (hors plage 1.." + std::to_string(impl_->n) + ")");

    std::vector<at::Tensor> inputs{X, edge_index, edge_weight, context};
    torch::Tensor h_out = it->second->run(inputs)[0];
    if (impl_->device.is_cuda()) h_out = h_out.to(torch::kCPU);

    torch::Tensor lam = h_out * g.c_scale / static_cast<double>(g.N);
    auto t2 = Clock::now();

    double ub_relative = lagrangian_v1ld_cpp(lam, g, impl_->epsilon);
    auto t3 = Clock::now();

    timing.t_graph_build_s   = std::chrono::duration<double>(t1 - t0).count();
    timing.t_forward_s       = std::chrono::duration<double>(t2 - t1).count();
    timing.t_bound_compute_s = std::chrono::duration<double>(t3 - t2).count();
    return ub_relative;
}
