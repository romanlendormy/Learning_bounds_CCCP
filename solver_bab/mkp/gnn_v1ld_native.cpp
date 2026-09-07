/*
 * gnn_v1ld_native.cpp — Implémentation (cf. gnn_v1ld_native.hpp).
 *
 * Inférence via TorchScript (torch::jit::script::Module), pas de
 * réimplémentation manuelle de l'architecture GNN en C++ — même approche que
 * solver_swann_darius.cpp (torch::jit::load() + .forward()), à la différence
 * près que ce dernier fait ensuite du raffinement itératif post-GNN
 * (activate_learning_and_grad) : ici la borne reste one-shot pure, comme
 * l'exige le reste du projet (cf. gnn_bound_backend.hpp / mode GNN existant).
 *
 * Le module TorchScript (gnn_sd_edit/gnn_v1ld_scriptable.py::GNNScriptable, exporté
 * une seule fois via gnn_sd_edit/gnn_v1ld_scriptable.py::export, mis en cache à côté
 * du checkpoint) réimplémente gnn.train_gnn_v1ld.GNN en torch pur — sans
 * torch_geometric, dont Data/MessagePassing ne sont pas scriptables
 * ("Compiled functions can't take variable number of arguments"). Le
 * recentrage (center_lam) est baké dans le module exporté lui-même — le
 * module est autonome (m/hidden/use_max_pool/center_lam ne sont PAS relus
 * côté C++, ils sont capturés dans le graphe scripté au moment de l'export).
 *
 * build_graph() (features 7m+5, arêtes item/scénario) et lagrangian_v1ld()
 * (theta/zeta/y*, ζ via solve_zeta_batch) restent réimplémentés à la main —
 * ce ne sont pas le modèle, juste de la construction de tenseurs et du calcul
 * de borne, comme le fait aussi solver_swann_darius.cpp pour son graphe.
 */

#include "gnn_v1ld_native.hpp"

#include <torch/script.h>
#include <torch/cuda.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <fstream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

extern "C" void solve_zeta_batch(int N_z, int m, int n_g, const double* cost,
                                  const double* A, const double* b_res, double* x_out);

namespace {

// ── build_graph (port de gnn.train_gnn_v1ld.build_graph) ────────────────────
// z_mask est toujours "libre" dans notre usage B&B (le more() C++ appelant ne
// fixe jamais aucun scénario) — cf. même hypothèse documentée dans
// gnn_inference_server.py. N_z = N, n_fixed_z1 = 0 en conséquence.
struct BuiltGraph {
    torch::Tensor X, edge_index, edge_weight, context;
    Vec3d A_g; Vec2d b_res_g; Vec1d c_g;
    int n_g, N_z, N, n_fixed_z1;
    double c_scale;
    // Diagnostic (cf. discussion "construction héritée du parent") : part du
    // temps de construction passée dans le calcul des features de nœuds vs
    // dans la génération des index/poids d'arêtes.
    double t_features_s = 0.0, t_edges_s = 0.0;
};

// ── Templates d'arêtes précalculés UNE FOIS (à la construction de
//    GNNV1LDNative, pas à chaque nœud) — validés numériquement équivalents
//    à la construction scalaire dans gnn_edge_vectorized_probe.cpp (mêmes
//    arêtes/poids, écarts CPU 1.05x-7.2x et CUDA 1.27x-12.3x selon n_g).
//
// sc_i1/sc_i2  : paires de scénarios (i1<i2) — dépend SEULEMENT de N, qui ne
//                rétrécit jamais dans l'arbre B&B (z_mask toujours libre).
// item_i1/i2[s]: paires d'items (k1<k2) pour n_g==s — s ne prend que n+1
//                valeurs possibles au total (mêmes n+1 formes que le
//                pré-chauffage TorchScript déjà en place).
struct EdgeTemplates {
    torch::Tensor sc_i1, sc_i2;
    std::vector<torch::Tensor> item_i1, item_i2;

    EdgeTemplates() = default;
    EdgeTemplates(int n, int N_z, torch::Device device) {
        auto opts = torch::TensorOptions().dtype(torch::kInt64).device(device);
        if (N_z >= 2) {
            auto sc_tri = torch::triu_indices(N_z, N_z, /*offset=*/1, opts);
            sc_i1 = sc_tri[0].contiguous(); sc_i2 = sc_tri[1].contiguous();
        }
        item_i1.resize(n + 1); item_i2.resize(n + 1);
        for (int s = 2; s <= n; ++s) {
            auto tri = torch::triu_indices(s, s, /*offset=*/1, opts);
            item_i1[s] = tri[0].contiguous(); item_i2[s] = tri[1].contiguous();
        }
    }
};

// ── Construction vectorisée des arêtes (templates + broadcast torch) ────────
// Remplace les boucles C++ scalaires + push_back — mêmes arêtes/poids,
// produits directement sur `device` (évite le transfert H2D séparé que
// payait la version scalaire pour edge_index/edge_weight).
std::pair<torch::Tensor, torch::Tensor> build_edges_vectorized(
    int n_g, int N_z, const EdgeTemplates& tpl, torch::Device device)
{
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    torch::Tensor ii_src = torch::empty({0}, opts_i64), ii_dst = torch::empty({0}, opts_i64),
                  ii_w   = torch::empty({0}, opts_f32);
    if (n_g >= 2) {
        const auto& i1 = tpl.item_i1[n_g];
        const auto& i2 = tpl.item_i2[n_g];
        auto scen_off = torch::arange(N_z, opts_i64) * n_g;                       // [N_z]
        auto fwd_src  = (i1.unsqueeze(0) + scen_off.unsqueeze(1)).flatten();      // [N_z * P_i]
        auto fwd_dst  = (i2.unsqueeze(0) + scen_off.unsqueeze(1)).flatten();
        ii_src = torch::cat({fwd_src, fwd_dst});
        ii_dst = torch::cat({fwd_dst, fwd_src});
        ii_w   = torch::full({ii_src.size(0)}, 1.0f / n_g, opts_f32);
    }

    torch::Tensor ss_src = torch::empty({0}, opts_i64), ss_dst = torch::empty({0}, opts_i64),
                  ss_w   = torch::empty({0}, opts_f32);
    if (N_z >= 2) {
        auto item_off = torch::arange(n_g, opts_i64);                            // [n_g]
        auto fwd_src  = (tpl.sc_i1.unsqueeze(1) * n_g + item_off.unsqueeze(0)).flatten(); // [P_s * n_g]
        auto fwd_dst  = (tpl.sc_i2.unsqueeze(1) * n_g + item_off.unsqueeze(0)).flatten();
        ss_src = torch::cat({fwd_src, fwd_dst});
        ss_dst = torch::cat({fwd_dst, fwd_src});
        ss_w   = torch::full({ss_src.size(0)}, 1.0f / N_z, opts_f32);
    }

    auto edge_index  = torch::stack({torch::cat({ii_src, ss_src}), torch::cat({ii_dst, ss_dst})});
    auto edge_weight = torch::cat({ii_w, ss_w});
    return {edge_index, edge_weight};
}

BuiltGraph build_graph_cpp(int n, int m, int N, const Vec1d& c, const Vec3d& A, const Vec2d& b,
                           const std::vector<int8_t>& x_mask, const Vec2d& b_res,
                           double curr_cost, int depth, double epsilon,
                           const EdgeTemplates& edge_tpl, torch::Device device)
{
    auto t_build_start = std::chrono::steady_clock::now();
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

    BuiltGraph g;
    g.X = torch::from_blob(X.data(), {static_cast<int64_t>(N_z) * n_g, node_dim}, torch::kFloat32).clone();

    auto t_features_done = std::chrono::steady_clock::now();
    g.t_features_s = std::chrono::duration<double>(t_features_done - t_build_start).count();

    auto [edge_index_v, edge_weight_v] = build_edges_vectorized(n_g, N_z, edge_tpl, device);
    g.edge_index = edge_index_v;
    g.edge_weight = edge_weight_v;

    auto t_edges_done = std::chrono::steady_clock::now();
    g.t_edges_s = std::chrono::duration<double>(t_edges_done - t_features_done).count();

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

// ── lagrangian_v1ld (port de gnn.train_gnn_v1ld.lagrangian_v1ld) ────────────
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

// ── Export TorchScript one-shot (fork python, mis en cache) ─────────────────
// Le checkpoint complet (dict {"model":OrderedDict,"optimizer":...,"args":...})
// n'est pas directement utilisable en C++ : c'est un objet Python générique.
// gnn_sd_edit/gnn_v1ld_scriptable.py fait le pont — UNE SEULE FOIS par checkpoint
// (résultat mis en cache à côté du .pt), pas un process persistant.
bool file_exists(const std::string& p) { struct stat st; return ::stat(p.c_str(), &st) == 0; }
double file_mtime(const std::string& p) { struct stat st; if (::stat(p.c_str(), &st) != 0) return -1.0; return static_cast<double>(st.st_mtime); }

std::string ensure_scripted_export(const std::string& checkpoint_path, const std::string& python_exe) {
    std::string scripted_path = checkpoint_path + ".scripted.pt";
    if (file_exists(scripted_path) && file_mtime(scripted_path) >= file_mtime(checkpoint_path))
        return scripted_path;

    // Export toujours en CPU : torch::jit::load(path, device) remappe les
    // tenseurs vers le device voulu au chargement (pas besoin de CUDA
    // disponible côté Python au moment de l'export, ni de fichier par device).
    std::string script = "gnn_sd_edit/gnn_v1ld_scriptable.py";

    // posix_spawn (pas fork()+exec()) : ce process a déjà chargé LibTorch —
    // cf. gnn_v1ld_native.hpp pour le vrai bug rencontré en dev (pas un
    // problème de fork() en soi, mais LD_LIBRARY_PATH pollué — ne PAS ajouter
    // external/libtorch/lib à LD_LIBRARY_PATH dans les scripts de lancement,
    // le rpath du binaire suffit).
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
        throw std::runtime_error("GNNV1LDNative: posix_spawnp a échoué: " + std::string(std::strerror(rc)));
    int status = 0;
    ::waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
        throw std::runtime_error("GNNV1LDNative: gnn_v1ld_scriptable.py a échoué (checkpoint invalide ?)");
    return scripted_path;
}

} // namespace

// ── Impl ──────────────────────────────────────────────────────────────────────
struct GNNV1LDNative::Impl {
    int n, m, N;
    double epsilon;
    Vec1d c; Vec3d A; Vec2d b;
    torch::Device device{torch::kCPU};
    torch::jit::script::Module module;
    EdgeTemplates edge_templates;  // précalculés une fois (cf. discussion "arêtes vectorisées")
};

GNNV1LDNative::GNNV1LDNative(const std::string& checkpoint_path, const std::string& device,
                             int n, int m, int N, double epsilon,
                             const Vec1d& c, const Vec1d& p, const Vec3d& A, const Vec2d& b)
    : impl_(std::make_unique<Impl>())
{
    const double p_unif = 1.0 / static_cast<double>(N);
    for (double pi : p)
        if (std::abs(pi - p_unif) > 1e-9)
            throw std::invalid_argument(
                "GNNV1LDNative : p non-uniforme — lagrangian_v1ld suppose des "
                "scénarios équiprobables (p_i = 1/N)");

    if (device == "cuda") {
        if (!torch::cuda::is_available())
            throw std::invalid_argument("GNNV1LDNative: device=cuda demandé mais CUDA indisponible "
                                         "(LibTorch CPU-only lié, ou pas de GPU/driver visible)");
        impl_->device = torch::Device(torch::kCUDA);
    } else if (device == "cpu") {
        impl_->device = torch::Device(torch::kCPU);
    } else {
        throw std::invalid_argument("GNNV1LDNative: device inconnu (\"" + device + "\"), attendu cpu|cuda");
    }

    impl_->n = n; impl_->m = m; impl_->N = N; impl_->epsilon = epsilon;
    impl_->c = c; impl_->A = A; impl_->b = b;
    impl_->edge_templates = EdgeTemplates(n, N, impl_->device);

    std::string scripted_path = ensure_scripted_export(checkpoint_path, "python3");
    impl_->module = torch::jit::load(scripted_path, impl_->device);
    impl_->module.eval();
}

GNNV1LDNative::~GNNV1LDNative() = default;

double GNNV1LDNative::query(const std::vector<int8_t>& x_mask, const Vec2d& b_res,
                            double curr_cost, int depth, GNNCallTiming& timing)
{
    using Clock = std::chrono::steady_clock;

    auto t0 = Clock::now();
    // b_res est déjà à jour (maintenu incrémentalement par l'appelant, cf.
    // MKPSpaceV1LD::b_res dans gecode_bab.hpp) — ne PLUS le recalculer ici
    // depuis x_mask (cf. revue niveau 1 point B : l'ancienne version
    // reparcourait toutes les variables fixées à 1 depuis la racine à chaque
    // appel, coût croissant avec la profondeur).
    BuiltGraph g = build_graph_cpp(impl_->n, impl_->m, impl_->N, impl_->c, impl_->A, impl_->b,
                                   x_mask, b_res, curr_cost, depth, impl_->epsilon,
                                   impl_->edge_templates, impl_->device);
    auto t1 = Clock::now();

    torch::NoGradGuard no_grad;
    torch::Tensor X = g.X, edge_index = g.edge_index, edge_weight = g.edge_weight, context = g.context;
    if (impl_->device.is_cuda()) {
        X = X.to(impl_->device);
        edge_index = edge_index.to(impl_->device);
        edge_weight = edge_weight.to(impl_->device);
        context = context.to(impl_->device);
    }

    std::vector<torch::jit::IValue> inputs = {X, edge_index, edge_weight, context, static_cast<int64_t>(g.n_g)};
    torch::Tensor h_out = impl_->module.forward(inputs).toTensor();
    if (impl_->device.is_cuda()) h_out = h_out.to(torch::kCPU);

    torch::Tensor lam = h_out * g.c_scale / static_cast<double>(g.N);
    auto t2 = Clock::now();

    double ub_relative = lagrangian_v1ld_cpp(lam, g, impl_->epsilon);
    auto t3 = Clock::now();

    timing.t_graph_build_s    = std::chrono::duration<double>(t1 - t0).count();
    timing.t_graph_features_s = g.t_features_s;
    timing.t_graph_edges_s    = g.t_edges_s;
    timing.t_forward_s       = std::chrono::duration<double>(t2 - t1).count();
    timing.t_bound_compute_s = std::chrono::duration<double>(t3 - t2).count();
    return ub_relative;
}
