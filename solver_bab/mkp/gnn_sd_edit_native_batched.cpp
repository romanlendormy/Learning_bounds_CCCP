/*
 * gnn_sd_edit_native_batched.cpp — Implémentation (cf.
 * gnn_sd_edit_native_batched.hpp).
 *
 * Différence structurelle avec gnn_sd_edit_native.cpp::build_graph_cpp() :
 * la construction est scindée en deux parties, exactement pour permettre le
 * batching :
 *   - build_features_cpp() : tout ce qui dépend du masque b_res/mask/coût du
 *     membre b (features de nœuds X, contexte 9-dim, A_g/b_res_g/c_g pour ζ)
 *     — appelé UNE FOIS PAR MEMBRE du batch (B fois).
 *   - build_edges_sd_edit_cpp() : la topologie d'arêtes (item_varies clique +
 *     scenario_varies étoile), qui ne dépend QUE de (n_g, N_z) — donc
 *     IDENTIQUE pour tous les membres d'un même batch (même n_g garanti par
 *     l'appelant, cf. gnn_bound_backend_batch.hpp) — appelée UNE SEULE FOIS
 *     par appel query_batch(), puis dupliquée B fois avec offset (bloc-
 *     diagonal, cf. gnn_sd_edit_scriptable_batched.py).
 *
 * lagrangian_sld_cpp (calcul de ζ/L1(λ)) reste appelé B fois séparément
 * (une fois par membre, sur ses propres A_g/b_res_g/c_g) — PAS batchable
 * simplement (dépend de données propres à chaque masque), et solve_zeta_batch
 * est déjà parallélisé OpenMP par scénario (pas le goulot ciblé par cette
 * variante).
 */

#include "gnn_sd_edit_native_batched.hpp"
#include "gnn_sd_edit_scriptable_batched_export.hpp"

#include <torch/script.h>
#include <torch/cuda.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

extern "C" void solve_zeta_batch(int N_z, int m, int n_g, const double* cost,
                                  const double* A, const double* b_res, double* x_out);
extern "C" void solve_zeta_batch_surrogate(int N_z, int m, int n_g, const double* cost,
                                            const double* A, const double* b_res, double* x_out);

namespace {

// ── Quantités qui ne dépendent QUE de active_x (l'ensemble des items encore
// libres), jamais de b_res/fp — donc EXACTEMENT réutilisables entre deux
// nœuds qui partagent le même active_x, indépendamment de quels items ont
// été fixés à 1 pour y arriver. Extrait de build_features_cpp (ex-lignes
// 63-77/99-123) pour permettre un cache par active_x dans Impl (cf.
// Impl::active_x_cache) — motivé par l'instrumentation active_set_counts
// (26 ensembles distincts sur ~20000 appels, cf. échange du 2026-07-17).
//
// ⚠️ NE PAS cacher par profondeur/n_g seul (contrairement aux arêtes,
// cf. edge_cache) : choice() (gecode_bab.hpp:542-545) saute les variables
// déjà fixées par propagation CP, donc active_x n'est pas une fonction pure
// de la profondeur — deux nœuds à la même profondeur peuvent avoir des
// active_x différents. La clé DOIT être l'ensemble exact (cf. active_key
// dans query_batch), pas n_g/depth.
struct ActiveXInvariant {
    std::vector<int> active_x;
    Vec1d c_g;
    Vec3d A_g;          // (N_z, m, n_g)
    double c_g_sum;
    Vec3d f_A;          // (N_z, n_g, m) = A_g / b (global, jamais b_res)
    Vec3d dantzig_rank;  // (N_z, m, n_g)
    Vec2d A_mean_scen, A_std_scen;  // (n_g, m)
};

ActiveXInvariant compute_active_x_invariant(
    int m, int N, const Vec1d& c, const Vec3d& A, const Vec2d& b,
    const std::vector<int8_t>& x_mask)
{
    const int n = static_cast<int>(x_mask.size());
    const int N_z = N;
    ActiveXInvariant inv;

    for (int k = 0; k < n; ++k) if (x_mask[k] == 2) inv.active_x.push_back(k);
    const int n_g = static_cast<int>(inv.active_x.size());

    inv.c_g.resize(n_g);
    for (int kf = 0; kf < n_g; ++kf) inv.c_g[kf] = c[inv.active_x[kf]];

    inv.A_g.assign(N_z, Vec2d(m, Vec1d(n_g)));
    for (int i = 0; i < N_z; ++i)
        for (int j = 0; j < m; ++j)
            for (int kf = 0; kf < n_g; ++kf)
                inv.A_g[i][j][kf] = A[i][j][inv.active_x[kf]];

    inv.c_g_sum = 0.0; for (double v : inv.c_g) inv.c_g_sum += v;

    inv.f_A.assign(N_z, Vec2d(n_g, Vec1d(m)));
    for (int i = 0; i < N_z; ++i)
        for (int kf = 0; kf < n_g; ++kf)
            for (int j = 0; j < m; ++j)
                inv.f_A[i][kf][j] = inv.A_g[i][j][kf] / std::max(b[i][j], 1e-9);

    inv.A_mean_scen.assign(n_g, Vec1d(m, 0.0));
    inv.A_std_scen.assign(n_g, Vec1d(m, 0.0));
    for (int kf = 0; kf < n_g; ++kf)
        for (int j = 0; j < m; ++j) {
            double sum = 0.0; for (int i = 0; i < N_z; ++i) sum += inv.f_A[i][kf][j];
            double mean = sum / N_z, var = 0.0;
            for (int i = 0; i < N_z; ++i) { double d = inv.f_A[i][kf][j] - mean; var += d * d; }
            inv.A_mean_scen[kf][j] = mean; inv.A_std_scen[kf][j] = std::sqrt(var / N_z);
        }

    inv.dantzig_rank.assign(N_z, Vec2d(m, Vec1d(n_g)));
    double rank_norm = std::max(n_g - 1, 1);
    for (int i = 0; i < N_z; ++i)
        for (int j = 0; j < m; ++j) {
            std::vector<int> order(n_g); std::iota(order.begin(), order.end(), 0);
            std::vector<double> eff(n_g);
            for (int kf = 0; kf < n_g; ++kf) eff[kf] = inv.c_g[kf] / std::max(inv.A_g[i][j][kf], 1e-9);
            std::sort(order.begin(), order.end(), [&](int a, int b2) { return eff[a] > eff[b2]; });
            for (int r = 0; r < n_g; ++r) inv.dantzig_rank[i][j][order[r]] = r / rank_norm;
        }

    return inv;
}

// ── Features seules (sans arêtes) — port de build_graph_cpp minus lignes
// 154-180/184-187 de gnn_sd_edit_native.cpp ────────────────────────────────
struct BuiltFeatures {
    std::vector<float> X_flat;   // (N_z*n_g, node_dim), row-major
    std::vector<float> ctx;      // (9,)
    Vec3d A_g; Vec2d b_res_g; Vec1d c_g;
    int n_g, N_z, N, n_fixed_z1, node_dim;
    double c_scale;
};

// inv : quantités active_x-invariantes déjà calculées (cache hit ou miss
// venant d'être rempli par l'appelant, cf. Impl::active_x_cache dans
// query_batch) — build_features_cpp ne recalcule plus que ce qui dépend de
// b_res/curr_cost (cap_frac, scen_*, c_scale, et les colonnes de X qui en
// dépendent).
//
// ⚠️ Tentative de vectorisation batchée (torch::Tensor) de cap_frac/scen_*
// essayée puis ABANDONNÉE le 2026-07-18 : régression mesurée (construction
// graphe 2.05s→2.63s, +28%) — le volume de données (B×N×m ~ quelques
// dizaines de milliers de doubles) est trop petit pour amortir le coût de
// dispatch par opération d'ATen (allocation de tenseur + ~6 appels séparés),
// contrairement au forward GNN où cette même logique paie (gros calcul GPU).
// Boucles scalaires ci-dessous : plus rapides pour ce volume, gardées telles
// quelles.
BuiltFeatures build_features_cpp(int n, int m, int N, const ActiveXInvariant& inv, const Vec2d& b,
                                  const Vec2d& b_res,
                                  double curr_cost, int depth, double epsilon)
{
    const int n_g = static_cast<int>(inv.active_x.size());
    int N_z = N, n_fixed_z1 = 0;

    const Vec2d& b_res_z = b_res;
    const Vec2d& b_z     = b;
    const Vec1d& c_g = inv.c_g;
    const Vec3d& A_g = inv.A_g;
    const Vec3d& f_A = inv.f_A;
    const Vec3d& dantzig_rank = inv.dantzig_rank;
    const Vec2d& A_mean_scen = inv.A_mean_scen;
    const Vec2d& A_std_scen = inv.A_std_scen;

    double c_scale = std::max(curr_cost + inv.c_g_sum, 1.0);

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

    double depth_norm = static_cast<double>(depth) / std::max(n, 1);
    double frac_fixed = 1.0 - static_cast<double>(n_g) / n;
    std::vector<float> ctx = {
        static_cast<float>(depth_norm), static_cast<float>(frac_fixed), static_cast<float>(tightest),
        static_cast<float>(curr_cost / c_scale), static_cast<float>(epsilon),
        static_cast<float>(static_cast<double>(n_fixed_z1) / N),
        static_cast<float>(std::log(n_g)), static_cast<float>(std::log(N_z)),
        static_cast<float>(static_cast<double>(N_z) / n_g)
    };

    BuiltFeatures f;
    f.X_flat = std::move(X);
    f.ctx = std::move(ctx);
    f.A_g = A_g; f.b_res_g = b_res_z; f.c_g = c_g;
    f.n_g = n_g; f.N_z = N_z; f.N = N; f.n_fixed_z1 = n_fixed_z1; f.node_dim = node_dim;
    f.c_scale = c_scale;
    return f;
}

// ── Arêtes sd_edit (item_varies clique + scenario_varies étoile) — port de
// gnn_sd_edit_native.cpp:154-187, ne dépend que de (n_g, N_z). Appelée UNE
// SEULE FOIS par appel query_batch() (premier gain, même sans compter
// l'amortissement GPU du forward). ──────────────────────────────────────────
void build_edges_sd_edit_cpp(int n_g, int N_z, torch::Tensor& edge_index, torch::Tensor& edge_weight)
{
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
    edge_index = torch::stack({
        torch::from_blob(src.data(), {(int64_t)src.size()}, torch::kInt64).clone(),
        torch::from_blob(dst.data(), {(int64_t)dst.size()}, torch::kInt64).clone()});
    edge_weight = torch::from_blob(ew.data(), {(int64_t)ew.size()}, torch::kFloat32).clone();
}

// ── lagrangian_sld (port MOT POUR MOT de gnn_sd_edit_native.cpp) — topologie-
// agnostique, ne lit que A_g/b_res_g/c_g/lam. Dupliquée (pas partagée) —
// même choix de duplication que les autres variantes de ce dossier. ────────
double lagrangian_sld_cpp(const torch::Tensor& lam, const Vec3d& A_g, const Vec2d& b_res_g,
                           const Vec1d& c_g, int N_z, int n_g, int N, int n_fixed_z1, double epsilon,
                           bool use_surrogate_zeta)
{
    int m = static_cast<int>(A_g[0].size());
    auto lam_2d = lam.view({N_z, n_g}).to(torch::kFloat64).contiguous();
    auto lam_acc = lam_2d.accessor<double, 2>();
    double p_unif = 1.0 / N;

    Vec1d Lambda_sum(n_g, 0.0);
    for (int i = 0; i < N_z; ++i) for (int k = 0; k < n_g; ++k) Lambda_sum[k] += lam_acc[i][k];

    Vec2d cost(N_z, Vec1d(n_g));
    for (int i = 0; i < N_z; ++i)
        for (int k = 0; k < n_g; ++k)
            cost[i][k] = c_g[k] + lam_acc[i][k] / p_unif - Lambda_sum[k];

    std::vector<double> cost_flat(static_cast<size_t>(N_z) * n_g);
    std::vector<double> A_flat(static_cast<size_t>(N_z) * m * n_g);
    std::vector<double> b_flat(static_cast<size_t>(N_z) * m);
    for (int i = 0; i < N_z; ++i) {
        for (int k = 0; k < n_g; ++k) cost_flat[i * n_g + k] = cost[i][k];
        for (int j = 0; j < m; ++j) {
            b_flat[i * m + j] = b_res_g[i][j];
            for (int k = 0; k < n_g; ++k) A_flat[(size_t)i * m * n_g + j * n_g + k] = A_g[i][j][k];
        }
    }
    std::vector<double> x_zeta(static_cast<size_t>(N_z) * n_g, 0.0);
    if (use_surrogate_zeta)
        solve_zeta_batch_surrogate(N_z, m, n_g, cost_flat.data(), A_flat.data(), b_flat.data(), x_zeta.data());
    else
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

    double required_prob = std::max(0.0, (1.0 - epsilon) - n_fixed_z1 * p_unif);
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

} // namespace

// ── Impl ──────────────────────────────────────────────────────────────────────
struct GNNSdEditNativeBatched::Impl {
    int n, m, N;
    double epsilon;
    Vec1d c; Vec3d A; Vec2d b;
    bool use_surrogate_zeta = false;
    // zero_lambda (GNN_ZERO_LAMBDA=1, lu une fois au constructeur) : ABLATION -- le forward GNN est sauté et L(λ) est
    // évalué en λ = 0 par EXACTEMENT le même lagrangian_sld_cpp (même ζ, même parallélisme OpenMP), dans le MÊME
    // moteur batché (même DFS, même bucketing) : seule la valeur de λ diffère de la campagne gnn_batched.
    bool zero_lambda = false;
    torch::Device device{torch::kCPU};
    torch::jit::script::Module module;

    // Cache d'arêtes par n_g : build_edges_sd_edit_cpp ne dépend que de
    // (n_g, N_z) en indices locaux (jamais de l'identité des items), et N_z
    // == N est constant pour la durée de vie de l'objet (cf. build_features_cpp,
    // N_z = N toujours) — donc sans risque de correction, contrairement au
    // cache de features (cf. active_set_counts ci-dessous).
    std::unordered_map<int, std::pair<torch::Tensor, torch::Tensor>> edge_cache;

    // Instrumentation (mesure combien de fois le même ensemble d'items libres
    // (active_x) recur à travers l'arbre) — a motivé le cache ci-dessous
    // (échange du 2026-07-17). Gardée pour visibilité continue.
    std::unordered_map<std::string, int64_t> active_set_counts;
    int64_t active_set_total_calls = 0;

    // Cache RÉEL des quantités active_x-invariantes (cf. ActiveXInvariant) —
    // clé = même active_key que active_set_counts (l'ensemble EXACT des
    // items libres, pas n_g/depth : cf. doc sur ActiveXInvariant, choice()
    // sautant des variables déjà fixées par propagation CP,
    // gecode_bab.hpp:542-545 — donc active_x n'est pas fonction pure de la
    // profondeur, contrairement aux arêtes/edge_cache ci-dessus).
    std::unordered_map<std::string, ActiveXInvariant> active_x_cache;
    int64_t active_x_cache_hits = 0, active_x_cache_misses = 0;
};

GNNSdEditNativeBatched::GNNSdEditNativeBatched(
    const std::string& checkpoint_path, const std::string& device,
    int n, int m, int N, double epsilon,
    const Vec1d& c, const Vec1d& p, const Vec3d& A, const Vec2d& b,
    bool use_surrogate_zeta)
    : impl_(std::make_unique<Impl>())
{
    const double p_unif = 1.0 / static_cast<double>(N);
    for (double pi : p)
        if (std::abs(pi - p_unif) > 1e-9)
            throw std::invalid_argument(
                "GNNSdEditNativeBatched : p non-uniforme — lagrangian_sld suppose des "
                "scénarios équiprobables (p_i = 1/N)");

    if (device == "cuda") {
        if (!torch::cuda::is_available())
            throw std::invalid_argument("GNNSdEditNativeBatched: device=cuda demandé mais CUDA indisponible");
        impl_->device = torch::Device(torch::kCUDA);
    } else if (device == "cpu") {
        impl_->device = torch::Device(torch::kCPU);
    } else {
        throw std::invalid_argument("GNNSdEditNativeBatched: device inconnu (\"" + device + "\"), attendu cpu|cuda");
    }

    impl_->n = n; impl_->m = m; impl_->N = N; impl_->epsilon = epsilon;
    impl_->c = c; impl_->A = A; impl_->b = b;
    impl_->use_surrogate_zeta = use_surrogate_zeta;
    if (const char* e = std::getenv("GNN_ZERO_LAMBDA")) impl_->zero_lambda = (std::atoi(e) == 1);
    if (impl_->zero_lambda) std::cerr << "# GNN_ZERO_LAMBDA=1 : ablation lambda=0 (forward GNN saute)\n";

    std::string scripted_path = ensure_scripted_batched_export(checkpoint_path, "python3");
    impl_->module = torch::jit::load(scripted_path, impl_->device);
    impl_->module.eval();

    // freeze + optimize_for_inference : internalise les attributs (poids)
    // comme constantes et applique les fusions/simplifications standard de
    // TorchScript (élimine le travail mort d'entraînement, fusionne
    // certaines op-chains) — aucun export/compilation externe, donc aucun
    // des pièges rencontrés avec AOTInductor (GRUCell/CUDA, dynamic shapes).
    // Risque : purement une optimisation du graphe déjà chargé, mais peut
    // légèrement réordonner des additions flottantes — à valider (nœuds/
    // objectif) comme les autres changements de ce fichier, cf. échange du
    // 2026-07-18.
    impl_->module = torch::jit::freeze(impl_->module);
    impl_->module = torch::jit::optimize_for_inference(impl_->module);
}

GNNSdEditNativeBatched::~GNNSdEditNativeBatched()
{
    if (impl_ && impl_->active_set_total_calls > 0) {
        const int64_t total    = impl_->active_set_total_calls;
        const int64_t distinct = static_cast<int64_t>(impl_->active_set_counts.size());
        int64_t max_reuse = 0;
        for (const auto& kv : impl_->active_set_counts) max_reuse = std::max(max_reuse, kv.second);
        std::fprintf(stderr,
            "[gnn_sd_edit_native_batched] active_x recurrence : %lld appels, %lld ensembles "
            "distincts (%.1f%% seraient des hits si cache-par-active_x), max reuse=%lld\n",
            static_cast<long long>(total), static_cast<long long>(distinct),
            100.0 * (1.0 - static_cast<double>(distinct) / static_cast<double>(total)),
            static_cast<long long>(max_reuse));
        const int64_t cache_total = impl_->active_x_cache_hits + impl_->active_x_cache_misses;
        std::fprintf(stderr,
            "[gnn_sd_edit_native_batched] active_x_cache : %lld hits / %lld misses (%.1f%% hit rate)\n",
            static_cast<long long>(impl_->active_x_cache_hits),
            static_cast<long long>(impl_->active_x_cache_misses),
            cache_total > 0 ? 100.0 * impl_->active_x_cache_hits / cache_total : 0.0);
    }
}

void GNNSdEditNativeBatched::query_batch(
    const std::vector<std::vector<int8_t>>& x_masks, const std::vector<double>& curr_costs,
    const std::vector<Vec2d>& b_res_list, int depth, std::vector<double>& ub_relative_out,
    GNNBatchCallTiming& timing)
{
    using Clock = std::chrono::steady_clock;
    const int B = static_cast<int>(x_masks.size());
    if (B == 0) { ub_relative_out.clear(); return; }
    if (static_cast<int>(curr_costs.size()) != B)
        throw std::invalid_argument("GNNSdEditNativeBatched::query_batch : x_masks/curr_costs de tailles différentes");
    if (static_cast<int>(b_res_list.size()) != B)
        throw std::invalid_argument("GNNSdEditNativeBatched::query_batch : x_masks/b_res_list de tailles différentes");

    auto t0 = Clock::now();

    std::vector<BuiltFeatures> feats;
    feats.reserve(B);
    for (int bIdx = 0; bIdx < B; ++bIdx) {
        // b_res_list[bIdx] est déjà à jour (maintenu incrémentalement par
        // l'appelant, cf. MKPSpaceGNNBatched::b_res / doc sur
        // GNNBoundBackendBatch::query_batch) — ne PLUS le reconstruire ici
        // depuis x_mask.
        const Vec2d& b_res = b_res_list[bIdx];

        // Clé = ensemble EXACT des items libres (mask==2), indépendante de
        // leur valeur fixée — cf. doc sur Impl::active_x_cache (pourquoi PAS
        // n_g/depth seul). Même clé pour l'instrumentation et le cache réel.
        std::string active_key;
        active_key.reserve(static_cast<size_t>(impl_->n) * 3);
        for (int k = 0; k < impl_->n; ++k)
            if (x_masks[bIdx][k] == 2) { active_key += std::to_string(k); active_key += ','; }
        ++impl_->active_set_counts[active_key];
        ++impl_->active_set_total_calls;

        auto inv_it = impl_->active_x_cache.find(active_key);
        if (inv_it == impl_->active_x_cache.end()) {
            ++impl_->active_x_cache_misses;
            inv_it = impl_->active_x_cache.emplace(
                active_key, compute_active_x_invariant(impl_->m, impl_->N, impl_->c, impl_->A, impl_->b,
                                                        x_masks[bIdx])).first;
        } else {
            ++impl_->active_x_cache_hits;
        }

        feats.push_back(build_features_cpp(impl_->n, impl_->m, impl_->N, inv_it->second, impl_->b,
                                            b_res, curr_costs[bIdx], depth, impl_->epsilon));
    }

    const int n_g = feats[0].n_g;
    const int N_z = feats[0].N_z;
    for (int bIdx = 1; bIdx < B; ++bIdx)
        if (feats[bIdx].n_g != n_g)
            throw std::invalid_argument(
                "GNNSdEditNativeBatched::query_batch : tous les masques du batch doivent "
                "partager le même n_g (violé au membre " + std::to_string(bIdx) +
                ") — contrat imposé par le bucketing par n_g (Étape 1)");

    const int node_dim = feats[0].node_dim;
    const int64_t V = static_cast<int64_t>(N_z) * n_g;

    if (impl_->zero_lambda) {   // ablation λ = 0 : pas de graphe ni de forward, même évaluation de L(λ)
        auto t1z = Clock::now();
        const torch::Tensor lam0 = torch::zeros({V}, torch::kFloat64);
        ub_relative_out.assign(B, 0.0);
        for (int bIdx = 0; bIdx < B; ++bIdx)
            ub_relative_out[bIdx] = lagrangian_sld_cpp(lam0, feats[bIdx].A_g, feats[bIdx].b_res_g, feats[bIdx].c_g, N_z, n_g,
                                                       feats[bIdx].N, feats[bIdx].n_fixed_z1, impl_->epsilon,
                                                       impl_->use_surrogate_zeta);
        auto t3z = Clock::now();
        timing.t_graph_build_s   = std::chrono::duration<double>(t1z - t0).count();   // construction des sous-problèmes
        timing.t_forward_s       = 0.0;
        timing.t_bound_compute_s = std::chrono::duration<double>(t3z - t1z).count();
        return;
    }

    // Arêtes construites UNE SEULE FOIS par n_g distinct (cachées entre appels
    // query_batch, pas seulement à l'intérieur d'un même appel) — sans risque
    // de correction : build_edges_sd_edit_cpp ne dépend que de (n_g, N_z) en
    // indices locaux, jamais de l'identité des items (cf. doc sur Impl::edge_cache).
    torch::Tensor edge_index_1, edge_weight_1;
    auto edge_it = impl_->edge_cache.find(n_g);
    if (edge_it != impl_->edge_cache.end()) {
        edge_index_1  = edge_it->second.first;
        edge_weight_1 = edge_it->second.second;
    } else {
        build_edges_sd_edit_cpp(n_g, N_z, edge_index_1, edge_weight_1);
        impl_->edge_cache.emplace(n_g, std::make_pair(edge_index_1, edge_weight_1));
    }
    const int64_t E = edge_index_1.size(1);

    torch::Tensor offsets = torch::arange(B, torch::kInt64) * V;
    torch::Tensor ei_rep = edge_index_1.unsqueeze(0).expand({B, 2, E}) + offsets.view({B, 1, 1});
    torch::Tensor edge_index_B = ei_rep.permute({1, 0, 2}).reshape({2, B * E}).contiguous();
    torch::Tensor edge_weight_B = edge_weight_1.unsqueeze(0).expand({B, E}).reshape({B * E}).contiguous();

    torch::Tensor X = torch::empty({static_cast<int64_t>(B) * V, node_dim}, torch::kFloat32);
    torch::Tensor ctx = torch::empty({B, 9}, torch::kFloat32);
    auto X_acc = X.accessor<float, 2>();
    auto ctx_acc = ctx.accessor<float, 2>();
    for (int bIdx = 0; bIdx < B; ++bIdx) {
        std::memcpy(X.data_ptr<float>() + static_cast<int64_t>(bIdx) * V * node_dim,
                    feats[bIdx].X_flat.data(), sizeof(float) * V * node_dim);
        for (int c9 = 0; c9 < 9; ++c9) ctx_acc[bIdx][c9] = feats[bIdx].ctx[c9];
    }
    (void)X_acc;

    auto t1 = Clock::now();

    torch::NoGradGuard no_grad;
    torch::Tensor X_dev = X, ei_dev = edge_index_B, ew_dev = edge_weight_B, ctx_dev = ctx;
    if (impl_->device.is_cuda()) {
        X_dev = X_dev.to(impl_->device);
        ei_dev = ei_dev.to(impl_->device);
        ew_dev = ew_dev.to(impl_->device);
        ctx_dev = ctx_dev.to(impl_->device);
    }

    std::vector<torch::jit::IValue> inputs = {X_dev, ei_dev, ew_dev, ctx_dev,
                                               static_cast<int64_t>(n_g), static_cast<int64_t>(B)};
    torch::Tensor h_out = impl_->module.forward(inputs).toTensor();
    if (impl_->device.is_cuda()) h_out = h_out.to(torch::kCPU);
    auto t2 = Clock::now();

    ub_relative_out.assign(B, 0.0);
    torch::Tensor h_out_2d = h_out.view({B, V});
    for (int bIdx = 0; bIdx < B; ++bIdx) {
        torch::Tensor lam_b = h_out_2d[bIdx] * feats[bIdx].c_scale / static_cast<double>(feats[bIdx].N);
        ub_relative_out[bIdx] = lagrangian_sld_cpp(lam_b, feats[bIdx].A_g, feats[bIdx].b_res_g,
                                                     feats[bIdx].c_g, N_z, n_g, feats[bIdx].N,
                                                     feats[bIdx].n_fixed_z1, impl_->epsilon,
                                                     impl_->use_surrogate_zeta);
    }
    auto t3 = Clock::now();

    timing.t_graph_build_s   = std::chrono::duration<double>(t1 - t0).count();
    timing.t_forward_s       = std::chrono::duration<double>(t2 - t1).count();
    timing.t_bound_compute_s = std::chrono::duration<double>(t3 - t2).count();
}
