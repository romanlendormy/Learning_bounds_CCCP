/*
 * gnn_sd_edit_native_batched_zaware.cpp — Implémentation (cf.
 * gnn_sd_edit_native_batched_zaware.hpp). Port ligne-pour-ligne de
 * gnn_sd_edit_native_batched.cpp, seule différence : active_x_invariant/
 * build_features_cpp/le cache d'arêtes sont paramétrés par free_z (au lieu de
 * supposer N_z = N toujours) — cf. header pour la justification mathématique
 * complète (exclusion, pas approximation).
 */

#include "gnn_sd_edit_native_batched_zaware.hpp"
#include "gnn_sd_edit_scriptable_batched_ragged_export.hpp"

#include <torch/script.h>
#include <torch/cuda.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
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

// ── Quantités qui ne dépendent QUE de (active_x, free_z) — extension de
// ActiveXInvariant (gnn_sd_edit_native_batched.cpp) pour paramétrer aussi sur
// les scénarios réellement libres, au lieu de N toujours. Cache clé = active_x
// ⊕ free_z (cf. Impl::active_x_cache ci-dessous) : pour la majorité des nœuds
// (~66.6%, aucun z encore résolu, cf. mesure de fréquence) free_z == [0..N),
// donc la clé retombe EXACTEMENT sur celle d'origine (même hit rate ~99.9%
// pour cette population) ; seuls les nœuds à z partiellement résolu (~33.4%)
// paient un recalcul (essentiellement toujours un miss, free_z y étant
// quasi-unique par nœud) — acceptable, la construction de features ne pèse
// que ~10% du temps total (cf. mesure du 2026-08-19).
struct ActiveXInvariant {
    std::vector<int> active_x;
    std::vector<int> free_z;
    Vec1d c_g;
    Vec3d A_g;          // (N_z, m, n_g)
    double c_g_sum;
    Vec3d f_A;          // (N_z, n_g, m) = A_g / b (global, jamais b_res)
    Vec3d dantzig_rank;  // (N_z, m, n_g)
    Vec2d A_mean_scen, A_std_scen;  // (n_g, m) — calculées sur free_z SEULEMENT
};

ActiveXInvariant compute_active_x_invariant(
    int m, int N, const Vec1d& c, const Vec3d& A, const Vec2d& b,
    const std::vector<int8_t>& x_mask, const std::vector<int>& free_z)
{
    const int n = static_cast<int>(x_mask.size());
    const int N_z = static_cast<int>(free_z.size());
    ActiveXInvariant inv;
    inv.free_z = free_z;

    for (int k = 0; k < n; ++k) if (x_mask[k] == 2) inv.active_x.push_back(k);
    const int n_g = static_cast<int>(inv.active_x.size());

    inv.c_g.resize(n_g);
    for (int kf = 0; kf < n_g; ++kf) inv.c_g[kf] = c[inv.active_x[kf]];

    inv.A_g.assign(N_z, Vec2d(m, Vec1d(n_g)));
    for (int iz = 0; iz < N_z; ++iz) {
        int i = free_z[iz];
        for (int j = 0; j < m; ++j)
            for (int kf = 0; kf < n_g; ++kf)
                inv.A_g[iz][j][kf] = A[i][j][inv.active_x[kf]];
    }

    inv.c_g_sum = 0.0; for (double v : inv.c_g) inv.c_g_sum += v;

    inv.f_A.assign(N_z, Vec2d(n_g, Vec1d(m)));
    for (int iz = 0; iz < N_z; ++iz) {
        int i = free_z[iz];
        for (int kf = 0; kf < n_g; ++kf)
            for (int j = 0; j < m; ++j)
                inv.f_A[iz][kf][j] = inv.A_g[iz][j][kf] / std::max(b[i][j], 1e-9);
    }

    inv.A_mean_scen.assign(n_g, Vec1d(m, 0.0));
    inv.A_std_scen.assign(n_g, Vec1d(m, 0.0));
    for (int kf = 0; kf < n_g; ++kf)
        for (int j = 0; j < m; ++j) {
            double sum = 0.0; for (int iz = 0; iz < N_z; ++iz) sum += inv.f_A[iz][kf][j];
            double mean = sum / N_z, var = 0.0;
            for (int iz = 0; iz < N_z; ++iz) { double d = inv.f_A[iz][kf][j] - mean; var += d * d; }
            inv.A_mean_scen[kf][j] = mean; inv.A_std_scen[kf][j] = std::sqrt(var / N_z);
        }

    inv.dantzig_rank.assign(N_z, Vec2d(m, Vec1d(n_g)));
    double rank_norm = std::max(n_g - 1, 1);
    for (int iz = 0; iz < N_z; ++iz)
        for (int j = 0; j < m; ++j) {
            std::vector<int> order(n_g); std::iota(order.begin(), order.end(), 0);
            std::vector<double> eff(n_g);
            for (int kf = 0; kf < n_g; ++kf) eff[kf] = inv.c_g[kf] / std::max(inv.A_g[iz][j][kf], 1e-9);
            std::sort(order.begin(), order.end(), [&](int a, int b2) { return eff[a] > eff[b2]; });
            for (int r = 0; r < n_g; ++r) inv.dantzig_rank[iz][j][order[r]] = r / rank_norm;
        }

    return inv;
}

struct BuiltFeatures {
    std::vector<float> X_flat;   // (N_z*n_g, node_dim), row-major
    std::vector<float> ctx;      // (9,)
    Vec3d A_g; Vec2d b_res_g; Vec1d c_g;
    int n_g, N_z, N, n_fixed_z1, node_dim;
    double c_scale;
};

// b_res_full/b_full : (N, m) COMPLETS (pas encore restreints à free_z) — le
// slicing par free_z se fait ICI (b_res/b varient à chaque appel, contrairement
// à inv qui est mise en cache) : b_res_z[iz] = b_res_full[inv.free_z[iz]].
BuiltFeatures build_features_cpp(int n, int m, int N, const ActiveXInvariant& inv,
                                  const Vec2d& b_full, const Vec2d& b_res_full,
                                  double curr_cost, int depth, double epsilon, int n_fixed_z1)
{
    const int n_g = static_cast<int>(inv.active_x.size());
    const int N_z = static_cast<int>(inv.free_z.size());

    Vec2d b_res_z(N_z, Vec1d(m)), b_z(N_z, Vec1d(m));
    for (int iz = 0; iz < N_z; ++iz) {
        int i = inv.free_z[iz];
        for (int j = 0; j < m; ++j) { b_res_z[iz][j] = b_res_full[i][j]; b_z[iz][j] = b_full[i][j]; }
    }

    const Vec1d& c_g = inv.c_g;
    const Vec3d& A_g = inv.A_g;
    const Vec3d& f_A = inv.f_A;
    const Vec3d& dantzig_rank = inv.dantzig_rank;
    const Vec2d& A_mean_scen = inv.A_mean_scen;
    const Vec2d& A_std_scen = inv.A_std_scen;

    double c_scale = std::max(curr_cost + inv.c_g_sum, 1.0);

    Vec2d cap_frac(N_z, Vec1d(m));
    for (int iz = 0; iz < N_z; ++iz)
        for (int j = 0; j < m; ++j) cap_frac[iz][j] = b_res_z[iz][j] / std::max(b_z[iz][j], 1e-9);

    Vec1d scen_min(N_z), scen_mean(N_z), scen_std(N_z), scen_sat(N_z);
    for (int iz = 0; iz < N_z; ++iz) {
        double mn = cap_frac[iz][0], sum = 0.0;
        for (int j = 0; j < m; ++j) { mn = std::min(mn, cap_frac[iz][j]); sum += cap_frac[iz][j]; }
        double mean = sum / m, var = 0.0; int sat = 0;
        for (int j = 0; j < m; ++j) {
            double d = cap_frac[iz][j] - mean; var += d * d;
            if (cap_frac[iz][j] < 0.1) ++sat;
        }
        scen_min[iz] = mn; scen_mean[iz] = mean; scen_std[iz] = std::sqrt(var / m);
        scen_sat[iz] = static_cast<double>(sat) / m;
    }

    int node_dim = 7 * m + 5;
    std::vector<float> X(static_cast<size_t>(N_z) * n_g * node_dim);
    double tightest = 1e300;
    for (int iz = 0; iz < N_z; ++iz) {
        for (int kf = 0; kf < n_g; ++kf) {
            size_t row = static_cast<size_t>(iz) * n_g + kf;
            float* out = &X[row * node_dim];
            int off = 0;
            double f_c = c_g[kf] / c_scale;
            out[off++] = static_cast<float>(f_c);
            for (int j = 0; j < m; ++j) out[off++] = static_cast<float>(f_A[iz][kf][j]);
            for (int j = 0; j < m; ++j)
                out[off++] = static_cast<float>(std::min(A_g[iz][j][kf] / std::max(b_res_z[iz][j], 1e-9), 10.0));
            for (int j = 0; j < m; ++j)
                out[off++] = static_cast<float>(std::log1p(f_c / std::max(f_A[iz][kf][j], 1e-9)));
            for (int j = 0; j < m; ++j) out[off++] = static_cast<float>(dantzig_rank[iz][j][kf]);
            for (int j = 0; j < m; ++j) out[off++] = static_cast<float>(cap_frac[iz][j]);
            out[off++] = static_cast<float>(scen_min[iz]);
            out[off++] = static_cast<float>(scen_mean[iz]);
            for (int j = 0; j < m; ++j) out[off++] = static_cast<float>(A_mean_scen[kf][j]);
            for (int j = 0; j < m; ++j) out[off++] = static_cast<float>(A_std_scen[kf][j]);
            out[off++] = static_cast<float>(scen_std[iz]);
            out[off++] = static_cast<float>(scen_sat[iz]);
        }
        for (int j = 0; j < m; ++j) tightest = std::min(tightest, cap_frac[iz][j]);
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

// lagrangian_v1ld_cpp — INCHANGÉ (déjà paramétré par N_z/n_fixed_z1, cf. header) :
// copie exacte de gnn_sd_edit_native_batched.cpp, aucune modification requise.
double lagrangian_v1ld_cpp(const torch::Tensor& lam, const Vec3d& A_g, const Vec2d& b_res_g,
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

struct GNNSdEditNativeBatchedZAware::Impl {
    int n, m, N;
    double epsilon;
    Vec1d c; Vec3d A; Vec2d b;
    bool use_surrogate_zeta = false;
    torch::Device device{torch::kCPU};
    torch::jit::script::Module module;

    // Cache d'arêtes clé = (n_g, N_z) — contrairement à la variante d'origine
    // (clé = n_g seul, N_z==N constant), ici N_z varie selon les z déjà
    // résolus, donc les arêtes doivent être recalculées par (n_g, N_z) distinct.
    std::unordered_map<int64_t, std::pair<torch::Tensor, torch::Tensor>> edge_cache;
    static int64_t edge_key(int n_g, int N_z) { return (int64_t)n_g << 32 | (uint32_t)N_z; }

    std::unordered_map<std::string, int64_t> active_set_counts;
    int64_t active_set_total_calls = 0;

    // Cache ActiveXInvariant, clé = active_x ⊕ free_z (cf. doc de header :
    // retombe sur l'ancienne clé x-seul quand free_z == [0..N), préservant le
    // hit rate ~99.9% pour cette population majoritaire).
    std::unordered_map<std::string, ActiveXInvariant> active_x_cache;
    int64_t active_x_cache_hits = 0, active_x_cache_misses = 0;
};

GNNSdEditNativeBatchedZAware::GNNSdEditNativeBatchedZAware(
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
                "GNNSdEditNativeBatchedZAware : p non-uniforme — lagrangian_v1ld suppose des "
                "scénarios équiprobables (p_i = 1/N)");

    if (device == "cuda") {
        if (!torch::cuda::is_available())
            throw std::invalid_argument("GNNSdEditNativeBatchedZAware: device=cuda demandé mais CUDA indisponible");
        impl_->device = torch::Device(torch::kCUDA);
    } else if (device == "cpu") {
        impl_->device = torch::Device(torch::kCPU);
    } else {
        throw std::invalid_argument("GNNSdEditNativeBatchedZAware: device inconnu (\"" + device + "\"), attendu cpu|cuda");
    }

    impl_->n = n; impl_->m = m; impl_->N = N; impl_->epsilon = epsilon;
    impl_->c = c; impl_->A = A; impl_->b = b;
    impl_->use_surrogate_zeta = use_surrogate_zeta;

    // Artefact RAGGED (pooling scatter/index_add_, cf. doc header) — MÊMES
    // poids que le checkpoint de production, aucun réentraînement, juste un
    // second export TorchScript distinct (.ragged_scripted_batched.pt).
    std::string scripted_path = ensure_scripted_batched_ragged_export(checkpoint_path, "python3");
    impl_->module = torch::jit::load(scripted_path, impl_->device);
    impl_->module.eval();
    impl_->module = torch::jit::freeze(impl_->module);
    impl_->module = torch::jit::optimize_for_inference(impl_->module);
}

GNNSdEditNativeBatchedZAware::~GNNSdEditNativeBatchedZAware()
{
    if (impl_ && impl_->active_set_total_calls > 0) {
        const int64_t cache_total = impl_->active_x_cache_hits + impl_->active_x_cache_misses;
        std::fprintf(stderr,
            "[gnn_sd_edit_native_batched_zaware] active_x_cache : %lld hits / %lld misses (%.1f%% hit rate)\n",
            static_cast<long long>(impl_->active_x_cache_hits),
            static_cast<long long>(impl_->active_x_cache_misses),
            cache_total > 0 ? 100.0 * impl_->active_x_cache_hits / cache_total : 0.0);
    }
}

void GNNSdEditNativeBatchedZAware::query_batch(
    const std::vector<std::vector<int8_t>>& x_masks,
    const std::vector<std::vector<int8_t>>& z_masks,
    const std::vector<double>& curr_costs,
    const std::vector<Vec2d>& b_res_list, int depth, std::vector<double>& ub_relative_out,
    GNNBatchCallTiming& timing)
{
    using Clock = std::chrono::steady_clock;
    const int B = static_cast<int>(x_masks.size());
    if (B == 0) { ub_relative_out.clear(); return; }
    if (static_cast<int>(curr_costs.size()) != B || static_cast<int>(b_res_list.size()) != B ||
        static_cast<int>(z_masks.size()) != B)
        throw std::invalid_argument("GNNSdEditNativeBatchedZAware::query_batch : tailles de batch incohérentes");

    auto t0 = Clock::now();

    std::vector<BuiltFeatures> feats;
    feats.reserve(B);
    for (int bIdx = 0; bIdx < B; ++bIdx) {
        const Vec2d& b_res = b_res_list[bIdx];

        std::vector<int> free_z;
        free_z.reserve(impl_->N);
        int n_fixed_z1 = 0;
        for (int i = 0; i < impl_->N; ++i) {
            if (z_masks[bIdx][i] == 2) free_z.push_back(i);
            else if (z_masks[bIdx][i] == 1) ++n_fixed_z1;
        }

        std::string active_key;
        active_key.reserve(static_cast<size_t>(impl_->n) * 3 + static_cast<size_t>(free_z.size()) * 4);
        for (int k = 0; k < impl_->n; ++k)
            if (x_masks[bIdx][k] == 2) { active_key += std::to_string(k); active_key += ','; }
        active_key += '|';
        for (int i : free_z) { active_key += std::to_string(i); active_key += ','; }
        ++impl_->active_set_counts[active_key];
        ++impl_->active_set_total_calls;

        auto inv_it = impl_->active_x_cache.find(active_key);
        if (inv_it == impl_->active_x_cache.end()) {
            ++impl_->active_x_cache_misses;
            inv_it = impl_->active_x_cache.emplace(
                active_key, compute_active_x_invariant(impl_->m, impl_->N, impl_->c, impl_->A, impl_->b,
                                                        x_masks[bIdx], free_z)).first;
        } else {
            ++impl_->active_x_cache_hits;
        }

        feats.push_back(build_features_cpp(impl_->n, impl_->m, impl_->N, inv_it->second, impl_->b,
                                            b_res, curr_costs[bIdx], depth, impl_->epsilon, n_fixed_z1));
    }

    // Seul n_g doit être uniforme (contrat imposé par le driver, bucketing par
    // n_g seul) — N_z PEUT varier membre à membre : batch RAGGED, pas de
    // padding (cf. doc header, révision 2026-08-19).
    const int n_g = feats[0].n_g;
    for (int bIdx = 1; bIdx < B; ++bIdx)
        if (feats[bIdx].n_g != n_g)
            throw std::invalid_argument(
                "GNNSdEditNativeBatchedZAware::query_batch : tous les masques du batch doivent "
                "partager le même n_g (violé au membre " + std::to_string(bIdx) + ")");

    const int node_dim = feats[0].node_dim;

    // Décalages cumulés (nœuds) et (créneaux scénario) par membre — offsets[b]
    // = nombre total de nœuds des membres 0..b-1 ; scen_offsets[b] = nombre
    // total de créneaux scénario des membres 0..b-1 (S_total = leur somme).
    std::vector<int64_t> Vs(B), offsets(B), scen_offsets(B);
    int64_t total_V = 0, S_total = 0;
    for (int bIdx = 0; bIdx < B; ++bIdx) {
        Vs[bIdx] = static_cast<int64_t>(feats[bIdx].N_z) * n_g;
        offsets[bIdx] = total_V; total_V += Vs[bIdx];
        scen_offsets[bIdx] = S_total; S_total += feats[bIdx].N_z;
    }

    // Arêtes construites PAR MEMBRE (dépendent de (n_g, N_z_b), mises en cache
    // par cette paire — cf. Impl::edge_cache — donc réutilisées à l'identique
    // pour les ~66% de membres à N_z=N inchangé, seuls les membres à N_z
    // distinct paient un calcul frais), puis concaténées avec un décalage
    // propre à chaque membre (offsets[b], PAS b*V comme la version dense —
    // c'est tout : GatedGraphConvScriptable (message passing) traite déjà un
    // batch bloc-diagonal à tailles hétérogènes comme B graphes séparés, cf.
    // doc de gnn_sd_edit_scriptable_batched.py).
    std::vector<torch::Tensor> ei_parts, ew_parts;
    ei_parts.reserve(B); ew_parts.reserve(B);
    for (int bIdx = 0; bIdx < B; ++bIdx) {
        const int N_z_b = feats[bIdx].N_z;
        torch::Tensor edge_index_1, edge_weight_1;
        const int64_t ekey = Impl::edge_key(n_g, N_z_b);
        auto edge_it = impl_->edge_cache.find(ekey);
        if (edge_it != impl_->edge_cache.end()) {
            edge_index_1  = edge_it->second.first;
            edge_weight_1 = edge_it->second.second;
        } else {
            build_edges_sd_edit_cpp(n_g, N_z_b, edge_index_1, edge_weight_1);
            impl_->edge_cache.emplace(ekey, std::make_pair(edge_index_1, edge_weight_1));
        }
        ei_parts.push_back(edge_index_1 + offsets[bIdx]);
        ew_parts.push_back(edge_weight_1);
    }
    torch::Tensor edge_index_B  = torch::cat(ei_parts, 1);
    torch::Tensor edge_weight_B = torch::cat(ew_parts, 0);

    torch::Tensor X   = torch::empty({total_V, node_dim}, torch::kFloat32);
    torch::Tensor ctx = torch::empty({B, 9}, torch::kFloat32);
    auto ctx_acc = ctx.accessor<float, 2>();
    torch::Tensor graph_id   = torch::empty({total_V}, torch::kInt64);
    torch::Tensor item_group = torch::empty({total_V}, torch::kInt64);
    torch::Tensor scen_group = torch::empty({total_V}, torch::kInt64);
    auto gid_acc = graph_id.accessor<int64_t, 1>();
    auto ig_acc  = item_group.accessor<int64_t, 1>();
    auto sg_acc  = scen_group.accessor<int64_t, 1>();
    for (int bIdx = 0; bIdx < B; ++bIdx) {
        std::memcpy(X.data_ptr<float>() + offsets[bIdx] * node_dim,
                    feats[bIdx].X_flat.data(), sizeof(float) * Vs[bIdx] * node_dim);
        for (int c9 = 0; c9 < 9; ++c9) ctx_acc[bIdx][c9] = feats[bIdx].ctx[c9];

        const int N_z_b = feats[bIdx].N_z;
        for (int i = 0; i < N_z_b; ++i)
            for (int k = 0; k < n_g; ++k) {
                int64_t g = offsets[bIdx] + static_cast<int64_t>(i) * n_g + k;
                gid_acc[g] = bIdx;
                ig_acc[g]  = static_cast<int64_t>(bIdx) * n_g + k;
                sg_acc[g]  = scen_offsets[bIdx] + i;
            }
    }

    auto t1 = Clock::now();

    torch::NoGradGuard no_grad;
    torch::Tensor X_dev = X, ei_dev = edge_index_B, ew_dev = edge_weight_B, ctx_dev = ctx;
    torch::Tensor gid_dev = graph_id, ig_dev = item_group, sg_dev = scen_group;
    if (impl_->device.is_cuda()) {
        X_dev = X_dev.to(impl_->device);
        ei_dev = ei_dev.to(impl_->device);
        ew_dev = ew_dev.to(impl_->device);
        ctx_dev = ctx_dev.to(impl_->device);
        gid_dev = gid_dev.to(impl_->device);
        ig_dev = ig_dev.to(impl_->device);
        sg_dev = sg_dev.to(impl_->device);
    }

    std::vector<torch::jit::IValue> inputs = {
        X_dev, ei_dev, ew_dev, ctx_dev, gid_dev, ig_dev, sg_dev,
        static_cast<int64_t>(n_g), static_cast<int64_t>(B), S_total};
    torch::Tensor h_out = impl_->module.forward(inputs).toTensor();
    if (impl_->device.is_cuda()) h_out = h_out.to(torch::kCPU);
    auto t2 = Clock::now();

    ub_relative_out.assign(B, 0.0);
    for (int bIdx = 0; bIdx < B; ++bIdx) {
        torch::Tensor lam_b = h_out.slice(0, offsets[bIdx], offsets[bIdx] + Vs[bIdx])
                                  * feats[bIdx].c_scale / static_cast<double>(feats[bIdx].N);
        ub_relative_out[bIdx] = lagrangian_v1ld_cpp(lam_b, feats[bIdx].A_g, feats[bIdx].b_res_g,
                                                     feats[bIdx].c_g, feats[bIdx].N_z, n_g, feats[bIdx].N,
                                                     feats[bIdx].n_fixed_z1, impl_->epsilon,
                                                     impl_->use_surrogate_zeta);
    }
    auto t3 = Clock::now();

    timing.t_graph_build_s   = std::chrono::duration<double>(t1 - t0).count();
    timing.t_forward_s       = std::chrono::duration<double>(t2 - t1).count();
    timing.t_bound_compute_s = std::chrono::duration<double>(t3 - t2).count();
}
