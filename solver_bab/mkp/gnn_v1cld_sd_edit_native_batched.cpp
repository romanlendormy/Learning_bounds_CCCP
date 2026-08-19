/*
 * gnn_v1cld_sd_edit_native_batched.cpp — Implémentation (cf.
 * gnn_v1cld_sd_edit_native_batched.hpp). Port MOT POUR MOT (mêmes formules,
 * même ordre d'opérations autant que possible) de
 * gnn_sd_edit/train_v1CLD_sd_edit.py::build_graph/GNN.forward/lagrangian_v1cld,
 * en suivant la structure de gnn_sd_edit_native_batched.cpp (V1LD) — cf.
 * docstring de ce dernier pour la logique générale de séparation
 * features/arêtes et le cache par active_x.
 *
 * Différences avec V1LD (au-delà de la 3e dimension j) :
 *   - node_dim = 6 fixe (pas 7*m+5) : A_g brut EST une feature de nœud
 *     directement (pas aplati/agrégé), donc n'a PAS besoin d'être recalculé
 *     depuis A_flat comme les colonnes f_A de V1LD — juste indexé.
 *   - 3 types d'arêtes (item clique + contrainte étoile@j=0 + scénario
 *     étoile@i=0) au lieu de 2 (item clique + scénario étoile) — cf.
 *     build_graph, lignes 264-310 de train_v1CLD_sd_edit.py.
 *   - lagrangian_v1cld centre Λ_sum PAR SCÉNARIO (Lambda_sum_i, pas une
 *     somme globale) — cf. PREUVE_v1CLD_centrage_par_scenario.md.
 *   - ζ_{i,j} résolu par solve_zeta_batch avec N_z*m sous-problèmes
 *     indépendants (au lieu de N_z) — même fonction C++ que V1LD, seule la
 *     taille du batch passé change (aucune modification de bnb_solver.cpp).
 */

#include "gnn_v1cld_sd_edit_native_batched.hpp"
#include "gnn_v1cld_sd_edit_scriptable_batched_export.hpp"

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

// ── Quantités active_x-invariantes (ne dépendent QUE de l'ensemble des items
// libres, jamais de b_res/curr_cost) — cf. doc ActiveXInvariant de
// gnn_sd_edit_native_batched.cpp pour la justification du cache et
// l'avertissement sur la clé (ensemble EXACT, pas n_g/depth seul). ─────────
struct ActiveXInvariant {
    std::vector<int> active_x;
    Vec1d c_g;               // (n_g,)
    Vec3d A_g;                // (N_z, m, n_g) — poids BRUTS, feature de nœud directe (pas de division ici)
    double c_g_sum;
    Vec3d dantzig_rank;       // (N_z, m, n_g) ∈ [0,1] — ne dépend que de c_g/A_g, jamais de b_res/curr_cost
};

ActiveXInvariant compute_active_x_invariant(
    int m, int N, const Vec1d& c, const Vec3d& A, const std::vector<int8_t>& x_mask)
{
    const int n = static_cast<int>(x_mask.size());
    const int N_z = N;   // cf. avertissement de header : z toujours entièrement libre ici
    ActiveXInvariant inv;

    for (int k = 0; k < n; ++k) if (x_mask[k] == 2) inv.active_x.push_back(k);
    const int n_g = static_cast<int>(inv.active_x.size());

    inv.c_g.resize(n_g);
    for (int kf = 0; kf < n_g; ++kf) inv.c_g[kf] = c[inv.active_x[kf]];
    inv.c_g_sum = 0.0; for (double v : inv.c_g) inv.c_g_sum += v;

    inv.A_g.assign(N_z, Vec2d(m, Vec1d(n_g)));
    for (int i = 0; i < N_z; ++i)
        for (int j = 0; j < m; ++j)
            for (int kf = 0; kf < n_g; ++kf)
                inv.A_g[i][j][kf] = A[i][j][inv.active_x[kf]];

    // Rang Dantzig normalisé : position de k dans le classement décroissant
    // d'efficacité c_k/A_ijk au sein du couple (i,j) — cf. build_graph.py:248-253.
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

// ── Features seules (sans arêtes) — port de build_graph (lignes 214-259 de
// train_v1CLD_sd_edit.py). node_dim = 6 fixe (cf. docstring de fichier). ────
struct BuiltFeatures {
    std::vector<float> X_flat;   // (N_z*m*n_g, 6), row-major, ordre (i,j,k) avec k le plus rapide
    std::vector<float> ctx;      // (9,)
    Vec3d A_g; Vec2d b_res_g; Vec1d c_g;
    int n_g, m, N_z, N, n_fixed_z1, node_dim;
    double c_scale;
};

BuiltFeatures build_features_cpp(int n, int m, int N, const ActiveXInvariant& inv, const Vec2d& b,
                                  const Vec2d& b_res, double curr_cost, int depth, double epsilon)
{
    const int n_g = static_cast<int>(inv.active_x.size());
    const int N_z = N, n_fixed_z1 = 0;   // cf. avertissement de header

    const Vec1d& c_g = inv.c_g;
    const Vec3d& A_g = inv.A_g;
    const Vec3d& dantzig_rank = inv.dantzig_rank;

    double c_scale = std::max(curr_cost + inv.c_g_sum, 1.0);

    // f_cap[i][j] = b_res[i][j] / b[i][j] — ne dépend pas de k, broadcast
    Vec2d f_cap(N_z, Vec1d(m));
    double tightest = 1e300;
    for (int i = 0; i < N_z; ++i)
        for (int j = 0; j < m; ++j) {
            f_cap[i][j] = b_res[i][j] / std::max(b[i][j], 1e-9);
            tightest = std::min(tightest, f_cap[i][j]);
        }

    const int node_dim = 6;
    std::vector<float> X(static_cast<size_t>(N_z) * m * n_g * node_dim);
    for (int i = 0; i < N_z; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int kf = 0; kf < n_g; ++kf) {
                size_t row = (static_cast<size_t>(i) * m + j) * n_g + kf;
                float* out = &X[row * node_dim];
                double f_c = c_g[kf] / c_scale;
                double f_A = A_g[i][j][kf];
                double f_frac = std::min(f_A / std::max(b_res[i][j], 1e-9), 10.0);
                double f_eff  = std::log1p(f_c / std::max(f_A, 1e-9));
                out[0] = static_cast<float>(f_c);
                out[1] = static_cast<float>(f_A);
                out[2] = static_cast<float>(f_frac);
                out[3] = static_cast<float>(f_eff);
                out[4] = static_cast<float>(dantzig_rank[i][j][kf]);
                out[5] = static_cast<float>(f_cap[i][j]);
            }
        }
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
    f.A_g = A_g; f.b_res_g = b_res; f.c_g = c_g;
    f.n_g = n_g; f.m = m; f.N_z = N_z; f.N = N; f.n_fixed_z1 = n_fixed_z1; f.node_dim = node_dim;
    f.c_scale = c_scale;
    return f;
}

// ── Arêtes V1CLD (item clique + contrainte étoile@j=0 + scénario étoile@i=0)
// — port de build_graph, lignes 264-310. Ne dépend que de (n_g, m, N_z) —
// appelée une seule fois par n_g distinct (cache Impl::edge_cache), comme V1LD. ─
void build_edges_v1cld_cpp(int n_g, int m, int N_z, torch::Tensor& edge_index, torch::Tensor& edge_weight)
{
    std::vector<int64_t> src, dst; std::vector<float> ew;

    // 1. Item varie (i,j fixés) : clique complète par (i,j), poids 1/n_g.
    for (int i = 0; i < N_z; ++i) {
        for (int j = 0; j < m; ++j) {
            int64_t base = (static_cast<int64_t>(i) * m + j) * n_g;
            for (int k1 = 0; k1 < n_g; ++k1)
                for (int k2 = k1 + 1; k2 < n_g; ++k2) {
                    int64_t a = base + k1, b2 = base + k2;
                    src.push_back(a); dst.push_back(b2); ew.push_back(1.0f / n_g);
                    src.push_back(b2); dst.push_back(a); ew.push_back(1.0f / n_g);
                }
        }
    }

    // 2. Contrainte varie (i,k fixés) : étoile ancrée à j=0, poids 1.
    if (m > 1) {
        for (int i = 0; i < N_z; ++i) {
            for (int k = 0; k < n_g; ++k) {
                int64_t anchor = static_cast<int64_t>(i) * m * n_g + k;   // j=0
                for (int j = 1; j < m; ++j) {
                    int64_t spoke = anchor + static_cast<int64_t>(j) * n_g;
                    src.push_back(anchor); dst.push_back(spoke); ew.push_back(1.0f);
                    src.push_back(spoke); dst.push_back(anchor); ew.push_back(1.0f);
                }
            }
        }
    }

    // 3. Scénario varie (j,k fixés) : étoile ancrée à i=0, poids 1.
    if (N_z > 1) {
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < n_g; ++k) {
                int64_t anchor = static_cast<int64_t>(j) * n_g + k;   // i=0
                for (int i = 1; i < N_z; ++i) {
                    int64_t spoke = anchor + static_cast<int64_t>(i) * m * n_g;
                    src.push_back(anchor); dst.push_back(spoke); ew.push_back(1.0f);
                    src.push_back(spoke); dst.push_back(anchor); ew.push_back(1.0f);
                }
            }
        }
    }

    edge_index = torch::stack({
        torch::from_blob(src.data(), {(int64_t)src.size()}, torch::kInt64).clone(),
        torch::from_blob(dst.data(), {(int64_t)dst.size()}, torch::kInt64).clone()});
    edge_weight = torch::from_blob(ew.data(), {(int64_t)ew.size()}, torch::kFloat32).clone();
}

// ── lagrangian_v1cld (port de train_v1CLD_sd_edit.py:720-804) — topologie-
// agnostique, ne lit que A_g/b_res_g/c_g/lam. ───────────────────────────────
double lagrangian_v1cld_cpp(const torch::Tensor& lam, const Vec3d& A_g, const Vec2d& b_res_g,
                             const Vec1d& c_g, int N_z, int m, int n_g, int N, int n_fixed_z1,
                             double epsilon, bool use_surrogate_zeta)
{
    auto lam_3d = lam.view({N_z, m, n_g}).to(torch::kFloat64).contiguous();
    auto lam_acc = lam_3d.accessor<double, 3>();
    double p_unif = 1.0 / N;
    double inv_m  = 1.0 / m;
    double p_ij   = p_unif * inv_m;

    // Λ_sum_i[i][k] = Σ_j λ[i,j,k] — PAR SCÉNARIO (PAS une somme globale sur
    // i,j — cf. PREUVE_v1CLD_centrage_par_scenario.md, docstring de
    // lagrangian_v1cld en Python).
    Vec2d Lambda_sum_i(N_z, Vec1d(n_g, 0.0));
    for (int i = 0; i < N_z; ++i)
        for (int j = 0; j < m; ++j)
            for (int k = 0; k < n_g; ++k)
                Lambda_sum_i[i][k] += lam_acc[i][j][k];

    // cost[i][j][k] = c_g[k] + lam[i,j,k]/p_ij − Λ_sum_i[i][k]/p_unif
    Vec3d cost(N_z, Vec2d(m, Vec1d(n_g)));
    for (int i = 0; i < N_z; ++i)
        for (int j = 0; j < m; ++j)
            for (int k = 0; k < n_g; ++k)
                cost[i][j][k] = c_g[k] + lam_acc[i][j][k] / p_ij - Lambda_sum_i[i][k] / p_unif;

    // ζ_{i,j} : N_z*m sous-problèmes indépendants, n_g items — MÊME fonction
    // C++ batchée que V1LD (solve_zeta_batch), juste un batch de taille N_z*m
    // au lieu de N_z (aucune modification de bnb_solver.cpp).
    const int NM = N_z * m;
    std::vector<double> cost_flat(static_cast<size_t>(NM) * n_g);
    std::vector<double> A_flat(static_cast<size_t>(NM) * n_g);   // 1 "contrainte" par sous-problème
    std::vector<double> b_flat(static_cast<size_t>(NM));
    for (int i = 0; i < N_z; ++i) {
        for (int j = 0; j < m; ++j) {
            int ij = i * m + j;
            b_flat[ij] = b_res_g[i][j];
            for (int k = 0; k < n_g; ++k) {
                cost_flat[static_cast<size_t>(ij) * n_g + k] = cost[i][j][k];
                A_flat[static_cast<size_t>(ij) * n_g + k]    = A_g[i][j][k];
            }
        }
    }
    std::vector<double> x_zeta(static_cast<size_t>(NM) * n_g, 0.0);
    // solve_zeta_batch(N_z, m, n_g, ...) attend un batch de N_z*m sous-problèmes
    // À UNE SEULE CONTRAINTE : on l'appelle avec "N_z"=NM et "m"=1 pour que sa
    // boucle interne (N_z*m appels à un knapsack 1D, cf. bnb_solver.cpp) traite
    // exactement NM sous-problèmes 1D — signature inchangée, juste réinterprétée.
    if (use_surrogate_zeta)
        solve_zeta_batch_surrogate(NM, 1, n_g, cost_flat.data(), A_flat.data(), b_flat.data(), x_zeta.data());
    else
        solve_zeta_batch(NM, 1, n_g, cost_flat.data(), A_flat.data(), b_flat.data(), x_zeta.data());

    // θ_{i,j}, ζ_{i,j}, Δ_i = mean_j max(0, θ-ζ)
    Vec2d theta(N_z, Vec1d(m)), zeta(N_z, Vec1d(m));
    Vec1d Delta(N_z, 0.0);
    for (int i = 0; i < N_z; ++i) {
        double delta_sum = 0.0;
        for (int j = 0; j < m; ++j) {
            int ij = i * m + j;
            double th = 0.0, ze = 0.0;
            for (int k = 0; k < n_g; ++k) {
                if (cost[i][j][k] > 0) th += cost[i][j][k];
                ze += cost[i][j][k] * x_zeta[static_cast<size_t>(ij) * n_g + k];
            }
            theta[i][j] = th; zeta[i][j] = ze;
            delta_sum += std::max(0.0, th - ze);
        }
        Delta[i] = delta_sum / m;
    }

    // Sélection gloutonne y* (scénarios "satisfaits"), triée par Δ croissant.
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

    // L1_theta = p_ij * Σ_{i,j} theta_lam[i,j] ; val_lam = Σ_{i,j} p_unif*inv_m*y*[i]*delta_lam[i,j]
    // (theta_lam/delta_lam = mêmes theta/zeta que ci-dessus, cost déjà en jeu de λ courant)
    double L1_theta = 0.0, val_lam = 0.0;
    for (int i = 0; i < N_z; ++i) {
        for (int j = 0; j < m; ++j) {
            L1_theta += p_ij * theta[i][j];
            val_lam  += p_unif * inv_m * y_star[i] * (theta[i][j] - zeta[i][j]);
        }
    }
    return L1_theta - val_lam;
}

} // namespace

// ── Impl ──────────────────────────────────────────────────────────────────────
struct GNNV1CLDSdEditNativeBatched::Impl {
    int n, m, N;
    double epsilon;
    Vec1d c; Vec3d A; Vec2d b;
    bool use_surrogate_zeta = false;
    torch::Device device{torch::kCPU};
    torch::jit::script::Module module;

    // Cache d'arêtes par n_g (m et N_z fixes pour la durée de vie de l'objet) —
    // cf. doc Impl::edge_cache de gnn_sd_edit_native_batched.cpp.
    std::unordered_map<int, std::pair<torch::Tensor, torch::Tensor>> edge_cache;

    std::unordered_map<std::string, ActiveXInvariant> active_x_cache;
    int64_t active_x_cache_hits = 0, active_x_cache_misses = 0;
    int64_t active_set_total_calls = 0;
};

GNNV1CLDSdEditNativeBatched::GNNV1CLDSdEditNativeBatched(
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
                "GNNV1CLDSdEditNativeBatched : p non-uniforme — lagrangian_v1cld suppose des "
                "scénarios équiprobables (p_i = 1/N)");

    if (device == "cuda") {
        if (!torch::cuda::is_available())
            throw std::invalid_argument("GNNV1CLDSdEditNativeBatched: device=cuda demandé mais CUDA indisponible");
        impl_->device = torch::Device(torch::kCUDA);
    } else if (device == "cpu") {
        impl_->device = torch::Device(torch::kCPU);
    } else {
        throw std::invalid_argument("GNNV1CLDSdEditNativeBatched: device inconnu (\"" + device + "\"), attendu cpu|cuda");
    }

    impl_->n = n; impl_->m = m; impl_->N = N; impl_->epsilon = epsilon;
    impl_->c = c; impl_->A = A; impl_->b = b;
    impl_->use_surrogate_zeta = use_surrogate_zeta;

    std::string scripted_path = ensure_v1cld_scripted_batched_export(checkpoint_path, "python3");
    impl_->module = torch::jit::load(scripted_path, impl_->device);
    impl_->module.eval();
    impl_->module = torch::jit::freeze(impl_->module);
    impl_->module = torch::jit::optimize_for_inference(impl_->module);
}

GNNV1CLDSdEditNativeBatched::~GNNV1CLDSdEditNativeBatched()
{
    if (impl_ && impl_->active_set_total_calls > 0) {
        const int64_t cache_total = impl_->active_x_cache_hits + impl_->active_x_cache_misses;
        std::fprintf(stderr,
            "[gnn_v1cld_sd_edit_native_batched] active_x_cache : %lld hits / %lld misses (%.1f%% hit rate)\n",
            static_cast<long long>(impl_->active_x_cache_hits),
            static_cast<long long>(impl_->active_x_cache_misses),
            cache_total > 0 ? 100.0 * impl_->active_x_cache_hits / cache_total : 0.0);
    }
}

void GNNV1CLDSdEditNativeBatched::query_batch(
    const std::vector<std::vector<int8_t>>& x_masks, const std::vector<double>& curr_costs,
    const std::vector<Vec2d>& b_res_list, int depth, std::vector<double>& ub_relative_out,
    GNNBatchCallTiming& timing)
{
    using Clock = std::chrono::steady_clock;
    const int B = static_cast<int>(x_masks.size());
    if (B == 0) { ub_relative_out.clear(); return; }
    if (static_cast<int>(curr_costs.size()) != B || static_cast<int>(b_res_list.size()) != B)
        throw std::invalid_argument("GNNV1CLDSdEditNativeBatched::query_batch : tailles incohérentes");

    auto t0 = Clock::now();

    std::vector<BuiltFeatures> feats;
    feats.reserve(B);
    for (int bIdx = 0; bIdx < B; ++bIdx) {
        const Vec2d& b_res = b_res_list[bIdx];

        std::string active_key;
        active_key.reserve(static_cast<size_t>(impl_->n) * 3);
        for (int k = 0; k < impl_->n; ++k)
            if (x_masks[bIdx][k] == 2) { active_key += std::to_string(k); active_key += ','; }
        ++impl_->active_set_total_calls;

        auto inv_it = impl_->active_x_cache.find(active_key);
        if (inv_it == impl_->active_x_cache.end()) {
            ++impl_->active_x_cache_misses;
            inv_it = impl_->active_x_cache.emplace(
                active_key, compute_active_x_invariant(impl_->m, impl_->N, impl_->c, impl_->A, x_masks[bIdx])).first;
        } else {
            ++impl_->active_x_cache_hits;
        }

        feats.push_back(build_features_cpp(impl_->n, impl_->m, impl_->N, inv_it->second, impl_->b,
                                            b_res, curr_costs[bIdx], depth, impl_->epsilon));
    }

    const int n_g = feats[0].n_g;
    const int m   = impl_->m;
    const int N_z = feats[0].N_z;
    for (int bIdx = 1; bIdx < B; ++bIdx)
        if (feats[bIdx].n_g != n_g)
            throw std::invalid_argument(
                "GNNV1CLDSdEditNativeBatched::query_batch : tous les masques du batch doivent "
                "partager le même n_g (violé au membre " + std::to_string(bIdx) + ")");

    const int node_dim = feats[0].node_dim;
    const int64_t V = static_cast<int64_t>(N_z) * m * n_g;

    torch::Tensor edge_index_1, edge_weight_1;
    auto edge_it = impl_->edge_cache.find(n_g);
    if (edge_it != impl_->edge_cache.end()) {
        edge_index_1  = edge_it->second.first;
        edge_weight_1 = edge_it->second.second;
    } else {
        build_edges_v1cld_cpp(n_g, m, N_z, edge_index_1, edge_weight_1);
        impl_->edge_cache.emplace(n_g, std::make_pair(edge_index_1, edge_weight_1));
    }
    const int64_t E = edge_index_1.size(1);

    torch::Tensor offsets = torch::arange(B, torch::kInt64) * V;
    torch::Tensor ei_rep = edge_index_1.unsqueeze(0).expand({B, 2, E}) + offsets.view({B, 1, 1});
    torch::Tensor edge_index_B = ei_rep.permute({1, 0, 2}).reshape({2, B * E}).contiguous();
    torch::Tensor edge_weight_B = edge_weight_1.unsqueeze(0).expand({B, E}).reshape({B * E}).contiguous();

    torch::Tensor X = torch::empty({static_cast<int64_t>(B) * V, node_dim}, torch::kFloat32);
    torch::Tensor ctx = torch::empty({B, 9}, torch::kFloat32);
    auto ctx_acc = ctx.accessor<float, 2>();
    for (int bIdx = 0; bIdx < B; ++bIdx) {
        std::memcpy(X.data_ptr<float>() + static_cast<int64_t>(bIdx) * V * node_dim,
                    feats[bIdx].X_flat.data(), sizeof(float) * V * node_dim);
        for (int c9 = 0; c9 < 9; ++c9) ctx_acc[bIdx][c9] = feats[bIdx].ctx[c9];
    }

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
                                               static_cast<int64_t>(n_g), static_cast<int64_t>(m),
                                               static_cast<int64_t>(B)};
    torch::Tensor h_out = impl_->module.forward(inputs).toTensor();
    if (impl_->device.is_cuda()) h_out = h_out.to(torch::kCPU);
    auto t2 = Clock::now();

    ub_relative_out.assign(B, 0.0);
    torch::Tensor h_out_2d = h_out.view({B, V});
    for (int bIdx = 0; bIdx < B; ++bIdx) {
        // Rescale : c_scale/(N*m), cf. GNN.forward de train_v1CLD_sd_edit.py:640.
        torch::Tensor lam_b = h_out_2d[bIdx] * feats[bIdx].c_scale
                              / (static_cast<double>(feats[bIdx].N) * static_cast<double>(m));
        ub_relative_out[bIdx] = lagrangian_v1cld_cpp(lam_b, feats[bIdx].A_g, feats[bIdx].b_res_g,
                                                      feats[bIdx].c_g, N_z, m, n_g, feats[bIdx].N,
                                                      feats[bIdx].n_fixed_z1, impl_->epsilon,
                                                      impl_->use_surrogate_zeta);
    }
    auto t3 = Clock::now();

    timing.t_graph_build_s   = std::chrono::duration<double>(t1 - t0).count();
    timing.t_forward_s       = std::chrono::duration<double>(t2 - t1).count();
    timing.t_bound_compute_s = std::chrono::duration<double>(t3 - t2).count();
}
