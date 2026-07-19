/*
 * gnn_edge_vectorized_probe.cpp — Sonde diagnostique : compare la
 * construction ACTUELLE des arêtes du graphe GNN (boucles C++ scalaires +
 * push_back, portées à l'identique depuis build_graph_cpp dans
 * gnn_v1ld_native.cpp) à une construction VECTORISÉE torch (templates
 * précalculés une fois + broadcast), sans changer le modèle ni le layout des
 * nœuds (row = i*n_g+k, cf. gnn_v1ld_scriptable.py:114-123, qui fait
 * h_v.view(n_z, n_g, hidden) — un simple reshape qui exige cet ordre exact).
 *
 * Contexte : décomposition mesurée sur job 16028283 (n=20/N=100) — arêtes
 * = 59% du temps de "construction de graphe" (17.11s / 28.99s cumulés sur
 * 52120 appels more()), features = 31%. Cf. discussion "héritage parent vs
 * calcul torch" : l'héritage cross-nœud ne réduit pas la complexité ici
 * (le stride n_g change à chaque nœud, donc les indices ne sont PAS stables
 * d'un parent à l'enfant) et alourdirait le clonage Gecode. Cette sonde
 * teste l'alternative retenue : exploiter que le PATRON de connexité est
 * fixe et connu à l'avance (scénario-scénario ne dépend que de N, qui ne
 * varie JAMAIS dans l'arbre ; item-item ne dépend que de n_g, qui ne prend
 * que n+1 valeurs possibles au total — mêmes n+1 templates précalculés UNE
 * FOIS, comme le pré-chauffage TorchScript déjà en place pour les formes
 * n_g=1..n).
 *
 * Ne dépend d'AUCUN checkpoint/modèle : seule la TOPOLOGIE (item-item +
 * scénario-scénario) est en jeu, indépendante des features/poids appris.
 * Vérifie l'équivalence en tant que MULTI-ENSEMBLE (src,dst,weight) — l'ordre
 * des arêtes n'importe pas pour un GNN à passage de messages (agrégation par
 * somme/scatter, invariante à l'ordre). Pas un binaire de production.
 *
 * Usage : gnn_edge_vectorized_probe [n=20] [N=100] [device=cpu] [n_iters=200]
 */

#include <torch/torch.h>
#include <torch/cuda.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

// ── Construction scalaire (portage exact des boucles de build_graph_cpp,
//    gnn_v1ld_native.cpp lignes ~157-189) — la RÉFÉRENCE à laquelle comparer.
std::pair<torch::Tensor, torch::Tensor> build_edges_scalar(int n_g, int N_z, torch::Device device)
{
    std::vector<int64_t> src, dst; std::vector<float> ew;
    src.reserve(static_cast<size_t>(N_z) * n_g * (n_g - 1) + static_cast<size_t>(n_g) * N_z * (N_z - 1));
    dst.reserve(src.capacity()); ew.reserve(src.capacity());

    for (int i = 0; i < N_z; ++i)
        for (int k1 = 0; k1 < n_g; ++k1)
            for (int k2 = k1 + 1; k2 < n_g; ++k2) {
                int64_t a = static_cast<int64_t>(i) * n_g + k1, b2 = static_cast<int64_t>(i) * n_g + k2;
                src.push_back(a); dst.push_back(b2); ew.push_back(1.0f / n_g);
                src.push_back(b2); dst.push_back(a); ew.push_back(1.0f / n_g);
            }
    for (int kf = 0; kf < n_g; ++kf)
        for (int i1 = 0; i1 < N_z; ++i1)
            for (int i2 = i1 + 1; i2 < N_z; ++i2) {
                int64_t a = static_cast<int64_t>(i1) * n_g + kf, b2 = static_cast<int64_t>(i2) * n_g + kf;
                src.push_back(a); dst.push_back(b2); ew.push_back(1.0f / N_z);
                src.push_back(b2); dst.push_back(a); ew.push_back(1.0f / N_z);
            }

    auto edge_index = torch::stack({
        torch::from_blob(src.data(), {(int64_t)src.size()}, torch::kInt64).clone(),
        torch::from_blob(dst.data(), {(int64_t)dst.size()}, torch::kInt64).clone()});
    auto edge_weight = torch::from_blob(ew.data(), {(int64_t)ew.size()}, torch::kFloat32).clone();
    if (device.is_cuda()) { edge_index = edge_index.to(device); edge_weight = edge_weight.to(device); }
    return {edge_index, edge_weight};
}

// ── Construction scalaire STAR (portage exact de gnn_sd_edit_native.cpp,
//    lignes ~163-180) — item-item IDENTIQUE à la variante homogène, mais
//    scénario-scénario en ÉTOILE ancrée au scénario 0 (poids 1.0, PAS 1/N_z),
//    O(n_g·N_z) au lieu de O(n_g·N_z²).
std::pair<torch::Tensor, torch::Tensor> build_star_edges_scalar(int n_g, int N_z, torch::Device device)
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

    auto edge_index = torch::stack({
        torch::from_blob(src.data(), {(int64_t)src.size()}, torch::kInt64).clone(),
        torch::from_blob(dst.data(), {(int64_t)dst.size()}, torch::kInt64).clone()});
    auto edge_weight = torch::from_blob(ew.data(), {(int64_t)ew.size()}, torch::kFloat32).clone();
    if (device.is_cuda()) { edge_index = edge_index.to(device); edge_weight = edge_weight.to(device); }
    return {edge_index, edge_weight};
}

// ── Templates précalculés UNE FOIS (hors boucle chronométrée) ───────────────
// sc_i1/sc_i2  : paires de scénarios (i1<i2), dépend seulement de N — fixe
//                pour TOUT l'arbre B&B (N ne rétrécit jamais). Clique
//                (variante homogène).
// spoke_scen   : scénarios 1..N_z-1 (étoile ancrée à 0), dépend seulement de
//                N — fixe pour tout l'arbre. Variante sd_edit.
// item_i1/i2[s]: paires d'items (k1<k2) pour n_g==s — s ne prend que n+1
//                valeurs possibles au total, précalculées comme le
//                pré-chauffage TorchScript existant (formes n_g=1..n).
//                IDENTIQUE pour les deux variantes (item-item ne change pas).
struct EdgeTemplates {
    torch::Tensor sc_i1, sc_i2, spoke_scen;
    std::vector<torch::Tensor> item_i1, item_i2;

    EdgeTemplates(int n, int N_z, torch::Device device) {
        auto opts = torch::TensorOptions().dtype(torch::kInt64).device(device);
        if (N_z >= 2) {
            auto sc_tri = torch::triu_indices(N_z, N_z, /*offset=*/1, opts);
            sc_i1 = sc_tri[0].contiguous(); sc_i2 = sc_tri[1].contiguous();
            spoke_scen = torch::arange(1, N_z, opts);
        }
        item_i1.resize(n + 1); item_i2.resize(n + 1);
        for (int s = 2; s <= n; ++s) {
            auto tri = torch::triu_indices(s, s, /*offset=*/1, opts);
            item_i1[s] = tri[0].contiguous(); item_i2[s] = tri[1].contiguous();
        }
    }
};

// ── Construction vectorisée : templates + broadcast, AUCUNE boucle C++ ──────
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

// ── Construction vectorisée STAR (item-item identique ; scénario-scénario en
//    étoile ancrée au scénario 0, poids 1.0 constant) ────────────────────────
std::pair<torch::Tensor, torch::Tensor> build_star_edges_vectorized(
    int n_g, int N_z, const EdgeTemplates& tpl, torch::Device device)
{
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    torch::Tensor ii_src = torch::empty({0}, opts_i64), ii_dst = torch::empty({0}, opts_i64),
                  ii_w   = torch::empty({0}, opts_f32);
    if (n_g >= 2) {
        const auto& i1 = tpl.item_i1[n_g];
        const auto& i2 = tpl.item_i2[n_g];
        auto scen_off = torch::arange(N_z, opts_i64) * n_g;
        auto fwd_src  = (i1.unsqueeze(0) + scen_off.unsqueeze(1)).flatten();
        auto fwd_dst  = (i2.unsqueeze(0) + scen_off.unsqueeze(1)).flatten();
        ii_src = torch::cat({fwd_src, fwd_dst});
        ii_dst = torch::cat({fwd_dst, fwd_src});
        ii_w   = torch::full({ii_src.size(0)}, 1.0f / n_g, opts_f32);
    }

    torch::Tensor ss_src = torch::empty({0}, opts_i64), ss_dst = torch::empty({0}, opts_i64),
                  ss_w   = torch::empty({0}, opts_f32);
    if (N_z >= 2) {
        auto item_off    = torch::arange(n_g, opts_i64);                                          // [n_g] (== indices ancre, 0*n_g+kf=kf)
        auto spoke_full  = (tpl.spoke_scen.unsqueeze(1) * n_g + item_off.unsqueeze(0)).contiguous().flatten(); // [(N_z-1)*n_g]
        auto anchor_full = item_off.unsqueeze(0).expand({N_z - 1, n_g}).contiguous().flatten();               // [(N_z-1)*n_g]
        ss_src = torch::cat({anchor_full, spoke_full});
        ss_dst = torch::cat({spoke_full, anchor_full});
        ss_w   = torch::full({ss_src.size(0)}, 1.0f, opts_f32);
    }

    auto edge_index  = torch::stack({torch::cat({ii_src, ss_src}), torch::cat({ii_dst, ss_dst})});
    auto edge_weight = torch::cat({ii_w, ss_w});
    return {edge_index, edge_weight};
}

// ── Comparaison en tant que MULTI-ENSEMBLE (src,dst,weight) ─────────────────
bool edges_equivalent(const torch::Tensor& idx_a, const torch::Tensor& w_a,
                       const torch::Tensor& idx_b, const torch::Tensor& w_b,
                       std::string& detail)
{
    if (idx_a.size(1) != idx_b.size(1)) {
        detail = "nb arêtes différent : " + std::to_string(idx_a.size(1))
                + " vs " + std::to_string(idx_b.size(1));
        return false;
    }
    auto pack = [](const torch::Tensor& idx, const torch::Tensor& w) {
        auto idx_cpu = idx.to(torch::kCPU).contiguous();
        auto w_cpu   = w.to(torch::kCPU).contiguous();
        int64_t E = idx_cpu.size(1);
        auto idx_acc = idx_cpu.accessor<int64_t, 2>();
        auto w_acc   = w_cpu.accessor<float, 1>();
        std::vector<std::tuple<int64_t, int64_t, float>> v(E);
        for (int64_t e = 0; e < E; ++e) v[e] = {idx_acc[0][e], idx_acc[1][e], w_acc[e]};
        std::sort(v.begin(), v.end());
        return v;
    };
    auto va = pack(idx_a, w_a), vb = pack(idx_b, w_b);
    if (va != vb) { detail = "contenu différent après tri"; return false; }
    detail = "OK (" + std::to_string(va.size()) + " arêtes)";
    return true;
}

struct Stats { double mean, median, min, max; };
Stats compute_stats(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double median = (n % 2 == 0) ? 0.5 * (v[n / 2 - 1] + v[n / 2]) : v[n / 2];
    return {sum / static_cast<double>(n), median, v.front(), v.back()};
}

double median_time_s(const std::function<void()>& fn, int n_iters, torch::Device device)
{
    fn();  // warm-up (init allocateur CUDA, premiers lancements de kernels)
    if (device.is_cuda()) torch::cuda::synchronize();
    std::vector<double> times(n_iters);
    for (int it = 0; it < n_iters; ++it) {
        auto t0 = Clock::now();
        fn();
        if (device.is_cuda()) torch::cuda::synchronize();
        times[it] = std::chrono::duration<double>(Clock::now() - t0).count();
    }
    return compute_stats(times).median;
}

} // namespace

int main(int argc, char* argv[])
{
    int n              = (argc > 1) ? std::stoi(argv[1]) : 20;
    int N              = (argc > 2) ? std::stoi(argv[2]) : 100;
    std::string dev_s  = (argc > 3) ? argv[3] : "cpu";
    int n_iters        = (argc > 4) ? std::stoi(argv[4]) : 200;

    torch::Device device(dev_s == "cuda" ? torch::kCUDA : torch::kCPU);
    if (device.is_cuda() && !torch::cuda::is_available()) {
        std::cerr << "[probe] CUDA demandé mais indisponible — bascule cpu\n";
        device = torch::Device(torch::kCPU);
        dev_s = "cpu";
    }

    std::cout << "[probe] n=" << n << " N=" << N << " device=" << dev_s
              << " n_iters=" << n_iters << "\n";

    EdgeTemplates tpl(n, N, device);

    std::vector<int> ng_values = {n, std::max(2, n - 1), std::max(2, n / 2), 3, 2};

    std::cout << "\n=== [clique/homogène] Vérification d'équivalence (scalaire vs vectorisé) ===\n";
    bool all_ok = true;
    for (int n_g : ng_values) {
        auto [idx_s, w_s] = build_edges_scalar(n_g, N, device);
        auto [idx_v, w_v] = build_edges_vectorized(n_g, N, tpl, device);
        std::string detail;
        bool ok = edges_equivalent(idx_s, w_s, idx_v, w_v, detail);
        std::cout << "  n_g=" << n_g << " : " << (ok ? "OK" : "ÉCHEC") << " — " << detail << "\n";
        all_ok = all_ok && ok;
    }

    std::cout << "\n=== [étoile/sd_edit] Vérification d'équivalence (scalaire vs vectorisé) ===\n";
    for (int n_g : ng_values) {
        auto [idx_s, w_s] = build_star_edges_scalar(n_g, N, device);
        auto [idx_v, w_v] = build_star_edges_vectorized(n_g, N, tpl, device);
        std::string detail;
        bool ok = edges_equivalent(idx_s, w_s, idx_v, w_v, detail);
        std::cout << "  n_g=" << n_g << " : " << (ok ? "OK" : "ÉCHEC") << " — " << detail << "\n";
        all_ok = all_ok && ok;
    }

    if (!all_ok) {
        std::cerr << "\n[probe] ÉCHEC : la construction vectorisée diverge de la référence scalaire.\n";
        return 1;
    }
    std::cout << "\n[probe] équivalence numérique validée sur toutes les configurations testées (clique + étoile).\n";

    std::cout << "\n=== [clique/homogène] Benchmark (médiane sur " << n_iters << " itérations, hors warm-up) ===\n";
    std::cout << std::fixed << std::setprecision(4);
    for (int n_g = n; n_g >= 1; n_g = (n_g > 10 ? n_g - 5 : n_g - 2)) {
        if (n_g < 1) break;
        double t_scalar = median_time_s([&]{ build_edges_scalar(n_g, N, device); }, n_iters, device);
        double t_vec    = median_time_s([&]{ build_edges_vectorized(n_g, N, tpl, device); }, n_iters, device);
        std::cout << "  n_g=" << std::setw(3) << n_g
                  << "  scalaire=" << t_scalar * 1000.0 << "ms"
                  << "  vectorisé=" << t_vec * 1000.0 << "ms"
                  << "  speedup=" << (t_vec > 0 ? t_scalar / t_vec : 0.0) << "x\n";
    }

    std::cout << "\n=== [étoile/sd_edit] Benchmark (médiane sur " << n_iters << " itérations, hors warm-up) ===\n";
    for (int n_g = n; n_g >= 1; n_g = (n_g > 10 ? n_g - 5 : n_g - 2)) {
        if (n_g < 1) break;
        double t_scalar = median_time_s([&]{ build_star_edges_scalar(n_g, N, device); }, n_iters, device);
        double t_vec    = median_time_s([&]{ build_star_edges_vectorized(n_g, N, tpl, device); }, n_iters, device);
        std::cout << "  n_g=" << std::setw(3) << n_g
                  << "  scalaire=" << t_scalar * 1000.0 << "ms"
                  << "  vectorisé=" << t_vec * 1000.0 << "ms"
                  << "  speedup=" << (t_vec > 0 ? t_scalar / t_vec : 0.0) << "x\n";
    }
    return 0;
}
