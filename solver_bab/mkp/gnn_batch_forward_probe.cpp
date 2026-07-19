/*
 * gnn_batch_forward_probe.cpp — Sonde isolée (aucune instance MKP réelle,
 * aucun Gecode) : le batching de B nœuds B&B dans UN SEUL appel GNN
 * amortit-il vraiment le plancher d'exécution/dispatch par kernel déjà
 * identifié (cf. [[project_gnn_light_sdedit_speedup_result]],
 * [[project_gnn_cudagraph_probe_result]]) ?
 *
 * Compare, pour chaque taille de batch B :
 *   (1) B appels séquentiels EAGER-ÉQUIVALENT du module scripté NON batché
 *       de production (chemin actuel, un nœud = un appel) ;
 *   (2) 1 appel du module scripté BATCHÉ (gnn_sd_edit_scriptable_batched.py)
 *       sur les MÊMES B graphes empilés en bloc-diagonal.
 *
 * edge_index/edge_weight sont structurellement invariants pour un (n_g, N_z)
 * fixé (ne dépendent jamais de A/b/c) — construits une fois, hors
 * chronométrage, comme gnn_cudagraph_probe.cpp. Seule la FORME des tenseurs
 * compte pour ce chronométrage, pas les valeurs numériques (validé
 * séparément par validate_batched_scriptable.py).
 *
 * C'est le go/no-go le MOINS CHER de l'Étape 2 (cf. plan) : si le speedup(B)
 * mesuré ici est décevant, inutile d'investir dans l'intégration Gecode
 * (Étape 2e).
 *
 * Usage : gnn_batch_forward_probe <scripted_single.pt> <scripted_batched.pt>
 *         [device=cuda] [n_g=20] [N_z=50] [m=5] [n_iters=200] [n_warmup=30]
 *         [B_list="1,2,4,8,16,32,64,128"]
 */
#include <torch/script.h>
#include <torch/cuda.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Stats { double mean, median; };

Stats compute_stats(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / static_cast<double>(n);
    double median = (n % 2 == 0) ? 0.5 * (v[n / 2 - 1] + v[n / 2]) : v[n / 2];
    return {mean, median};
}

// Arêtes sd_edit (clique item + étoile scénario ancrée au scénario 0) — cf.
// gnn_sd_edit_native.cpp / gnn_cudagraph_probe.cpp, ne dépend que de (n_g, N_z).
void build_edges_sd_edit(int n_g, int N_z,
                          std::vector<int64_t>& src, std::vector<int64_t>& dst,
                          std::vector<float>& ew)
{
    for (int i = 0; i < N_z; ++i)
        for (int k1 = 0; k1 < n_g; ++k1)
            for (int k2 = k1 + 1; k2 < n_g; ++k2) {
                int64_t a  = static_cast<int64_t>(i) * n_g + k1;
                int64_t b2 = static_cast<int64_t>(i) * n_g + k2;
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
}

std::vector<int> parse_int_list(const std::string& s)
{
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) out.push_back(std::stoi(tok));
    return out;
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage : gnn_batch_forward_probe <scripted_single.pt> "
                     "<scripted_batched.pt> [device=cuda] [n_g=20] [N_z=50] "
                     "[m=5] [n_iters=200] [n_warmup=30] "
                     "[B_list=1,2,4,8,16,32,64,128]\n";
        return 1;
    }
    std::string single_path  = argv[1];
    std::string batched_path = argv[2];
    std::string device_str   = (argc > 3) ? argv[3] : "cuda";
    int n_g       = (argc > 4) ? std::stoi(argv[4]) : 20;
    int N_z       = (argc > 5) ? std::stoi(argv[5]) : 50;
    int m         = (argc > 6) ? std::stoi(argv[6]) : 5;
    int n_iters   = (argc > 7) ? std::stoi(argv[7]) : 200;
    int n_warmup  = (argc > 8) ? std::stoi(argv[8]) : 30;
    std::vector<int> B_list = (argc > 9) ? parse_int_list(argv[9])
                                          : std::vector<int>{1, 2, 4, 8, 16, 32, 64, 128};

    if (device_str == "cuda" && !torch::cuda::is_available()) {
        std::cerr << "[probe] CUDA indisponible.\n";
        return 1;
    }
    torch::Device device(device_str == "cuda" ? torch::kCUDA : torch::kCPU);

    int V        = N_z * n_g;
    int node_dim = 7 * m + 5;
    std::cout << "[probe] n_g=" << n_g << " N_z=" << N_z << " m=" << m
              << " node_dim=" << node_dim << " V(par graphe)=" << V
              << " device=" << device_str << "\n";

    torch::jit::script::Module module_single = torch::jit::load(single_path, device);
    module_single.eval();
    torch::jit::script::Module module_batched = torch::jit::load(batched_path, device);
    module_batched.eval();
    torch::NoGradGuard no_grad;

    std::vector<int64_t> src, dst; std::vector<float> ew;
    build_edges_sd_edit(n_g, N_z, src, dst, ew);
    int64_t E = static_cast<int64_t>(src.size());
    std::cout << "[probe] E(par graphe)=" << E << " arêtes (directed)\n";

    torch::Tensor edge_index_1 = torch::stack({
        torch::from_blob(src.data(), {E}, torch::kInt64).clone(),
        torch::from_blob(dst.data(), {E}, torch::kInt64).clone()}).to(device);
    torch::Tensor edge_weight_1 = torch::from_blob(ew.data(), {E}, torch::kFloat32).clone().to(device);

    auto rand_X   = [&](int rows) { return torch::rand({rows, node_dim}, torch::kFloat32); };
    auto rand_ctx = [&](int rows) { return torch::rand({rows, 9}, torch::kFloat32); };

    std::cout << "\n" << std::string(78, '=') << "\n";
    std::cout << "  RÉSUMÉ (µs/graphe)\n";
    std::cout << std::string(78, '=') << "\n";
    std::cout << "  " << std::setw(6) << "B" << std::setw(16) << "seq_mean" << std::setw(16)
              << "batched_mean" << std::setw(14) << "speedup" << "\n";

    for (int B : B_list) {
        // ── (1) B appels séquentiels du module NON batché ──────────────────
        std::vector<double> t_seq;
        t_seq.reserve(n_iters);
        for (int it = 0; it < n_iters + n_warmup; ++it) {
            std::vector<torch::Tensor> X_cpu(B), ctx_cpu(B);
            for (int b = 0; b < B; ++b) { X_cpu[b] = rand_X(V); ctx_cpu[b] = rand_ctx(1).squeeze(0); }

            if (device.is_cuda()) torch::cuda::synchronize();
            auto t0 = std::chrono::steady_clock::now();
            for (int b = 0; b < B; ++b) {
                torch::Tensor X   = X_cpu[b].to(device);
                torch::Tensor ctx = ctx_cpu[b].to(device);
                std::vector<torch::jit::IValue> inputs = {X, edge_index_1, edge_weight_1, ctx,
                                                            static_cast<int64_t>(n_g)};
                auto out = module_single.forward(inputs).toTensor();
            }
            if (device.is_cuda()) torch::cuda::synchronize();
            auto t1 = std::chrono::steady_clock::now();
            if (it >= n_warmup)
                t_seq.push_back(std::chrono::duration<double>(t1 - t0).count() / B);
        }

        // ── (2) 1 appel batché sur les mêmes B graphes ──────────────────────
        // edge_index/edge_weight dupliqués B fois avec offset — construits
        // UNE FOIS par valeur de B (structure invariante tant que n_g/N_z/B
        // fixés), hors chronométrage — même convention que edge_index_1.
        torch::Tensor offsets = torch::arange(B, torch::kInt64).to(device) * V;
        torch::Tensor ei_rep = edge_index_1.unsqueeze(0).expand({B, 2, E})
                                   + offsets.view({B, 1, 1});
        torch::Tensor edge_index_B = ei_rep.permute({1, 0, 2}).reshape({2, B * E}).contiguous();
        torch::Tensor edge_weight_B = edge_weight_1.unsqueeze(0).expand({B, E}).reshape({B * E}).contiguous();

        std::vector<double> t_batched;
        t_batched.reserve(n_iters);
        for (int it = 0; it < n_iters + n_warmup; ++it) {
            torch::Tensor X_cpu   = rand_X(B * V);
            torch::Tensor ctx_cpu = rand_ctx(B);

            if (device.is_cuda()) torch::cuda::synchronize();
            auto t0 = std::chrono::steady_clock::now();
            torch::Tensor X   = X_cpu.to(device);
            torch::Tensor ctx = ctx_cpu.to(device);
            std::vector<torch::jit::IValue> inputs = {X, edge_index_B, edge_weight_B, ctx,
                                                        static_cast<int64_t>(n_g), static_cast<int64_t>(B)};
            auto out = module_batched.forward(inputs).toTensor();
            if (device.is_cuda()) torch::cuda::synchronize();
            auto t1 = std::chrono::steady_clock::now();
            if (it >= n_warmup)
                t_batched.push_back(std::chrono::duration<double>(t1 - t0).count() / B);
        }

        Stats s_seq = compute_stats(t_seq);
        Stats s_batched = compute_stats(t_batched);
        double speedup = s_seq.mean / s_batched.mean;

        std::cout << "  " << std::setw(6) << B
                  << std::setw(13) << (s_seq.mean * 1e6) << "us"
                  << std::setw(13) << (s_batched.mean * 1e6) << "us"
                  << std::setw(11) << std::fixed << std::setprecision(2) << speedup << "x\n";
    }

    return 0;
}
