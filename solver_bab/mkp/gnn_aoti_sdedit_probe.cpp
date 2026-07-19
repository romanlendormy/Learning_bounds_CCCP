/*
 * gnn_aoti_sdedit_probe.cpp — Mesure du gain wall-clock RÉEL (pas une
 * estimation profiler) d'AOTInductor par rapport au chemin de production
 * actuel (TorchScript, torch::jit::load) sur le VRAI modèle sd_edit.
 *
 * Complète [[project_gnn_compile_fusion_potential_result]] (gain ×1.4-1.6
 * mesuré côté profiler Python, PAS encore une mesure C++ directe) — même
 * méthodologie que gnn_cudagraph_probe.cpp (chronométrage wall-clock,
 * warmup séparé, edge_index/edge_weight hors chronométrage) mais compare
 * TorchScript à un artefact AOTInductor réel (cf.
 * gnn_sd_edit/export_aoti_sdedit.py) au lieu de eager vs CUDA graph replay.
 *
 * Usage : gnn_aoti_sdedit_probe <scripted_module.pt> <aoti_package.pt2>
 *         [device=cuda] [n_g=20] [N_z=50] [m=5] [n_iters=2000] [n_warmup=50]
 *
 * Charge le .pt2 via AOTIModelPackageLoader (PAS AOTIModelContainerRunnerCuda
 * sur un .so brut) — le modèle utilise nn.GRUCell, un op sans C-shim natif,
 * qui nécessite le "proxy executor" (table JSON) que seule la voie package
 * génère (découvert empiriquement, cf. commentaire dans export_aoti_sdedit.py).
 */
#include <torch/script.h>
#include <torch/cuda.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct Stats { double mean, median, stdev, min, max; };

Stats compute_stats(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / static_cast<double>(n);
    double sq = 0.0;
    for (double x : v) sq += (x - mean) * (x - mean);
    double stdev = std::sqrt(sq / static_cast<double>(n));
    double median = (n % 2 == 0) ? 0.5 * (v[n / 2 - 1] + v[n / 2]) : v[n / 2];
    return {mean, median, stdev, v.front(), v.back()};
}

void print_stats(const std::string& label, const Stats& s)
{
    std::cout << "  " << label << " : mean=" << s.mean * 1000.0 << "ms"
              << " median=" << s.median * 1000.0 << "ms"
              << " stdev=" << s.stdev * 1000.0 << "ms"
              << " min=" << s.min * 1000.0 << "ms"
              << " max=" << s.max * 1000.0 << "ms\n";
}

// Étoile scénario-scénario ancrée au scénario 0 — IDENTIQUE à
// gnn_sd_edit_native.cpp / gnn_cudagraph_probe.cpp / export_aoti_sdedit.py.
void build_sdedit_edges(int n_g, int N_z,
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
        int64_t anchor = kf;
        for (int i2 = 1; i2 < N_z; ++i2) {
            int64_t spoke = static_cast<int64_t>(i2) * n_g + kf;
            src.push_back(anchor); dst.push_back(spoke); ew.push_back(1.0f);
            src.push_back(spoke); dst.push_back(anchor); ew.push_back(1.0f);
        }
    }
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage : gnn_aoti_sdedit_probe <scripted_module.pt> "
                     "<aoti_package.pt2> [device=cuda] [n_g=20] [N_z=50] [m=5] "
                     "[n_iters=2000] [n_warmup=50]\n";
        return 1;
    }
    std::string module_path = argv[1];
    std::string pt2_path    = argv[2];
    std::string device_str  = (argc > 3) ? argv[3] : "cuda";
    int n_g                 = (argc > 4) ? std::stoi(argv[4]) : 20;
    int N_z                 = (argc > 5) ? std::stoi(argv[5]) : 50;
    int m                   = (argc > 6) ? std::stoi(argv[6]) : 5;
    int n_iters             = (argc > 7) ? std::stoi(argv[7]) : 2000;
    int n_warmup            = (argc > 8) ? std::stoi(argv[8]) : 50;

    if (device_str != "cuda" || !torch::cuda::is_available()) {
        std::cerr << "[probe] cette sonde ne teste que device=cuda (AOTIModelPackageLoader).\n";
        return 1;
    }
    torch::Device device(torch::kCUDA);

    int V        = N_z * n_g;
    int node_dim = 7 * m + 5;
    std::cout << "[probe] n_g=" << n_g << " N_z=" << N_z << " m=" << m
              << " node_dim=" << node_dim << " V=" << V << " variant=sd_edit\n";

    torch::jit::script::Module module = torch::jit::load(module_path, device);
    module.eval();
    torch::NoGradGuard no_grad;

    torch::inductor::AOTIModelPackageLoader aoti_loader(pt2_path);

    std::vector<int64_t> src, dst; std::vector<float> ew;
    build_sdedit_edges(n_g, N_z, src, dst, ew);
    int64_t E = static_cast<int64_t>(src.size());
    std::cout << "[probe] E=" << E << " arêtes (directed)\n";

    torch::Tensor edge_index = torch::stack({
        torch::from_blob(src.data(), {E}, torch::kInt64).clone(),
        torch::from_blob(dst.data(), {E}, torch::kInt64).clone()}).to(device);
    torch::Tensor edge_weight = torch::from_blob(ew.data(), {E}, torch::kFloat32).clone().to(device);

    auto rand_X   = [&]() { return torch::rand({V, node_dim}, torch::kFloat32); };
    auto rand_ctx = [&]() { return torch::rand({9}, torch::kFloat32); };

    // ── Sanity check : mêmes données, TorchScript vs AOTInductor ───────────
    {
        torch::Tensor X_cpu = rand_X(), ctx_cpu = rand_ctx();
        torch::Tensor X = X_cpu.to(device), ctx = ctx_cpu.to(device);

        std::vector<torch::jit::IValue> ts_inputs = {X, edge_index, edge_weight, ctx,
                                                       static_cast<int64_t>(n_g)};
        torch::Tensor out_ts = module.forward(ts_inputs).toTensor();

        std::vector<at::Tensor> aoti_inputs{X, edge_index, edge_weight, ctx};
        torch::Tensor out_aoti = aoti_loader.run(aoti_inputs)[0];

        double max_diff = (out_ts - out_aoti).abs().max().item<double>();
        std::cout << "[probe] sanity check TorchScript vs AOTInductor : max|diff|="
                   << max_diff << "\n";
        if (max_diff > 1e-3) {
            std::cerr << "[probe] ATTENTION : écart suspect, résultats potentiellement invalides.\n";
        }
    }

    // ── TorchScript (production actuelle) ───────────────────────────────────
    std::vector<double> t_ts;
    t_ts.reserve(n_iters);
    for (int it = 0; it < n_iters; ++it) {
        torch::Tensor X_cpu = rand_X(), ctx_cpu = rand_ctx();
        auto t0 = std::chrono::steady_clock::now();
        torch::Tensor X = X_cpu.to(device), ctx = ctx_cpu.to(device);
        std::vector<torch::jit::IValue> inputs = {X, edge_index, edge_weight, ctx,
                                                    static_cast<int64_t>(n_g)};
        torch::Tensor out = module.forward(inputs).toTensor();
        torch::cuda::synchronize();
        auto t1 = std::chrono::steady_clock::now();
        t_ts.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    std::vector<double> t_ts_steady(t_ts.begin() + std::min(n_warmup, n_iters - 1), t_ts.end());
    std::cout << "\n[probe] === TorchScript (production actuelle) ===\n";
    print_stats("forward TorchScript (" + std::to_string(t_ts_steady.size()) + " appels)",
                compute_stats(t_ts_steady));

    // ── AOTInductor ──────────────────────────────────────────────────────────
    std::vector<double> t_aoti;
    t_aoti.reserve(n_iters);
    for (int it = 0; it < n_iters; ++it) {
        torch::Tensor X_cpu = rand_X(), ctx_cpu = rand_ctx();
        auto t0 = std::chrono::steady_clock::now();
        torch::Tensor X = X_cpu.to(device), ctx = ctx_cpu.to(device);
        std::vector<at::Tensor> inputs{X, edge_index, edge_weight, ctx};
        torch::Tensor out = aoti_loader.run(inputs)[0];
        torch::cuda::synchronize();
        auto t1 = std::chrono::steady_clock::now();
        t_aoti.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    std::vector<double> t_aoti_steady(t_aoti.begin() + std::min(n_warmup, n_iters - 1), t_aoti.end());
    std::cout << "\n[probe] === AOTInductor ===\n";
    print_stats("forward AOTInductor (" + std::to_string(t_aoti_steady.size()) + " appels)",
                compute_stats(t_aoti_steady));

    double speedup = compute_stats(t_ts_steady).mean / compute_stats(t_aoti_steady).mean;
    std::cout << "\n[probe] === Gain wall-clock réel (TorchScript / AOTInductor) === x"
              << speedup << "\n";

    return 0;
}
