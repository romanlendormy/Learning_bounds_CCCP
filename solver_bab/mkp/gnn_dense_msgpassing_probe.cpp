/*
 * gnn_dense_msgpassing_probe.cpp — Mesure du gain réel (C++/TorchScript,
 * pas Python — cf. mémoire "cudagraph-python-probe-overestimate", un
 * chrono Python confond l'overhead d'interpréteur avec le vrai gain GPU) de
 * la reformulation dense du message passing sd_edit
 * (gnn_sd_edit/prototype_dense_msgpassing.py), validée numériquement
 * équivalente à la version sparse actuelle (max|diff|~1e-7 sur 6 formes,
 * mêmes poids, aucun réentraînement).
 *
 * Charge DEUX modules scriptés distincts (mêmes poids, agrégation
 * différente) :
 *   - référence (sparse)  : ckpt_..._ep0500.pt.scripted.pt
 *                           (X, edge_index, edge_weight, context, n_g)
 *   - dense (sans scatter): ckpt_..._ep0500.pt.dense_sdedit.scripted.pt
 *                           (X, context, n_g) — pas d'arêtes nécessaires
 *
 * Usage : gnn_dense_msgpassing_probe <sparse_module.pt> <dense_module.pt>
 *         [device=cuda] [n_g=20] [N_z=50] [m=5] [n_iters=2000] [n_warmup=50]
 */
#include <torch/script.h>
#include <torch/cuda.h>

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

// Arêtes sd_edit (clique item + étoile scénario) — IDENTIQUE à
// gnn_sd_edit_native.cpp L163-180 et gnn_cudagraph_probe.cpp.
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
        int64_t anchor = static_cast<int64_t>(0) * n_g + kf;
        for (int i2 = 1; i2 < N_z; ++i2) {
            int64_t spoke = static_cast<int64_t>(i2) * n_g + kf;
            src.push_back(anchor); dst.push_back(spoke); ew.push_back(1.0f);
            src.push_back(spoke); dst.push_back(anchor); ew.push_back(1.0f);
        }
    }
}

double time_forward_sparse(torch::jit::script::Module& module, torch::Tensor X,
                            torch::Tensor edge_index, torch::Tensor edge_weight,
                            torch::Tensor context, int n_g, bool is_cuda)
{
    std::vector<torch::jit::IValue> inputs = {X, edge_index, edge_weight, context,
                                                static_cast<int64_t>(n_g)};
    auto t0 = std::chrono::steady_clock::now();
    torch::Tensor out = module.forward(inputs).toTensor();
    if (is_cuda) torch::cuda::synchronize();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

double time_forward_dense(torch::jit::script::Module& module, torch::Tensor X,
                           torch::Tensor context, int n_g, bool is_cuda)
{
    std::vector<torch::jit::IValue> inputs = {X, context, static_cast<int64_t>(n_g)};
    auto t0 = std::chrono::steady_clock::now();
    torch::Tensor out = module.forward(inputs).toTensor();
    if (is_cuda) torch::cuda::synchronize();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage : gnn_dense_msgpassing_probe <sparse_module.pt> "
                     "<dense_module.pt> [device=cuda] [n_g=20] [N_z=50] [m=5] "
                     "[n_iters=2000] [n_warmup=50]\n";
        return 1;
    }
    std::string sparse_path = argv[1];
    std::string dense_path  = argv[2];
    std::string device_str  = (argc > 3) ? argv[3] : "cuda";
    int n_g      = (argc > 4) ? std::stoi(argv[4]) : 20;
    int N_z      = (argc > 5) ? std::stoi(argv[5]) : 50;
    int m        = (argc > 6) ? std::stoi(argv[6]) : 5;
    int n_iters  = (argc > 7) ? std::stoi(argv[7]) : 2000;
    int n_warmup = (argc > 8) ? std::stoi(argv[8]) : 50;

    if (device_str == "cuda" && !torch::cuda::is_available()) {
        std::cerr << "[probe] CUDA indisponible.\n";
        return 1;
    }
    torch::Device device(device_str == "cuda" ? torch::kCUDA : torch::kCPU);
    bool is_cuda = device.is_cuda();

    int V        = N_z * n_g;
    int node_dim = 7 * m + 5;
    std::cout << "[probe] n_g=" << n_g << " N_z=" << N_z << " m=" << m
              << " node_dim=" << node_dim << " V=" << V << " device=" << device_str << "\n";

    torch::jit::script::Module sparse_module = torch::jit::load(sparse_path, device);
    torch::jit::script::Module dense_module  = torch::jit::load(dense_path, device);
    sparse_module.eval();
    dense_module.eval();
    torch::NoGradGuard no_grad;

    std::vector<int64_t> src, dst; std::vector<float> ew;
    build_sdedit_edges(n_g, N_z, src, dst, ew);
    int64_t E = static_cast<int64_t>(src.size());
    std::cout << "[probe] E=" << E << " arêtes (directed, sd_edit)\n";

    torch::Tensor edge_index = torch::stack({
        torch::from_blob(src.data(), {E}, torch::kInt64).clone(),
        torch::from_blob(dst.data(), {E}, torch::kInt64).clone()}).to(device);
    torch::Tensor edge_weight = torch::from_blob(ew.data(), {E}, torch::kFloat32).clone().to(device);

    auto rand_X   = [&]() { return torch::rand({V, node_dim}, torch::kFloat32); };
    auto rand_ctx = [&]() { return torch::rand({9}, torch::kFloat32); };

    // ── Sanity check : mêmes données, les deux modules doivent converger ────
    {
        torch::Tensor X   = rand_X().to(device);
        torch::Tensor ctx = rand_ctx().to(device);
        std::vector<torch::jit::IValue> in_sparse = {X, edge_index, edge_weight, ctx,
                                                       static_cast<int64_t>(n_g)};
        std::vector<torch::jit::IValue> in_dense  = {X, ctx, static_cast<int64_t>(n_g)};
        torch::Tensor out_sparse = sparse_module.forward(in_sparse).toTensor();
        torch::Tensor out_dense  = dense_module.forward(in_dense).toTensor();
        double max_diff = (out_sparse - out_dense).abs().max().item<double>();
        std::cout << "[probe] Sanity check (sparse vs dense, mêmes données) : max|diff|=" << max_diff
                   << (max_diff < 1e-3 ? "  [OK]" : "  [SUSPECT]") << "\n";
    }

    // ── Chronométrage : nouvelles données à chaque appel (comme en prod) ────
    std::vector<double> t_sparse, t_dense;
    t_sparse.reserve(n_iters); t_dense.reserve(n_iters);
    for (int it = 0; it < n_iters; ++it) {
        torch::Tensor X_cpu   = rand_X();
        torch::Tensor ctx_cpu = rand_ctx();
        torch::Tensor X   = X_cpu.to(device);
        torch::Tensor ctx = ctx_cpu.to(device);
        t_sparse.push_back(time_forward_sparse(sparse_module, X, edge_index, edge_weight, ctx, n_g, is_cuda));
    }
    for (int it = 0; it < n_iters; ++it) {
        torch::Tensor X_cpu   = rand_X();
        torch::Tensor ctx_cpu = rand_ctx();
        torch::Tensor X   = X_cpu.to(device);
        torch::Tensor ctx = ctx_cpu.to(device);
        t_dense.push_back(time_forward_dense(dense_module, X, ctx, n_g, is_cuda));
    }

    std::vector<double> t_sparse_steady(t_sparse.begin() + std::min(n_warmup, n_iters - 1), t_sparse.end());
    std::vector<double> t_dense_steady(t_dense.begin() + std::min(n_warmup, n_iters - 1), t_dense.end());

    std::cout << "\n[probe] === Sparse (référence, production actuelle) ===\n";
    print_stats("forward sparse (régime stable, " + std::to_string(t_sparse_steady.size()) + " appels)",
                compute_stats(t_sparse_steady));
    std::cout << "\n[probe] === Dense (nouvelle agrégation, mêmes poids) ===\n";
    print_stats("forward dense (régime stable, " + std::to_string(t_dense_steady.size()) + " appels)",
                compute_stats(t_dense_steady));

    Stats ss = compute_stats(t_sparse_steady);
    Stats sd = compute_stats(t_dense_steady);
    std::cout << "\n[probe] === Résumé ===\n";
    std::cout << "  Sparse (production actuelle) : " << ss.mean * 1000.0 << " ms/appel\n";
    std::cout << "  Dense  (nouvelle agrégation)  : " << sd.mean * 1000.0 << " ms/appel\n";
    std::cout << "  Gain                          : x" << (ss.mean / sd.mean) << "\n";
    return 0;
}
