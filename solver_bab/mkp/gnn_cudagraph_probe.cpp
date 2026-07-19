/*
 * gnn_cudagraph_probe.cpp — Sonde de faisabilité : le forward TorchScript de
 * la borne GNN (module scripté, cf. gnn_v1ld_scriptable.py) est-il capturable
 * par CUDA Graphs (at::cuda::CUDAGraph), et si oui, quel gain réel par
 * rapport à l'appel eager actuel (production, cf. gnn_v1ld_native.cpp /
 * gnn_sd_edit_native.cpp) ?
 *
 * Contexte (cf. mémoire "sdedit-vs-homogene-forward-overhead-confirmed") :
 * bench_sdedit_vs_homogene_n20_N50.sh (job 15987340) a montré que diviser
 * les arêtes scénario-scénario par ~25x (clique -> étoile) ne réduit le
 * forward que de ~4.8%, alors que la construction de graphe (CPU,
 * proportionnelle aux arêtes) chute de 34%. Ça indique un coût dominé par
 * l'overhead FIXE de dispatch (~20-30 kernels CUDA séquentiels + transferts
 * H2D/D2H par appel), pas par le volume de message passing — cohérent avec
 * project_gnn_inference_forward_cost_benchmark (0.947ms à N=50 -> 1.212ms à
 * N=100, sous-linéaire). CUDA graphs amortit cet overhead en remplaçant
 * ~20-30 lancements de kernels par UN seul cudaGraphLaunch.
 *
 * Ne charge PAS de vraie instance MKP : seule la FORME des tenseurs compte
 * pour la mécanique/le chronométrage CUDA graphs, pas les valeurs numériques
 * de la borne. edge_index/edge_weight sont structurellement INVARIANTS pour
 * un (n_g, N_z) fixé (ne dépendent jamais de A/b/c, cf. gnn_v1ld_native.cpp
 * L154-172 et gnn_sd_edit_native.cpp L163-180) — reproduits ici à l'identique
 * et transférés UNE SEULE FOIS (hors chronométrage), pour les deux boucles
 * (eager et graph) : on isole ainsi le coût du forward proprement dit,
 * indépendamment de l'inefficacité actuelle (non liée à CUDA graphs) qui
 * reconstruit ces arêtes à chaque appel more() en production alors qu'elles
 * ne changent jamais pour un n_g donné.
 *
 * Usage : gnn_cudagraph_probe <scripted_module.pt> [device=cuda] [n_g=20]
 *         [N_z=50] [m=5] [n_iters_eager=2000] [n_warmup=50]
 *         [n_iters_graph=2000] [variant=homogene|sd_edit]
 */
#include <torch/script.h>
#include <torch/cuda.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

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

// Arêtes homogène (clique item + clique scénario) ou sd_edit (clique item +
// étoile scénario) — IDENTIQUE à gnn_v1ld_native.cpp / gnn_sd_edit_native.cpp,
// ne dépend que de (n_g, N_z), jamais des données numériques de l'instance.
void build_edges(int n_g, int N_z, const std::string& variant,
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
    if (variant == "homogene") {
        for (int kf = 0; kf < n_g; ++kf)
            for (int i1 = 0; i1 < N_z; ++i1)
                for (int i2 = i1 + 1; i2 < N_z; ++i2) {
                    int64_t a  = static_cast<int64_t>(i1) * n_g + kf;
                    int64_t b2 = static_cast<int64_t>(i2) * n_g + kf;
                    src.push_back(a); dst.push_back(b2); ew.push_back(1.0f / N_z);
                    src.push_back(b2); dst.push_back(a); ew.push_back(1.0f / N_z);
                }
    } else {
        for (int kf = 0; kf < n_g; ++kf) {
            int64_t anchor = static_cast<int64_t>(0) * n_g + kf;
            for (int i2 = 1; i2 < N_z; ++i2) {
                int64_t spoke = static_cast<int64_t>(i2) * n_g + kf;
                src.push_back(anchor); dst.push_back(spoke); ew.push_back(1.0f);
                src.push_back(spoke); dst.push_back(anchor); ew.push_back(1.0f);
            }
        }
    }
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage : gnn_cudagraph_probe <scripted_module.pt> [device=cuda] "
                     "[n_g=20] [N_z=50] [m=5] [n_iters_eager=2000] [n_warmup=50] "
                     "[n_iters_graph=2000] [variant=homogene|sd_edit]\n";
        return 1;
    }
    std::string module_path = argv[1];
    std::string device_str  = (argc > 2) ? argv[2] : "cuda";
    int n_g                 = (argc > 3) ? std::stoi(argv[3]) : 20;
    int N_z                 = (argc > 4) ? std::stoi(argv[4]) : 50;
    int m                   = (argc > 5) ? std::stoi(argv[5]) : 5;
    int n_iters_eager       = (argc > 6) ? std::stoi(argv[6]) : 2000;
    int n_warmup            = (argc > 7) ? std::stoi(argv[7]) : 50;
    int n_iters_graph       = (argc > 8) ? std::stoi(argv[8]) : 2000;
    std::string variant     = (argc > 9) ? argv[9] : "homogene";

    if (device_str == "cuda" && !torch::cuda::is_available()) {
        std::cerr << "[probe] CUDA indisponible.\n";
        return 1;
    }
    torch::Device device(device_str == "cuda" ? torch::kCUDA : torch::kCPU);

    int V        = N_z * n_g;
    int node_dim = 7 * m + 5;
    std::cout << "[probe] n_g=" << n_g << " N_z=" << N_z << " m=" << m
              << " node_dim=" << node_dim << " V=" << V << " variant=" << variant
              << " device=" << device_str << "\n";

    torch::jit::script::Module module = torch::jit::load(module_path, device);
    module.eval();
    torch::NoGradGuard no_grad;

    std::vector<int64_t> src, dst; std::vector<float> ew;
    build_edges(n_g, N_z, variant, src, dst, ew);
    int64_t E = static_cast<int64_t>(src.size());
    std::cout << "[probe] E=" << E << " arêtes (directed)\n";

    torch::Tensor edge_index = torch::stack({
        torch::from_blob(src.data(), {E}, torch::kInt64).clone(),
        torch::from_blob(dst.data(), {E}, torch::kInt64).clone()}).to(device);
    torch::Tensor edge_weight = torch::from_blob(ew.data(), {E}, torch::kFloat32).clone().to(device);

    auto rand_X   = [&]() { return torch::rand({V, node_dim}, torch::kFloat32); };
    auto rand_ctx = [&]() { return torch::rand({9}, torch::kFloat32); };

    // ── Baseline eager (chemin de production actuel) ────────────────────────
    // Chronométré : transfert H2D de X/context (nouvelles données à chaque
    // appel, comme en production) + forward. edge_index/edge_weight déjà sur
    // device, hors chronométrage (cf. commentaire d'en-tête).
    std::vector<double> t_eager;
    t_eager.reserve(n_iters_eager);
    for (int it = 0; it < n_iters_eager; ++it) {
        torch::Tensor X_cpu   = rand_X();
        torch::Tensor ctx_cpu = rand_ctx();
        auto t0 = std::chrono::steady_clock::now();
        torch::Tensor X   = X_cpu.to(device);
        torch::Tensor ctx = ctx_cpu.to(device);
        std::vector<torch::jit::IValue> inputs = {X, edge_index, edge_weight, ctx,
                                                    static_cast<int64_t>(n_g)};
        torch::Tensor out = module.forward(inputs).toTensor();
        torch::cuda::synchronize();
        auto t1 = std::chrono::steady_clock::now();
        t_eager.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    std::vector<double> t_eager_steady(t_eager.begin() + std::min(n_warmup, n_iters_eager - 1), t_eager.end());
    std::cout << "\n[probe] === Eager (chemin production actuel) ===\n";
    print_stats("forward eager (régime stable, " + std::to_string(t_eager_steady.size()) + " appels)",
                compute_stats(t_eager_steady));

    if (!device.is_cuda()) {
        std::cout << "\n[probe] device=cpu : CUDA graphs non applicable, arrêt ici.\n";
        return 0;
    }

    // ── Tentative de capture CUDA graph ──────────────────────────────────────
    std::cout << "\n[probe] === Tentative de capture CUDA graph ===\n";
    torch::Tensor X_static   = rand_X().to(device);
    torch::Tensor ctx_static = rand_ctx().to(device);
    torch::Tensor out_static;

    torch::cuda::synchronize();  // s'assurer que la création ci-dessus est bien terminée
    c10::cuda::CUDAStream capture_stream = c10::cuda::getStreamFromPool();
    at::cuda::CUDAGraph graph;
    bool capture_ok = false;
    std::string capture_error;
    try {
        c10::cuda::CUDAStreamGuard guard(capture_stream);
        // Warm-up sur le stream de capture AVANT capture_begin (requis : laisse
        // l'allocateur CUDA se stabiliser pour cette forme — cf. doc PyTorch sur
        // torch.cuda.graph).
        for (int i = 0; i < 3; ++i) {
            std::vector<torch::jit::IValue> inputs = {X_static, edge_index, edge_weight,
                                                       ctx_static, static_cast<int64_t>(n_g)};
            auto tmp = module.forward(inputs).toTensor();
        }
        capture_stream.synchronize();

        auto pool = at::cuda::graph_pool_handle();
        graph.capture_begin(pool);
        std::vector<torch::jit::IValue> inputs = {X_static, edge_index, edge_weight,
                                                   ctx_static, static_cast<int64_t>(n_g)};
        out_static = module.forward(inputs).toTensor();
        graph.capture_end();
        capture_ok = true;
    } catch (const std::exception& e) {
        capture_error = e.what();
    }

    if (!capture_ok) {
        std::cout << "  ÉCHEC de la capture : " << capture_error << "\n";
        std::cout << "  -> CUDA graphs n'est pas un simple wrapper ici, nécessiterait de "
                     "contourner l'interpréteur TorchScript (réimplémenter le forward en "
                     "ATen pur avec buffers persistants) avant d'envisager cette piste.\n";
        return 0;
    }
    std::cout << "  Capture réussie.\n";

    // Sanity check : un forward eager sur les MEMES données figées doit
    // correspondre exactement à ce que produit le premier replay.
    {
        std::vector<torch::jit::IValue> inputs = {X_static, edge_index, edge_weight,
                                                   ctx_static, static_cast<int64_t>(n_g)};
        torch::Tensor out_eager_ref = module.forward(inputs).toTensor();
        {
            c10::cuda::CUDAStreamGuard guard(capture_stream);
            graph.replay();
        }
        torch::cuda::synchronize();
        double max_diff = (out_static - out_eager_ref).abs().max().item<double>();
        std::cout << "  Sanity check (replay vs eager, mêmes données) : max|diff|=" << max_diff
                   << (max_diff < 1e-4 ? "  [OK]" : "  [SUSPECT]") << "\n";
    }

    // ── Replay pur (données figées, borne supérieure du gain) ────────────────
    std::vector<double> t_replay_static;
    t_replay_static.reserve(n_iters_graph);
    for (int it = 0; it < n_iters_graph; ++it) {
        auto t0 = std::chrono::steady_clock::now();
        {
            c10::cuda::CUDAStreamGuard guard(capture_stream);
            graph.replay();
        }
        torch::cuda::synchronize();
        auto t1 = std::chrono::steady_clock::now();
        t_replay_static.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    std::cout << "\n[probe] === Replay pur (données figées, borne sup. du gain) ===\n";
    print_stats("replay seul (" + std::to_string(n_iters_graph) + " appels)",
                compute_stats(t_replay_static));

    // ── Replay + rafraîchissement des données (scénario réaliste : nouvelles
    //    données à chaque nœud B&B ; edge_index/edge_weight inchangés, comme
    //    en pratique pour un n_g fixé) ────────────────────────────────────────
    std::vector<double> t_replay_refresh;
    t_replay_refresh.reserve(n_iters_graph);
    for (int it = 0; it < n_iters_graph; ++it) {
        torch::Tensor new_X_cpu   = rand_X();
        torch::Tensor new_ctx_cpu = rand_ctx();
        auto t0 = std::chrono::steady_clock::now();
        {
            c10::cuda::CUDAStreamGuard guard(capture_stream);
            X_static.copy_(new_X_cpu, /*non_blocking=*/true);
            ctx_static.copy_(new_ctx_cpu, /*non_blocking=*/true);
            graph.replay();
        }
        torch::cuda::synchronize();
        auto t1 = std::chrono::steady_clock::now();
        t_replay_refresh.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    std::cout << "\n[probe] === Replay + rafraîchissement des données (scénario réaliste) ===\n";
    print_stats("replay+refresh (" + std::to_string(n_iters_graph) + " appels)",
                compute_stats(t_replay_refresh));

    Stats se = compute_stats(t_eager_steady);
    Stats sr = compute_stats(t_replay_refresh);
    Stats sp = compute_stats(t_replay_static);
    std::cout << "\n[probe] === Résumé ===\n";
    std::cout << "  Eager (production actuelle)       : " << se.mean * 1000.0 << " ms/appel\n";
    std::cout << "  CUDA graph, replay pur (borne sup) : " << sp.mean * 1000.0 << " ms/appel\n";
    std::cout << "  CUDA graph, replay+refresh (réel)  : " << sr.mean * 1000.0 << " ms/appel\n";
    std::cout << "  Gain (réaliste)                    : x" << (se.mean / sr.mean) << "\n";
    return 0;
}
