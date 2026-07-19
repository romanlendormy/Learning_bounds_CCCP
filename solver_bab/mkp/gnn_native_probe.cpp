/*
 * gnn_native_probe.cpp — Sonde diagnostique : appelle GNNV1LDNative::query()
 * directement à la racine (x_mask tout libre, depth=0), sans passer par
 * Gecode/le moteur B&B — pour valider le portage torch::jit isolément
 * (comparaison à la référence Python, cf. session "load train checkpoint
 * into GECODE BAB"). Pas un binaire de production.
 *
 * n_g = nombre d'items actifs (x_mask == libre) : à la racine (x_mask tout
 * libre), n_g = n, donc le graphe est à sa taille MAXIMALE (edges scénario-
 * scénario par item ~ O(n_g * N^2)). Les nœuds B&B plus profonds ont un
 * graphe plus petit (n_g < n) donc un coût réel ≤ à celui mesuré ici — cette
 * sonde donne un majorant conservateur du coût par nœud, pas une moyenne.
 *
 * n_iters/n_warmup permettent de distinguer le coût "à froid" (première
 * init CUDA : contexte, premiers lancements de kernels, cache cuBLAS) du
 * régime stable, dilué sur trop peu d'appels dans un run B&B court.
 *
 * Usage : gnn_native_probe <dataset.txt> <ckpt.pt> [epsilon=0.10] [device=cpu]
 *         [rep=0] [n_iters=5] [n_warmup=0]
 */
#include "gnn_v1ld_native.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

struct MKPInst {
    Vec1d c; Vec1d p; Vec3d A; Vec2d b;
    int n, m, N;
};

static MKPInst load_song_local(const std::string& fp, int rep = 0)
{
    std::ifstream f(fp);
    if (!f) { std::cerr << "Erreur : " << fp << "\n"; std::exit(1); }

    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string tok;
        std::vector<double> vals;
        while (std::getline(iss, tok, ',')) vals.push_back(std::stod(tok));
        if (static_cast<int>(vals[4]) != rep) continue;
        rows.push_back(std::move(vals));
    }
    if (rows.empty()) { std::cerr << "Aucun scénario pour rep=" << rep << "\n"; std::exit(1); }

    int m = static_cast<int>(rows[0][2]);
    int n = static_cast<int>(rows[0][3]);
    int N = static_cast<int>(rows.size());

    Vec3d A(N, Vec2d(m, Vec1d(n, 0.0)));
    Vec2d b(N, Vec1d(m, 0.0));
    Vec1d c(n, 0.0);

    for (int i = 0; i < N; ++i) {
        const auto& v = rows[i];
        for (int j = 0; j < m; ++j)
            for (int k = 0; k < n; ++k)
                A[i][j][k] = v[5 + j * n + k];
        for (int j = 0; j < m; ++j)
            b[i][j] = v[5 + m * n + j];
        if (i == 0)
            for (int k = 0; k < n; ++k)
                c[k] = v[5 + m * n + m + k];
    }
    Vec1d p(N, 1.0 / N);
    return {c, p, A, b, n, m, N};
}

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

} // namespace

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage : gnn_native_probe <dataset.txt> <ckpt.pt> "
                     "[epsilon=0.10] [device=cpu] [rep=0] [n_iters=5] [n_warmup=0] "
                     "[vary_ng=0]\n"
                     "  vary_ng>0 : alterne le masque toutes les vary_ng itérations entre "
                     "n_g=n (tout libre) et n_g=n/2 (moitié fixée à 0), pour tester si un "
                     "changement de forme (n_g) redéclenche la compilation TorchScript.\n";
        return 1;
    }
    std::string dataset = argv[1];
    std::string ckpt    = argv[2];
    double epsilon      = (argc > 3) ? std::stod(argv[3]) : 0.10;
    std::string device  = (argc > 4) ? argv[4] : "cpu";
    int rep             = (argc > 5) ? std::stoi(argv[5]) : 0;
    int n_iters         = (argc > 6) ? std::stoi(argv[6]) : 5;
    int n_warmup        = (argc > 7) ? std::stoi(argv[7]) : 0;
    int vary_ng         = (argc > 8) ? std::stoi(argv[8]) : 0;

    MKPInst inst = load_song_local(dataset, rep);
    std::cout << "[probe] n=" << inst.n << " m=" << inst.m << " N=" << inst.N
              << " epsilon=" << epsilon << " device=" << device
              << " n_iters=" << n_iters << " n_warmup=" << n_warmup << "\n";

    std::cout << "[probe] construction GNNV1LDNative (chargement checkpoint)...\n" << std::flush;
    GNNV1LDNative backend(ckpt, device, inst.n, inst.m, inst.N, epsilon,
                          inst.c, inst.p, inst.A, inst.b);
    std::cout << "[probe] backend prêt.\n" << std::flush;

    std::vector<int8_t> x_mask_full(inst.n, 2);       // tout libre => n_g = n
    std::vector<int8_t> x_mask_half(inst.n, 2);
    for (int k = inst.n / 2; k < inst.n; ++k) x_mask_half[k] = 0;  // n_g = n/2
    std::cout << std::fixed << std::setprecision(6);

    std::vector<double> t_graph, t_fwd, t_bound;
    t_graph.reserve(n_iters); t_fwd.reserve(n_iters); t_bound.reserve(n_iters);

    for (int iter = 0; iter < n_iters; ++iter) {
        bool use_half = vary_ng > 0 && (iter / vary_ng) % 2 == 1;
        const std::vector<int8_t>& x_mask = use_half ? x_mask_half : x_mask_full;
        int n_g = use_half ? (inst.n - inst.n / 2) : inst.n;

        GNNCallTiming timing;
        // x_mask_full/x_mask_half ne fixent jamais rien à 1 (seulement 0/libre)
        // => b_res == inst.b (capacité pleine).
        double ub_relative = backend.query(x_mask, inst.b, /*curr_cost=*/0.0, /*depth=*/0, timing);
        std::cout << "[probe] iter=" << iter << " n_g=" << n_g << " ub_relative=" << ub_relative
                  << " t_graph_build_s=" << timing.t_graph_build_s
                  << " t_forward_s=" << timing.t_forward_s
                  << " t_bound_compute_s=" << timing.t_bound_compute_s << "\n";
        t_graph.push_back(timing.t_graph_build_s);
        t_fwd.push_back(timing.t_forward_s);
        t_bound.push_back(timing.t_bound_compute_s);
    }
    std::cout << std::flush;

    if (n_warmup > 0 && n_warmup < n_iters) {
        std::vector<double> f_all(t_fwd.begin(), t_fwd.begin() + n_warmup);
        std::vector<double> f_steady(t_fwd.begin() + n_warmup, t_fwd.end());
        std::vector<double> g_steady(t_graph.begin() + n_warmup, t_graph.end());
        std::vector<double> b_steady(t_bound.begin() + n_warmup, t_bound.end());
        std::cout << "\n[probe] === résumé (n=" << inst.n << " N=" << inst.N
                  << " device=" << device << ") ===\n";
        print_stats("forward (warm-up, " + std::to_string(n_warmup) + " 1ers appels)",
                    compute_stats(f_all));
        print_stats("forward (régime stable, " + std::to_string(n_iters - n_warmup) + " appels)",
                    compute_stats(f_steady));
        print_stats("graph_build (régime stable)", compute_stats(g_steady));
        print_stats("bound_compute (régime stable)", compute_stats(b_steady));
    }
    return 0;
}
