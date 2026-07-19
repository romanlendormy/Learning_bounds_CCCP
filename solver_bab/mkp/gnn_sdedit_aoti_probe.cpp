/*
 * gnn_sdedit_aoti_probe.cpp — Sonde diagnostique isolée : appelle
 * GNNSdEditNativeAOTI::query() directement à la racine (x_mask tout libre,
 * depth=0), sans passer par Gecode/le moteur B&B — pour isoler si le hang
 * observé dans le vrai B&B (gnn_backend=native_aoti, TIMEOUT à 0 nœuds) vient
 * de query() lui-même ou de l'intégration Gecode/propagateur. Même pattern
 * que gnn_native_probe.cpp (GNNV1LDNative), adapté à GNNSdEditNativeAOTI.
 *
 * Usage : gnn_sdedit_aoti_probe <dataset.txt> <ckpt.pt> [epsilon=0.10]
 *         [device=cuda] [rep=0] [n_iters=3]
 */
#include "gnn_sd_edit_native_aoti.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
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

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage : gnn_sdedit_aoti_probe <dataset.txt> <ckpt.pt> "
                     "[epsilon=0.10] [device=cuda] [rep=0] [n_iters=3]\n";
        return 1;
    }
    std::string dataset = argv[1];
    std::string ckpt    = argv[2];
    double epsilon      = (argc > 3) ? std::stod(argv[3]) : 0.10;
    std::string device  = (argc > 4) ? argv[4] : "cuda";
    int rep             = (argc > 5) ? std::stoi(argv[5]) : 0;
    int n_iters         = (argc > 6) ? std::stoi(argv[6]) : 3;

    MKPInst inst = load_song_local(dataset, rep);
    std::cout << "[probe] n=" << inst.n << " m=" << inst.m << " N=" << inst.N
              << " epsilon=" << epsilon << " device=" << device << "\n";

    std::cout << "[probe] construction GNNSdEditNativeAOTI (chargement/export artefacts)...\n" << std::flush;
    GNNSdEditNativeAOTI backend(ckpt, device, inst.n, inst.m, inst.N, epsilon,
                                inst.c, inst.p, inst.A, inst.b);
    std::cout << "[probe] backend prêt.\n" << std::flush;

    std::vector<int8_t> x_mask_full(inst.n, 2);  // tout libre => n_g = n, comme la racine Gecode

    std::cout << std::fixed << std::setprecision(6);
    for (int iter = 0; iter < n_iters; ++iter) {
        std::cout << "[probe] iter=" << iter << " appel query() (curr_cost=0.0, depth=0)...\n" << std::flush;
        auto t0 = std::chrono::steady_clock::now();
        GNNCallTiming timing;
        // x_mask_full ne fixe rien à 1 => b_res == inst.b (capacité pleine).
        double ub_relative = backend.query(x_mask_full, inst.b, /*curr_cost=*/0.0, /*depth=*/0, timing);
        auto t1 = std::chrono::steady_clock::now();
        double wall = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "[probe] iter=" << iter << " OK ub_relative=" << ub_relative
                  << " wall=" << wall << "s"
                  << " t_graph_build_s=" << timing.t_graph_build_s
                  << " t_forward_s=" << timing.t_forward_s
                  << " t_bound_compute_s=" << timing.t_bound_compute_s << "\n" << std::flush;
    }
    return 0;
}
