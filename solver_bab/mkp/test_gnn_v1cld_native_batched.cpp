/*
 * test_gnn_v1cld_native_batched.cpp — Validation autonome de
 * GNNV1CLDSdEditNativeBatched AVANT tout branchement dans le vrai B&B
 * (cf. plan : valider la borne du backend C++ contre lagrangian_v1cld en
 * Python sur quelques instances, comme le projet l'a systématiquement fait
 * pour les autres backends GNN de ce dossier).
 *
 * Construit UN nœud B&B réaliste (racine : x_mask/z_mask entièrement
 * libres, b_res = b) depuis une instance réelle, appelle query_batch(B=1),
 * et imprime : n_g, N_z, m, c_scale, le vecteur λ (après rescale) et la
 * borne L1(λ) obtenue — à comparer avec un script Python compagnon
 * (test_paper/validate_v1cld_native_reference.py) qui recalcule EXACTEMENT
 * la même chose via train_v1CLD_sd_edit.py (build_graph + GNN + lagrangian_v1cld),
 * sur le MÊME fichier d'instance/checkpoint.
 *
 * Usage :
 *   test_gnn_v1cld_native_batched <dataset.txt> <checkpoint.pt> [device=cpu]
 */

#include "gnn_v1cld_sd_edit_native_batched.hpp"

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

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "Usage: test_gnn_v1cld_native_batched <dataset.txt> <checkpoint.pt> [device=cpu]\n";
        return 1;
    }
    std::string dataset = argv[1];
    std::string checkpoint = argv[2];
    std::string device = (argc > 3) ? argv[3] : "cpu";

    MKPInst inst = load_song_local(dataset, 0);
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "n=" << inst.n << " m=" << inst.m << " N=" << inst.N << "\n";

    GNNV1CLDSdEditNativeBatched backend(checkpoint, device, inst.n, inst.m, inst.N, 0.10,
                                         inst.c, inst.p, inst.A, inst.b, false);

    // Nœud racine : x/z entièrement libres, b_res = b, curr_cost = 0, depth = 0.
    std::vector<std::vector<int8_t>> x_masks(1, std::vector<int8_t>(inst.n, 2));
    std::vector<double> curr_costs = {0.0};
    std::vector<Vec2d> b_res_list = {inst.b};

    std::vector<double> ub_relative;
    GNNBatchCallTiming timing;
    backend.query_batch(x_masks, curr_costs, b_res_list, 0, ub_relative, timing);

    std::cout << "ub_relative (= L1(lambda_GNN)) = " << ub_relative[0] << "\n";
    std::cout << "t_graph_build_s=" << timing.t_graph_build_s
              << " t_forward_s=" << timing.t_forward_s
              << " t_bound_compute_s=" << timing.t_bound_compute_s << "\n";
    return 0;
}
