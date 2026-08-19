/*
 * main_bab_v1cld_gnn_batched.cpp — CLI pour solve_gecode_bab_mkp_v1cld_gnn_batched
 * (port V1CLD de main_bab_gnn_batched.cpp/Étape 2e — cf. gecode_bab_v1cld_gnn_batched.hpp).
 *
 * Usage :
 *   bab_mkp_gecode_v1cld_gnn_batched <dataset.txt> <checkpoint.pt> [epsilon=0.1]
 *       [time_limit=-1] [rep=0] [device=cuda] [K=1]
 *
 * Comparer directement à bab_mkp_gecode_gnn_batched <dataset> <ckpt_v1ld> ...
 * — même instance, même n/m/N/epsilon, borne V1LD (batché) vs borne V1CLD
 * (batché, ce binaire).
 */
#include "gecode_bab_v1cld_gnn_batched.hpp"

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
        std::cerr << "Usage : bab_mkp_gecode_v1cld_gnn_batched <dataset.txt> <checkpoint.pt> "
                     "[epsilon=0.1] [time_limit=-1] [rep=0] [device=cuda] [K=1]\n";
        return 1;
    }
    std::string dataset   = argv[1];
    std::string ckpt_path = argv[2];
    double epsilon     = (argc > 3) ? std::stod(argv[3]) : 0.10;
    double time_limit  = (argc > 4) ? std::stod(argv[4]) : -1.0;
    int    rep         = (argc > 5) ? std::stoi(argv[5]) : 0;
    std::string device = (argc > 6) ? argv[6] : "cuda";
    int    K           = (argc > 7) ? std::stoi(argv[7]) : 1;

    MKPInst inst = load_song_local(dataset, rep);

    std::cout << "═══════════════════════════════════════════════════\n";
    std::cout << "  CCMKP — Gecode BAB V1CLD GNN BATCHÉ (K=" << K << ")\n";
    std::cout << "═══════════════════════════════════════════════════\n";
    std::cout << "  Fichier : " << dataset << "\n";
    std::cout << "  Checkpoint : " << ckpt_path << "\n";
    std::cout << "  n="   << inst.n << "  m=" << inst.m
              << "  N="   << inst.N << "  ε=" << epsilon
              << "  rep=" << rep << "\n";
    std::cout << "  time_limit=" << time_limit << "s  device=" << device << "  K=" << K << "\n";
    std::cout << "───────────────────────────────────────────────────\n";

    GNNBatchedBabResult res = solve_gecode_bab_mkp_v1cld_gnn_batched(
        inst.n, inst.m, inst.N, epsilon, inst.c, inst.p, inst.A, inst.b,
        time_limit, K, ckpt_path, device);

    std::cout << "\n═══════════════════════════════════════════════════\n";
    std::cout << "  RÉSULTATS\n";
    std::cout << "═══════════════════════════════════════════════════\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Statut                    : " << res.status    << "\n";
    std::cout << "  Valeur optimale           : " << res.objective << "\n";
    std::cout << "  Incumbent initial (greedy): " << res.greedy_value << "\n\n";

    std::cout << std::setprecision(0);
    std::cout << "  Nœuds                     : " << res.nodes << "\n";
    std::cout << "  Échecs totaux             : " << res.fails << "\n";
    std::cout << "  Améliorations incumbent   : " << res.incumbent_improvements << "\n\n";

    std::cout << "  Flushes de bucket         : " << res.n_bucket_flushes << "\n";
    std::cout << std::setprecision(2);
    std::cout << "  Remplissage moyen au flush: " << res.mean_bucket_fill_at_flush << " / K=" << K << "\n";
    std::cout << std::setprecision(0);
    std::cout << "  Pic de nœuds en attente   : " << res.max_pending_live << "\n\n";

    std::cout << std::setprecision(3);
    std::cout << "  Temps total               : " << res.solve_time_s << " s\n";
    std::cout << "  Temps borne (GNN batché)  : " << res.t_bound_total_s << " s"
              << "  (" << std::setprecision(1) << std::fixed
              << (res.solve_time_s > 0 ? 100.0 * res.t_bound_total_s / res.solve_time_s : 0.0)
              << "% du total)\n" << std::setprecision(3);
    std::cout << "    dont construction graphe: " << res.t_gnn_graph_build_s << " s\n";
    std::cout << "    dont forward GNN        : " << res.t_gnn_forward_s << " s\n";
    std::cout << "    dont calcul borne (ζ)   : " << res.t_gnn_bound_compute_s << " s\n";
    std::cout << "  Temps BAB pur             : " << res.t_bab_pure_s << " s\n";
    if (res.nodes > 0)
        std::cout << "  Temps moyen par nœud      : " << (res.solve_time_s / res.nodes * 1000.0) << " ms\n";
    std::cout << "═══════════════════════════════════════════════════\n";

    return 0;
}
