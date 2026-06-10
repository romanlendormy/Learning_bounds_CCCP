/*
 * main_gecode_base.cpp — CLI Gecode CP pur (sans borne Lagrangienne).
 *
 * Usage :
 *   bab_mkp_gecode_base <dataset.txt> [epsilon=0.1] [time_limit_s=-1] [replication=0]
 */

#include "../gecode_base/gecode_solver.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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
    if (argc < 2) {
        std::cerr << "Usage : bab_mkp_gecode_base <dataset.txt> "
                     "[epsilon=0.1] [time_limit=-1] [replication=0]\n";
        return 1;
    }

    std::string dataset     = argv[1];
    double      epsilon     = (argc > 2) ? std::stod(argv[2]) : 0.10;
    double      time_limit  = (argc > 3) ? std::stod(argv[3]) : -1.0;
    int         replication = (argc > 4) ? std::stoi(argv[4]) : 0;

    MKPInst inst = load_song_local(dataset, replication);

    std::cout << "═══════════════════════════════════════════════════\n";
    std::cout << "  CCMKP — Gecode BAB CP pur (baseline)\n";
    std::cout << "═══════════════════════════════════════════════════\n";
    std::cout << "  Fichier : " << dataset << "\n";
    std::cout << "  n="   << inst.n << "  m=" << inst.m
              << "  N="   << inst.N << "  ε=" << epsilon
              << "  rep=" << replication << "\n";
    std::cout << "───────────────────────────────────────────────────\n";

    GecodeResult res = solve_gecode_mkp(
        inst.n, inst.m, inst.N, epsilon,
        inst.c, inst.p, inst.A, inst.b,
        time_limit);

    long long naive_nodes = 1LL << inst.n;
    int depth_max = inst.n + inst.N;

    std::cout << "\n═══════════════════════════════════════════════════\n";
    std::cout << "  RÉSULTATS\n";
    std::cout << "═══════════════════════════════════════════════════\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Statut                    : " << res.status    << "\n";
    std::cout << "  Valeur optimale           : " << res.objective << "\n\n";

    std::cout << std::setprecision(0);
    std::cout << "  Nœuds Gecode              : " << res.nodes
              << " / " << naive_nodes << " naïf"
              << "  (" << std::fixed << std::setprecision(2)
              << static_cast<double>(res.nodes) / naive_nodes * 100.0 << "%)\n";
    std::cout << std::setprecision(0);
    std::cout << "  Profondeur max            : " << res.depth
              << " / " << depth_max << " max théorique\n\n";

    std::cout << "  Échecs totaux             : " << res.fails      << "\n";
    std::cout << "  Améliorations incumbent   : " << res.incumbent_improvements << "\n\n";

    std::cout << std::setprecision(3);
    std::cout << "  Temps total               : " << res.solve_time_s << " s\n";
    if (res.nodes > 0)
        std::cout << "  Temps moyen par nœud     : "
                  << std::setprecision(3)
                  << res.solve_time_s / res.nodes * 1000.0 << " ms\n";
    std::cout << "═══════════════════════════════════════════════════\n";

    return 0;
}
