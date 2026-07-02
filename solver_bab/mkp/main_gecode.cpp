/*
 * main_gecode.cpp — CLI pour le B&B Gecode + borne V1LD.
 *
 * Usage :
 *   bab_mkp_gecode  <dataset.txt>  [epsilon]  [T_node]  [time_limit_s]  [replication]
 *
 * Exemple :
 *   bab_mkp_gecode  ../data/knapsack/datasets-song/ccmk-song-mknap1_4-S20.txt
 *   bab_mkp_gecode  ../data/knapsack/datasets-song/ccmk-song-mknap1_4-S20.txt 0.1 30 120 2
 */

#include "gecode_bab.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ── Loader (copie locale pour éviter la dépendance sur bab_mkp.cpp) ──────────

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

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage : bab_mkp_gecode <dataset.txt> "
                     "[epsilon=0.1] [T_node=30] [time_limit=120]\n";
        return 1;
    }

    std::string dataset     = argv[1];
    double      epsilon     = (argc > 2) ? std::stod(argv[2]) : 0.10;
    int         T_node      = (argc > 3) ? std::stoi(argv[3]) : 30;
    double      time_limit  = (argc > 4) ? std::stod(argv[4]) : 120.0;
    int         replication = (argc > 5) ? std::stoi(argv[5]) : 0;
    int         T_root      = (argc > 6) ? std::stoi(argv[6])  : 200;
    // mode : "v1ld" (défaut) ou "v1cld"
    std::string mode_str    = (argc > 7) ? argv[7] : "v1ld";
    BoundMode   bound_mode  = (mode_str == "v1cld") ? BoundMode::V1CLD : BoundMode::V1LD;
    // patience : "patience" active le mode convergence, sinon itérations fixes
    bool        patience    = (argc > 8) && std::string(argv[8]) == "patience";
    double      patience_tol = (argc > 9) ? std::stod(argv[9]) : 0.1;

    MKPInst inst = load_song_local(dataset, replication);

    std::cout << "═══════════════════════════════════════════════════\n";
    std::cout << "  CCMKP — Gecode BAB + borne "
              << (bound_mode == BoundMode::V1LD ? "V1LD (ζ exact B&B)"
                                                : "V1CLD (ζ DP 1D exact)")
              << "\n";
    std::cout << "═══════════════════════════════════════════════════\n";
    std::cout << "  Fichier : " << dataset << "\n";
    std::cout << "  n="   << inst.n << "  m=" << inst.m
              << "  N="   << inst.N << "  ε=" << epsilon
              << "  rep=" << replication << "\n";
    std::cout << "  T_root=" << T_root << "  T_node=" << T_node
              << "  time_limit=" << time_limit << "s"
              << "  mode=" << mode_str
              << (patience ? ("  patience=oui(tol=" + std::to_string(patience_tol) + ")") : "  patience=non")
              << "\n";
    std::cout << "───────────────────────────────────────────────────\n";

    GecodeV1LDResult res = solve_gecode_bab_mkp(
        inst.n, inst.m, inst.N, epsilon,
        inst.c, inst.p, inst.A, inst.b,
        time_limit, T_root, T_node, bound_mode,
        patience, patience_tol);

    std::cout << "\n═══════════════════════════════════════════════════\n";
    std::cout << "  RÉSULTATS\n";
    std::cout << "═══════════════════════════════════════════════════\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Statut                    : " << res.status        << "\n";
    std::cout << "  Valeur optimale           : " << res.objective     << "\n";
    std::cout << "  UB finale                 : " << res.best_open_ub   << "\n";
    std::cout << "  Incumbent initial (greedy): " << res.greedy_value  << "\n\n";

    // Nombre de nœuds dans l'arbre naïf (énumération complète sur x) : 2^n
    long long naive_nodes = 1LL << inst.n;  // 2^n feuilles

    std::cout << std::setprecision(0);
    std::cout << "  Nœuds Gecode              : " << res.nodes
              << " / " << naive_nodes << " naïf"
              << "  (" << std::fixed << std::setprecision(2)
              << static_cast<double>(res.nodes) / naive_nodes * 100.0 << "%)\n";

    std::cout << std::setprecision(0);
    std::cout << "  Profondeur max            : " << res.depth
              << " / " << (inst.n + inst.N) << " max théorique\n\n";

    std::cout << "  Échecs totaux             : " << res.fails      << "\n";
    std::cout << "    dont borne V1LD         : " << res.fails_from_bound
              << "  (" << std::fixed << std::setprecision(1)
              << (res.fails > 0 ? 100.0 * res.fails_from_bound / res.fails : 0.0)
              << "% des fails)\n";
    std::cout << "    dont propagation CP     : " << res.fails_from_cp
              << "  (" << std::fixed << std::setprecision(1)
              << (res.fails > 0 ? 100.0 * res.fails_from_cp / res.fails : 0.0)
              << "% des fails)\n\n";

    std::cout << "  Améliorations incumbent             : " << res.incumbent_improvements << "\n";
    std::cout << "  Appels more() (commits réussis)     : " << res.more_calls << "\n";
    std::cout << "  UB V1LD utile (réduit obj.max)      : " << res.ub_tightened
              << " / " << res.more_calls
              << "  (" << std::fixed << std::setprecision(1)
              << (res.more_calls > 0 ? 100.0 * res.ub_tightened / res.more_calls : 0.0)
              << "% des appels)\n\n";

    std::cout << std::setprecision(3);
    std::cout << "  Temps total               : " << res.solve_time_s     << " s\n";
    std::cout << "  Temps borne root          : " << res.root_bound_time_s << " s"
              << "  (T_root=" << T_root << " iters)\n";
    std::cout << "  Temps borne total         : " << res.t_bound_total_s   << " s"
              << "  (" << std::fixed << std::setprecision(1)
              << (res.solve_time_s > 0 ? 100.0 * res.t_bound_total_s / res.solve_time_s : 0.0)
              << "% du total)\n";
    std::cout << "  Temps BAB pur             : " << res.t_bab_pure_s      << " s"
              << "  (" << std::fixed << std::setprecision(1)
              << (res.solve_time_s > 0 ? 100.0 * res.t_bab_pure_s / res.solve_time_s : 0.0)
              << "% du total)\n";
    if (res.nodes > 0)
        std::cout << "  Temps moyen par nœud     : "
                  << std::setprecision(3)
                  << res.solve_time_s / res.nodes * 1000.0 << " ms\n";
    std::cout << "═══════════════════════════════════════════════════\n";

    return 0;
}
