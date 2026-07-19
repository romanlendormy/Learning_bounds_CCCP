/*
 * main_bab_bucketed.cpp — CLI pour solve_gecode_bab_mkp_bucketed (variante
 * expérimentale : bucketing des nœuds fils par n_g avant pruning/incumbent,
 * borne inchangée V1LD/V1CLD/REF — cf. gecode_bab_bucketed.hpp).
 *
 * Usage :
 *   bab_mkp_gecode_bucketed <dataset.txt> [epsilon=0.1] [T_node=30]
 *       [time_limit=-1] [replication=0] [T_root=200] [mode=v1ld|v1cld|ref] [K=1]
 *
 * time_limit <= 0 : illimité — utile pour obtenir le nombre de nœuds RÉEL
 * nécessaire à la preuve d'optimalité (pas tronqué par un timeout), cf. plan.
 */

#include "gecode_bab_bucketed.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ── Loader (copie locale, même convention que main_gecode.cpp) ──────────────

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
        std::cerr << "Usage : bab_mkp_gecode_bucketed <dataset.txt> "
                     "[epsilon=0.1] [T_node=30] [time_limit=-1] [rep=0] "
                     "[T_root=200] [mode=v1ld|v1cld|ref] [K=1]\n";
        return 1;
    }

    std::string dataset     = argv[1];
    double      epsilon     = (argc > 2) ? std::stod(argv[2]) : 0.10;
    int         T_node      = (argc > 3) ? std::stoi(argv[3]) : 30;
    double      time_limit  = (argc > 4) ? std::stod(argv[4]) : -1.0;
    int         replication = (argc > 5) ? std::stoi(argv[5]) : 0;
    int         T_root      = (argc > 6) ? std::stoi(argv[6])  : 200;
    std::string mode_str    = (argc > 7) ? argv[7] : "v1ld";
    BoundMode   bound_mode  = (mode_str == "v1cld") ? BoundMode::V1CLD
                            : (mode_str == "ref")   ? BoundMode::REF
                                                     : BoundMode::V1LD;
    int         K           = (argc > 8) ? std::stoi(argv[8]) : 1;

    MKPInst inst = load_song_local(dataset, replication);

    std::cout << "═══════════════════════════════════════════════════\n";
    std::cout << "  CCMKP — Gecode BAB BUCKETED (K=" << K << ") + borne "
              << (bound_mode == BoundMode::V1LD  ? "V1LD (ζ exact B&B)"
                : bound_mode == BoundMode::V1CLD ? "V1CLD (ζ DP 1D exact)"
                                                  : "REF (squelette nu, sans borne)")
              << "\n";
    std::cout << "═══════════════════════════════════════════════════\n";
    std::cout << "  Fichier : " << dataset << "\n";
    std::cout << "  n="   << inst.n << "  m=" << inst.m
              << "  N="   << inst.N << "  ε=" << epsilon
              << "  rep=" << replication << "\n";
    std::cout << "  T_root=" << T_root << "  T_node=" << T_node
              << "  time_limit=" << time_limit << "s"
              << "  mode=" << mode_str << "  K=" << K << "\n";
    std::cout << "───────────────────────────────────────────────────\n";

    BucketedBabResult res = solve_gecode_bab_mkp_bucketed(
        inst.n, inst.m, inst.N, epsilon,
        inst.c, inst.p, inst.A, inst.b,
        time_limit, T_root, T_node, bound_mode,
        /*patience_mode=*/false, /*patience_tol=*/0.1, K);

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
    std::cout << "  Améliorations incumbent   : " << res.incumbent_improvements << "\n";
    std::cout << "  Appels more()             : " << res.more_calls << "\n\n";

    std::cout << "  Flushes de bucket         : " << res.n_bucket_flushes << "\n";
    std::cout << std::setprecision(2);
    std::cout << "  Remplissage moyen au flush: " << res.mean_bucket_fill_at_flush
              << " / K=" << K << "\n";
    std::cout << std::setprecision(0);
    std::cout << "  Pic de nœuds en attente   : " << res.max_pending_live << "\n\n";

    std::cout << std::setprecision(3);
    std::cout << "  Temps total               : " << res.solve_time_s << " s\n";
    std::cout << "  Temps borne total         : " << res.t_bound_total_s << " s\n";
    std::cout << "═══════════════════════════════════════════════════\n";

    return 0;
}
