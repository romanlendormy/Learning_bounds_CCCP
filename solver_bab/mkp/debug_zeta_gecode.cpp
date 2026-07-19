/*
 * debug_zeta_gecode.cpp — diagnostic temporaire : trouve le premier scénario
 * i où solve_zeta_gecode diverge de MKPProblem::solve_zeta (CP-SAT, référence),
 * imprime tout (Ai, bi, cost, x des deux méthodes, vérif manuelle de
 * faisabilité + valeur) pour localiser la cause exacte.
 *
 * Usage : debug_zeta_gecode <dataset.txt> <rep> [epsilon=0.10]
 */
#include "mkp_problem.hpp"
#include "zeta_gecode_solver.hpp"

#include <cmath>
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
        for (int j = 0; j < m; ++j) b[i][j] = v[5 + m * n + j];
        if (i == 0) for (int k = 0; k < n; ++k) c[k] = v[5 + m * n + m + k];
    }
    Vec1d p(N, 1.0 / N);
    return {c, p, A, b, n, m, N};
}

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "Usage: debug_zeta_gecode <dataset.txt> <rep> [epsilon] [target_i]\n"; return 1; }
    std::string path = argv[1];
    int rep = std::atoi(argv[2]);
    double epsilon = (argc > 3) ? std::atof(argv[3]) : 0.10;
    int target_i = (argc > 4) ? std::atoi(argv[4]) : -1;

    MKPInst inst = load_song_local(path, rep);
    MKPProblem prob_cpsat(inst.c, inst.p, inst.A, inst.b, epsilon);
    const Vec1d& cost = inst.c;

    std::cout << std::fixed << std::setprecision(6);
    int i_start = (target_i >= 0) ? target_i : 0;
    int i_end   = (target_i >= 0) ? target_i + 1 : inst.N;
    for (int i = i_start; i < i_end; ++i) {
        auto [vc, xc] = prob_cpsat.solve_zeta(i, cost);
        auto [vg, xg] = solve_zeta_gecode(inst.n, inst.m, inst.A[i], inst.b[i], cost);
        double diff = std::abs(vc - vg);
        if (diff > 1e-6) {
            std::cout << "=== Divergence a i=" << i << " ===\n";
            std::cout << "v_cpsat=" << vc << "  v_gecode=" << vg << "  diff=" << diff << "\n";

            std::cout << "cost = ["; for (int k=0;k<inst.n;++k) std::cout << cost[k] << (k+1<inst.n?",":""); std::cout << "]\n";
            for (int j = 0; j < inst.m; ++j) {
                std::cout << "A[" << j << "] = ["; for (int k=0;k<inst.n;++k) std::cout << inst.A[i][j][k] << (k+1<inst.n?",":""); std::cout << "]  b=" << inst.b[i][j] << "\n";
            }

            std::cout << "x_cpsat  = ["; for (int k=0;k<inst.n;++k) std::cout << (int)xc[k] << (k+1<inst.n?",":""); std::cout << "]\n";
            std::cout << "x_gecode = ["; for (int k=0;k<inst.n;++k) std::cout << (int)xg[k] << (k+1<inst.n?",":""); std::cout << "]\n";

            // Vérif manuelle de la solution gecode : valeur + faisabilité
            double manual_val = 0.0;
            bool feasible = true;
            for (int k = 0; k < inst.n; ++k) manual_val += xg[k] * cost[k];
            for (int j = 0; j < inst.m; ++j) {
                double used = 0.0;
                for (int k = 0; k < inst.n; ++k) used += xg[k] * inst.A[i][j][k];
                if (used > inst.b[i][j] + 1e-6) { feasible = false; std::cout << "  VIOLATION dim " << j << " : used=" << used << " > b=" << inst.b[i][j] << "\n"; }
            }
            std::cout << "recalcul manuel valeur(x_gecode) = " << manual_val << "  faisable=" << (feasible?"OUI":"NON") << "\n";

            double manual_val_c = 0.0;
            for (int k = 0; k < inst.n; ++k) manual_val_c += xc[k] * cost[k];
            std::cout << "recalcul manuel valeur(x_cpsat)  = " << manual_val_c << "\n";
            return 1;
        }
    }
    std::cout << "Aucune divergence sur " << inst.N << " scenarios.\n";
    return 0;
}
