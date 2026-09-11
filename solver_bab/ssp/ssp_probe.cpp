/*
 * ssp_probe.cpp — Sonde rapide de faisabilité/difficulté d'une instance SSP.
 *
 * Charge une instance texte (cf. ssp_loader.hpp) et la résout avec Gecode
 * base (CP pur, size_min natif) sous un budget de temps donné. N'exécute
 * PAS DLD/SLD (trop coûteux pour balayer beaucoup d'instances) — sert
 * uniquement à classer une instance : INFEASIBLE / trivial / dur / TIMEOUT.
 *
 * Usage :
 *   ssp_probe <répertoire instance> <time_lim_s>
 *
 * Sortie : une ligne CSV sur stdout —
 *   dir,E,T,N,eps,Q,n_trans,status,nodes,fails,time_s,objective
 */

#include "ssp_loader.hpp"
#include "../../gecode_base/gecode_solver.hpp"

#include <algorithm>
#include <iostream>

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "Usage: ssp_probe <répertoire instance> <time_lim_s>\n";
        return 1;
    }
    std::string dir      = argv[1];
    double      time_lim = std::atof(argv[2]);

    SSPInstance inst = load_ssp_instance(dir);

    int Q = inst.initial + 1;
    for (auto& [src, sym, dst] : inst.transitions)
        Q = std::max({Q, src + 1, dst + 1});
    for (int q : inst.accepting)
        Q = std::max(Q, q + 1);
    const int m = 4;

    std::vector<std::vector<int>> delta(Q, std::vector<int>(m, -1));
    for (auto& [src, sym, dst] : inst.transitions) delta[src][sym] = dst;
    std::vector<int> work_ind(m, 0);
    work_ind[3] = 1;

    GecodeResult r = solve_gecode_ssp(
        inst.E, inst.T, m, Q, inst.N, inst.epsilon,
        work_ind, inst.salary, delta, inst.initial,
        inst.accepting, inst.demand, time_lim, /*break_symmetry=*/true);

    std::cout << dir << ","
              << inst.E << "," << inst.T << "," << inst.N << "," << inst.epsilon << ","
              << Q << "," << inst.transitions.size() << ","
              << r.status << "," << r.nodes << "," << r.fails << ","
              << r.solve_time_s << "," << r.objective << "\n";
    return 0;
}
