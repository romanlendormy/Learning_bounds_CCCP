/*
 * debug_v1cld_parallel_check.cpp — Test de non-régression : compare
 * solve_v1cld (séquentiel, solver_generic/v1cld_solver.cpp, INCHANGÉ) vs
 * solve_v1cld_parallel (nouvelle copie parallélisée OpenMP,
 * solver_generic/v1cld_solver_parallel.cpp), sur la MÊME instance et les
 * MÊMES paramètres, pour vérifier qu'elles produisent un résultat identique
 * (à la tolérance flottante près) avant d'utiliser la variante parallèle
 * dans campaign_v1ld_v1cld_vs_ref.
 *
 * Sert uniquement à ce test de non-régression — solve_v1cld_parallel est une
 * copie séparée (pas une modif en place) précisément pour ne pas risquer de
 * casser v1cld_solver.cpp, partagé avec MKP et d'autres scripts SSP établis.
 *
 * Usage : debug_v1cld_parallel_check <répertoire instance> [T=100]
 *
 * Relancer ce binaire avec OMP_NUM_THREADS=1 puis OMP_NUM_THREADS>1 : le
 * résultat "parallel" doit être identique dans les deux cas (et identique à
 * "sequential"), puisque la boucle (B) n'a aucune réduction inter-thread —
 * chaque itération i écrit dans sa propre case, indépendante de l'ordre
 * d'exécution des threads.
 */
#include "gecode_bab_ssp.hpp"
#include "ssp_loader.hpp"
#include "../../solver_generic/ssp/ssp_cld_problem_bnb.hpp"
#include "../../solver_generic/v1cld_solver.hpp"
#include "../../solver_generic/v1cld_solver_parallel.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: debug_v1cld_parallel_check <dir> [T=100]\n";
        return 1;
    }
    std::string dir = argv[1];
    int T = (argc > 2) ? std::atoi(argv[2]) : 100;

    SSPInstance inst = load_ssp_instance(dir);
    std::cout << "Instance : E=" << inst.E << " T=" << inst.T << " N=" << inst.N
              << " eps=" << inst.epsilon << "  T_iters=" << T << "\n";

    Vec2i mask(inst.E, Vec1i(inst.T, -1));  // racine : tout libre
    SSPCLDProblemBnB prob(inst.transitions, inst.accepting, inst.initial,
                          inst.salary, inst.demand, inst.p, inst.epsilon,
                          inst.E, mask);

    // rel_tol=-1, time_limit_s=-1 : déroule TOUJOURS les T itérations
    // complètes des deux côtés, pas d'arrêt anticipé qui pourrait masquer une
    // divergence tardive.
    V1CLDResult res_seq = solve_v1cld(prob, T, /*log_interval=*/1, /*verbose=*/false,
                                       /*time_limit_s=*/-1.0, /*rel_tol=*/-1.0,
                                       /*tol_window=*/30, /*v_lower=*/-1e18, {});
    V1CLDResult res_par = solve_v1cld_parallel(prob, T, 1, false,
                                                -1.0, -1.0, 30, -1e18, {});

    std::cout << std::fixed << std::setprecision(9);
    std::cout << "iterations_run       seq=" << res_seq.iterations_run
              << "  par=" << res_par.iterations_run << "\n";
    std::cout << "v1CLD_star           seq=" << res_seq.v1CLD_star
              << "  par=" << res_par.v1CLD_star << "\n";

    double diff_star = std::abs(res_seq.v1CLD_star - res_par.v1CLD_star);
    std::cout << "diff v1CLD_star      = " << diff_star << "\n";

    size_t n = std::min(res_seq.convergence.size(), res_par.convergence.size());
    double max_diff_traj = 0.0;
    size_t worst_t = 0;
    for (size_t t = 0; t < n; ++t) {
        double dt = std::abs(res_seq.convergence[t] - res_par.convergence[t]);
        if (dt > max_diff_traj) { max_diff_traj = dt; worst_t = t; }
    }
    std::cout << "diff max trajectoire = " << max_diff_traj
              << "  (iteration " << worst_t << ", n_comparees=" << n << ")\n";

    bool ok = (res_seq.iterations_run == res_par.iterations_run)
              && (diff_star < 1e-6)
              && (max_diff_traj < 1e-6);

    if (ok) {
        std::cout << "=== OK : solve_v1cld_parallel identique a solve_v1cld ===\n";
        return 0;
    }
    std::cout << "*** DIVERGENCE : solve_v1cld_parallel NE correspond PAS a solve_v1cld ***\n";
    return 1;
}
