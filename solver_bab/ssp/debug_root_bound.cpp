/*
 * debug_root_bound.cpp — Diagnostic temporaire : la borne v1CLD à la racine
 * (mask entièrement libre) peut-elle dépasser un coût réellement atteignable ?
 *
 * Usage : debug_root_bound <répertoire instance> <cout_atteignable> [T_root]
 *
 * Si le v1CLD_star convergé donne lb_salary > cout_atteignable, la borne est
 * invalide (viole la dualité faible) même à la racine — sans besoin de
 * connaître la solution optimale x*, puisque toute solution atteignable est
 * incluse dans le domaine faisable de la racine.
 */
#include "gecode_bab_ssp.hpp"
#include "ssp_loader.hpp"
#include "../../solver_generic/ssp/ssp_cld_problem_bnb.hpp"
#include "../../solver_generic/v1cld_solver.hpp"

#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: debug_root_bound <dir> <cout_atteignable> [T_root]\n";
        return 1;
    }
    std::string dir = argv[1];
    double achievable = std::atof(argv[2]);
    int T_root = (argc > 3) ? std::atoi(argv[3]) : 200;
    bool use_polyak = (argc > 4) && std::atoi(argv[4]) != 0;

    SSPInstance inst = load_ssp_instance(dir);
    std::cout << "Instance : E=" << inst.E << " T=" << inst.T << " N=" << inst.N
              << " eps=" << inst.epsilon << "\n";

    Vec2i mask(inst.E, Vec1i(inst.T, -1));  // racine : tout libre
    SSPCLDProblemBnB prob(inst.transitions, inst.accepting, inst.initial,
                          inst.salary, inst.demand, inst.p, inst.epsilon,
                          inst.E, mask);

    // rel_tol=-1 : pas d'arret anticipe, on veut voir TOUTE la trajectoire
    // (y compris apres divergence) pour chercher un L1 invalide (< -achievable
    // en convention max(-salaire), i.e. lb_salary > achievable a un instant t).
    double v_lower = use_polyak ? -achievable : -1e18;
    std::cout << "v_lower passe = " << v_lower << (use_polyak ? " (Polyak)" : " (fallback 1/sqrt(t))") << "\n";
    V1CLDResult res = solve_v1cld(prob, T_root, 1, /*verbose=*/false,
                                  /*time_limit_s=*/-1.0, /*rel_tol=*/-1.0,
                                  /*tol_window=*/30, v_lower, {});

    double lb_salary = -res.v1CLD_star;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "iterations executees                = " << res.iterations_run << "\n";

    double worst_lb = -1e300;   // pire (plus grand) lb_salary observe sur TOUTE la trajectoire
    int    worst_it = -1;
    for (size_t k = 0; k < res.convergence.size(); ++k) {
        double lb_k = -res.convergence[k];
        if (lb_k > worst_lb) { worst_lb = lb_k; worst_it = static_cast<int>(k); }
    }
    std::cout << "pire lb_salary sur la trajectoire   = " << worst_lb
              << "  (iteration " << worst_it << ")\n";
    std::cout << "lb_salary (best_upper convergee)    = " << lb_salary << "\n";
    std::cout << "cout reellement atteignable         = " << achievable << "\n";

    bool invalid_trajectory = worst_lb > achievable + 1e-6;
    bool invalid_reported   = lb_salary > achievable + 1e-6;
    if (invalid_reported) {
        std::cout << "*** BORNE INVALIDE (valeur retournee/best_upper) ***\n";
        return 1;
    }
    if (invalid_trajectory) {
        std::cout << "*** ATTENTION : la trajectoire a produit un lb_salary invalide a "
                     "l'iteration " << worst_it << " (mais pas retenu comme 'best') ***\n";
        return 2;
    }
    std::cout << "Borne valide sur toute la trajectoire (lb <= cout atteignable).\n";
    return 0;
}
