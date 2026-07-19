/*
 * debug_warmstart_rounds.cpp — Simule le regime EXACT de more() : rafales
 * courtes (T_node iterations), lambda warm-starte d'un round au suivant,
 * v_lower=-1e18 (jamais de vrai pas de Polyak) — mais SANS descendre dans
 * l'arbre (masque racine fixe a tous les rounds), pour isoler l'effet du
 * warm-start cumulatif de l'effet du masquage.
 *
 * Compare V1LD (mode=ld, decomposition par scenario, prouvee valide sur
 * papier) et V1CLD (mode=cld, decomposition additionnelle par creneau) sous
 * EXACTEMENT le meme protocole, sur la meme instance.
 *
 * Usage : debug_warmstart_rounds <dir> <cout_atteignable> <ld|cld> <n_rounds> <T_node>
 */
#include "gecode_bab_ssp.hpp"
#include "ssp_loader.hpp"
#include "../../solver_generic/ssp/ssp_cld_problem_bnb.hpp"
#include "../../solver_generic/ssp/ssp_problem_bnb.hpp"
#include "../../solver_generic/v1cld_solver.hpp"
#include "../../solver_generic/v1ld_solver.hpp"

#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: debug_warmstart_rounds <dir> <cout_atteignable> <ld|cld> <n_rounds> <T_node>\n";
        return 1;
    }
    std::string dir        = argv[1];
    double      achievable = std::atof(argv[2]);
    std::string mode       = argv[3];
    int         n_rounds   = std::atoi(argv[4]);
    int         T_node     = std::atoi(argv[5]);

    SSPInstance inst = load_ssp_instance(dir);
    std::cout << "Instance : E=" << inst.E << " T=" << inst.T << " N=" << inst.N
              << " eps=" << inst.epsilon << "  mode=" << mode << "\n";

    Vec2i mask(inst.E, Vec1i(inst.T, -1));  // racine fixe a TOUS les rounds

    double worst_lb = -1e300;
    int    worst_round = -1;

    if (mode == "cld") {
        SSPCLDProblemBnB prob(inst.transitions, inst.accepting, inst.initial,
                              inst.salary, inst.demand, inst.p, inst.epsilon,
                              inst.E, mask);
        Vec3d lambda;  // vide au round 0
        for (int r = 0; r < n_rounds; ++r) {
            V1CLDResult res = solve_v1cld(prob, T_node, 0, false, -1.0,
                                          1e-4, 30, -1e18, lambda);
            lambda = res.lambda_star;
            double lb = -res.v1CLD_star;
            if (lb > worst_lb) { worst_lb = lb; worst_round = r; }
            if (lb > achievable + 1e-6)
                std::cout << "  round " << r << " : lb_salary=" << lb
                          << "  *** INVALIDE (> " << achievable << ") ***\n";
        }
    } else if (mode == "ld") {
        SSPProblemBnB prob(inst.transitions, inst.accepting, inst.initial,
                           inst.salary, inst.demand, inst.p, inst.epsilon,
                           inst.E, mask);
        Vec2d lambda;  // vide au round 0
        for (int r = 0; r < n_rounds; ++r) {
            V1LDResult res = solve_v1ld(prob, T_node, 0, false, -1.0,
                                       1e-4, 30, -1e18, lambda);
            lambda = res.lambda_star;
            double lb = -res.v1LD_star;
            if (lb > worst_lb) { worst_lb = lb; worst_round = r; }
            if (lb > achievable + 1e-6)
                std::cout << "  round " << r << " : lb_salary=" << lb
                          << "  *** INVALIDE (> " << achievable << ") ***\n";
        }
    } else {
        std::cerr << "mode inconnu : " << mode << " (attendu 'ld' ou 'cld')\n";
        return 1;
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "pire lb_salary sur " << n_rounds << " rounds  = " << worst_lb
              << "  (round " << worst_round << ")\n";
    std::cout << "cout reellement atteignable          = " << achievable << "\n";
    if (worst_lb > achievable + 1e-6) {
        std::cout << "*** BORNE INVALIDE detectee sous ce protocole ***\n";
        return 1;
    }
    std::cout << "Borne restee valide sous ce protocole.\n";
    return 0;
}
