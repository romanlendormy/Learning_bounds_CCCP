/*
 * debug_capture_v1ld.cpp — Lance UNIQUEMENT le B&B v1LD (pas v1CLD) sur une
 * instance chargee depuis fichiers texte, pour capturer un dump
 * SSP_DEBUG_BOUND=1/SSP_DEBUG_DUMP=<path> specifique au mode V1LD (le dump
 * global est premier-arrive-premier-servi ; bench_ssp_bab_file lance v1CLD
 * AVANT v1LD et capturerait donc toujours un dump CLD en premier).
 *
 * Usage : debug_capture_v1ld <dir_instance> [T_node time_lim]
 */
#include "gecode_bab_ssp.hpp"
#include "ssp_loader.hpp"

#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: debug_capture_v1ld <dir_instance> [T_node time_lim]\n";
        return 1;
    }
    std::string dir = argv[1];
    int    T_node   = (argc > 2) ? std::atoi(argv[2]) : 3;
    double time_lim = (argc > 3) ? std::atof(argv[3]) : 90.0;

    SSPInstance inst = load_ssp_instance(dir);

    std::cout << "=== debug_capture_v1ld : " << dir << " ===\n";
    std::cout << "E=" << inst.E << " T=" << inst.T << " N=" << inst.N
               << " T_node=" << T_node << " time_lim=" << time_lim << "s\n";

    GecodeSSPBabResult r = solve_gecode_bab_ssp(
        inst.E, inst.T, inst.N, inst.epsilon,
        inst.transitions, inst.accepting, inst.initial,
        inst.salary, inst.demand, inst.p,
        time_lim, /*T_root=*/10, T_node, SSPBoundMode::V1LD);

    std::cout << "objectif=" << r.objective << " nodes=" << r.nodes
               << " statut=" << r.status
               << " tightenings=" << r.bound_tightenings
               << " fails_from_bound=" << r.fails_from_bound << "\n";
    return 0;
}
