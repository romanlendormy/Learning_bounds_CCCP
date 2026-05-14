/*
 * main.cpp — Binaire de test standalone pour v1ld_cpp2.
 * Usage : ./v1ld_test2 [data_file] [T] [decomp=true|false] [max_ep]
 */

#include "instance_reader.hpp"
#include "v1ld_core.hpp"

#include <iostream>
#include <iomanip>
#include <string>

int main(int argc, char* argv[]) {
    std::string path    = (argc > 1) ? argv[1]
                        : "../data/instances_n10_m5_N50.txt";
    int         T       = (argc > 2) ? std::stoi(argv[2]) : 500;
    bool        decomp  = (argc > 3) ? (std::string(argv[3]) == "true") : true;
    int         max_ep  = (argc > 4) ? std::stoi(argv[4]) : 5;

    std::vector<CCSPInstance> insts;
    try {
        insts = load_instances(path, 0.1, max_ep);
    } catch (const std::exception& e) {
        std::cerr << "Erreur : " << e.what() << "\n";
        return 1;
    }

    std::cout << std::fixed << std::setprecision(4);
    std::cout << std::setw(6)  << "ep"
              << std::setw(6)  << "n"
              << std::setw(6)  << "m"
              << std::setw(6)  << "N"
              << std::setw(12) << "v1LD"
              << std::setw(10) << "time(s)"
              << "\n"
              << std::string(46, '-') << "\n";

    for (const auto& inst : insts) {
        SolverResult res = solve_impl(
            inst.c, inst.p, inst.A, inst.b,
            inst.epsilon, T, decomp, /*verbose=*/false
        );
        std::cout << std::setw(6)  << inst.episode
                  << std::setw(6)  << inst.n
                  << std::setw(6)  << inst.m
                  << std::setw(6)  << inst.N
                  << std::setw(12) << res.v1LD_star
                  << std::setw(10) << res.solve_time_s
                  << "\n";
    }
    return 0;
}
