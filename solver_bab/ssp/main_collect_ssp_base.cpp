/*
 * main_collect_ssp_base.cpp — CLI de collecte GNN (SSP) avec pruning CP pur.
 *
 * Usage :
 *   bab_ssp_gecode_base_collect <instance_dir> [time_limit_s=-1] [outdir=.] [max_depth=10]
 *
 * <instance_dir> doit contenir instance.txt, automaton.txt, cost.txt,
 * demand.txt (cf. ssp_loader.hpp / solver_generic/ssp/export_to_txt.py).
 *
 * Format de sortie :
 *   <outdir>_base/instances_E{E}_T{T}_N{N}.txt
 *   <outdir>_base/nodes_E{E}_T{T}_N{N}.txt      — Format C : inst_id, depth,
 *                                                  curr_cost, x_mask[E*T], z_mask[N]
 *
 * Seule l'instance résolue à optimalité est écrite. start_id déduit du
 * nombre de lignes déjà présentes dans instances_E{E}_T{T}_N{N}.txt (reprise).
 */

#include "collect_ssp_base.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage : bab_ssp_gecode_base_collect <instance_dir> "
                     "[time_limit_s=-1] [outdir=.] [max_depth=10]\n"
                     "\n"
                     "Sortie : <outdir>_base/instances_E{E}_T{T}_N{N}.txt et nodes_...\n"
                     "Format nœuds (C) : inst_id, depth, curr_cost, x_mask[E*T], z_mask[N]\n";
        return 1;
    }

    std::string instance_dir = argv[1];
    double      time_limit   = (argc > 2) ? std::stod(argv[2]) : -1.0;
    std::string outdir       = (argc > 3) ? argv[3] : ".";
    int         max_depth    = (argc > 4) ? std::stoi(argv[4]) : 10;

    SSPInstance inst = load_ssp_instance(instance_dir);

    std::string outdir_mode = outdir + "_base";
    std::string sep = (outdir_mode.back() == '/') ? "" : "/";
    std::string tag = "E" + std::to_string(inst.E) + "_T" + std::to_string(inst.T)
                     + "_N" + std::to_string(inst.N);
    std::string instances_fp = outdir_mode + sep + "instances_" + tag + ".txt";
    std::string nodes_fp     = outdir_mode + sep + "nodes_"     + tag + ".txt";

    int start_id = 0;
    {
        std::ifstream existing(instances_fp);
        std::string tmp;
        while (std::getline(existing, tmp))
            if (!tmp.empty()) ++start_id;
    }

    std::filesystem::create_directories(outdir_mode);

    std::ofstream inst_out(instances_fp, std::ios::app);
    std::ofstream node_out(nodes_fp,     std::ios::app);
    if (!inst_out || !node_out) {
        std::cerr << "Erreur : impossible d'ouvrir les fichiers de sortie\n";
        return 1;
    }

    std::cout << "Instance : " << instance_dir
              << "  (E=" << inst.E << " T=" << inst.T << " N=" << inst.N
              << " eps=" << inst.epsilon << ")\n";
    std::cout << "Sortie   : " << instances_fp << "  /  " << nodes_fp << "\n";
    std::cout << "start_id = " << start_id << " (lignes déjà présentes)\n";
    std::cout << std::string(60, '-') << "\n";

    SSPCollectResult res = solve_gecode_bab_ssp_collect_base(
        inst, inst_out, node_out, start_id, time_limit, max_depth);

    std::cout << std::fixed << std::setprecision(2);
    if (res.optimal) {
        std::cout << "OPTIMAL  obj=" << res.objective
                  << "  nodes=" << res.nodes
                  << "  t=" << res.solve_time_s << "s\n";
    } else {
        std::cout << res.status << "  (ignoré)\n";
    }

    return 0;
}
