/*
 * main_collect_gecode_ccpo.cpp — CLI de collecte de traces B&B (REF, CP pur)
 * pour l'entraînement du GNN CCPO.
 *
 * Charge un épisode CCPO (generators/ccpo) (mêmes fichiers/format que
 * main_gecode_ccpo.cpp -- chargeur dupliqué ici à l'identique, pour rester
 * autonome comme solver_bab/mkp/main_collect_gecode_base.cpp côté MKP).
 *
 * Usage :
 *   bab_ccpo_collect <répertoire épisode> <instance_id> <time_lim_s> \
 *                      <max_depth> <instances_out.txt> <nodes_out.txt>
 *
 * Les deux fichiers de sortie sont ouverts en APPEND -- appelé en boucle
 * (une fois par épisode) depuis un script Python/SLURM, cf.
 * collect/ccpo/collect_ccpo_samples.py.
 */

#include "collect_gecode_ccpo.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

struct CCPOInstanceFile {
    int n, N, k, Q, L, U;
    long long B;
    int eps_num, eps_den;
    std::vector<int> c_int;
    std::vector<std::vector<SigmaTriplet>> sigma;
};

static CCPOInstanceFile load_ccpo_episode(const std::string& dir)
{
    CCPOInstanceFile inst;
    {
        std::ifstream f(dir + "/instance.txt");
        if (!f) throw std::runtime_error("impossible d'ouvrir " + dir + "/instance.txt");
        f >> inst.n >> inst.N >> inst.k >> inst.Q >> inst.L >> inst.U
          >> inst.eps_num >> inst.eps_den;
        f >> inst.B;
        inst.c_int.resize(inst.n);
        for (int i = 0; i < inst.n; ++i) f >> inst.c_int[i];
        if (!f) throw std::runtime_error(dir + "/instance.txt : format invalide ou tronqué");
    }
    {
        std::ifstream f(dir + "/scenarios.txt");
        if (!f) throw std::runtime_error("impossible d'ouvrir " + dir + "/scenarios.txt");
        const int n_triplets = inst.n * (inst.n + 1) / 2;
        inst.sigma.assign(inst.N, {});
        for (int j = 0; j < inst.N; ++j) {
            int scen_id;
            f >> scen_id;
            if (!f || scen_id != j + 1)
                throw std::runtime_error(dir + "/scenarios.txt : bloc corrompu");
            long long psi_dummy;
            for (int i = 0; i < inst.n; ++i) f >> psi_dummy;
            inst.sigma[j].reserve(n_triplets);
            for (int t = 0; t < n_triplets; ++t) {
                int r, cc, v;
                f >> r >> cc >> v;
                inst.sigma[j].push_back({r - 1, cc - 1, v});
            }
        }
        if (!f) throw std::runtime_error(dir + "/scenarios.txt : format invalide ou tronqué");
    }
    return inst;
}

static int saa_threshold(int eps_num, int eps_den, int N)
{
    const long long num = static_cast<long long>(eps_den - eps_num) * N;
    return static_cast<int>((num + eps_den - 1) / eps_den);
}

int main(int argc, char** argv)
{
    if (argc < 7) {
        std::cerr << "Usage: bab_ccpo_collect <épisode> <instance_id> <time_lim_s> "
                     "<max_depth> <instances_out.txt> <nodes_out.txt>\n";
        return 1;
    }
    const std::string dir         = argv[1];
    const int instance_id         = std::atoi(argv[2]);
    const double time_lim         = std::atof(argv[3]);
    const int max_depth           = std::atoi(argv[4]);
    const std::string instances_p = argv[5];
    const std::string nodes_p     = argv[6];

    CCPOInstanceFile inst;
    try {
        inst = load_ccpo_episode(dir);
    } catch (const std::exception& e) {
        std::cerr << "Erreur de chargement (" << dir << "): " << e.what() << "\n";
        return 1;
    }
    const int min_feas = saa_threshold(inst.eps_num, inst.eps_den, inst.N);

    std::ofstream instances_out(instances_p, std::ios::app);
    std::ofstream nodes_out(nodes_p, std::ios::app);
    if (!instances_out || !nodes_out) {
        std::cerr << "Impossible d'ouvrir les fichiers de sortie\n";
        return 1;
    }

    CCPOResult r = solve_gecode_bab_ccpo_collect(
        inst.n, inst.N, inst.k, inst.Q, inst.L, inst.U, inst.B, min_feas,
        inst.c_int, inst.sigma, instances_out, nodes_out, instance_id,
        time_lim, max_depth);

    std::cout << dir << "," << instance_id << "," << r.status << "," << r.nodes << ","
              << r.depth << "," << r.solve_time_s << "," << r.objective << "\n";
    return 0;
}
