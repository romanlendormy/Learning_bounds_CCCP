/*
 * main_gecode_collect.cpp - CLI de collecte de données d'entraînement GNN.
 *
 * Usage :
 *   bab_mkp_gecode_collect <dataset.txt> [OPTIONS]
 *
 * Options (positionnelles, dans l'ordre) :
 *   epsilon         (défaut 0.1)
 *   T_node          (défaut 30)
 *   time_limit_s    (défaut 120)
 *   T_root          (défaut 200)
 *   mode            "v1ld" ou "v1cld"  (défaut "v1ld")
 *   instances_out   chemin fichier instances  (défaut "instances.txt")
 *   nodes_out       chemin fichier nœuds      (défaut "nodes.txt")
 *   start_id        instance_id de départ, utile pour accumuler  (défaut 0)
 *
 * Le programme charge toutes les réplications du fichier dataset (colonne 4),
 * résout chacune en B&B avec collecte, et append les résultats dans les fichiers
 * de sortie.  Seules les instances résolues à optimalité sont écrites.
 *
 * Exemple :
 *   bab_mkp_gecode_collect data/ccmk-n10-m2-S50.txt 0.1 30 120 200 v1ld \
 *       instances.txt nodes.txt 0
 */

#include "gecode_bab_collect.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Chargement d'une instance (réplication rep) depuis un fichier Song/Pisinger
// ─────────────────────────────────────────────────────────────────────────────

struct MKPInst {
    Vec1d c, p;
    Vec3d A;
    Vec2d b;
    int n, m, N;
};

static MKPInst load_rep(const std::string& fp, int rep)
{
    std::ifstream f(fp);
    if (!f) { std::cerr << "Erreur : impossible d'ouvrir " << fp << "\n"; std::exit(1); }

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
    if (rows.empty()) {
        std::cerr << "Aucun scénario pour rep=" << rep << " dans " << fp << "\n";
        std::exit(1);
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
        for (int j = 0; j < m; ++j)
            b[i][j] = v[5 + m * n + j];
        if (i == 0)
            for (int k = 0; k < n; ++k)
                c[k] = v[5 + m * n + m + k];
    }
    Vec1d p(N, 1.0 / N);
    return {c, p, A, b, n, m, N};
}

// ─────────────────────────────────────────────────────────────────────────────
// Détection des réplications disponibles dans un fichier
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<int> get_reps(const std::string& fp)
{
    std::ifstream f(fp);
    if (!f) { std::cerr << "Erreur : impossible d'ouvrir " << fp << "\n"; std::exit(1); }

    std::set<int> reps;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string tok;
        std::vector<std::string> tokens;
        while (std::getline(iss, tok, ',')) tokens.push_back(tok);
        if (tokens.size() > 4)
            reps.insert(std::stoi(tokens[4]));
    }
    return std::vector<int>(reps.begin(), reps.end());
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage : bab_mkp_gecode_collect <dataset.txt> "
                     "[epsilon=0.1] [T_node=30] [time_limit=-1.0] [T_root=200] "
                     "[mode=v1ld] [outdir=.] [max_depth_collect=-1]\n"
                     "\n"
                     "Les fichiers de sortie sont nommés automatiquement :\n"
                     "  <outdir>/instances_n{n}.txt  et  <outdir>/nodes_n{n}.txt\n"
                     "où n est le nombre de variables de la première instance.\n"
                     "Les fichiers sont ouverts en mode append ; start_id est\n"
                     "déduit du nombre de lignes déjà présentes.\n";
        return 1;
    }

    std::string dataset    = argv[1];
    double      epsilon    = (argc > 2) ? std::stod(argv[2]) : 0.10;
    int         T_node     = (argc > 3) ? std::stoi(argv[3]) : 60;
    double      time_limit = (argc > 4) ? std::stod(argv[4]) : -1.0;
    int         T_root     = (argc > 5) ? std::stoi(argv[5]) : 200;
    std::string mode_str        = (argc > 6) ? argv[6] : "v1ld";
    std::string outdir          = (argc > 7) ? argv[7] : ".";
    int         max_depth_collect = (argc > 8) ? std::stoi(argv[8]) :10;

    BoundMode bound_mode = (mode_str == "v1cld") ? BoundMode::V1CLD : BoundMode::V1LD;

    // Détecter les réplications disponibles et lire n depuis la première
    std::vector<int> reps = get_reps(dataset);
    if (reps.empty()) {
        std::cerr << "Aucune réplication trouvée dans " << dataset << "\n";
        return 1;
    }
    MKPInst first = load_rep(dataset, reps[0]);
    int n_val = first.n;

    // Chemins de sortie dérivés de n
    std::string sep = (outdir.back() == '/') ? "" : "/";
    std::string instances_fp = outdir + sep + "instances_n" + std::to_string(n_val) + ".txt";
    std::string nodes_fp     = outdir + sep + "nodes_n"     + std::to_string(n_val) + ".txt";

    // Déduire start_id en comptant les lignes non vides déjà présentes
    int start_id = 0;
    {
        std::ifstream existing(instances_fp);
        std::string tmp;
        while (std::getline(existing, tmp))
            if (!tmp.empty()) ++start_id;
    }

    // Créer le répertoire de sortie si nécessaire
    std::filesystem::create_directories(outdir);

    // Ouvrir en mode append (crée le fichier s'il n'existe pas)
    std::ofstream inst_out(instances_fp, std::ios::app);
    std::ofstream node_out(nodes_fp,     std::ios::app);
    if (!inst_out || !node_out) {
        std::cerr << "Erreur : impossible d'ouvrir les fichiers de sortie\n";
        return 1;
    }

    std::cout << "Fichier : " << dataset << "  -  "
              << reps.size() << " réplication(s) détectée(s)\n";
    std::cout << "Sortie  : " << instances_fp << "  /  " << nodes_fp << "\n";
    std::cout << "start_id = " << start_id << " (lignes déjà présentes)\n";
    std::cout << std::string(60, '-') << "\n";

    int instance_id   = start_id;
    int solved        = 0;
    int skipped       = 0;

    // La première réplication est déjà chargée dans `first` — on la réutilise
    // pour éviter un double chargement.
    bool first_rep_done = false;
    for (int rep : reps) {
        MKPInst inst = (!first_rep_done) ? std::move(first) : load_rep(dataset, rep);
        first_rep_done = true;

        std::cout << "rep=" << rep
                  << "  n=" << inst.n << "  m=" << inst.m << "  N=" << inst.N
                  << "  id=" << instance_id << " ... " << std::flush;

        GecodeV1LDResult res = solve_gecode_bab_mkp_collect(
            inst.n, inst.m, inst.N, epsilon,
            inst.c, inst.p, inst.A, inst.b,
            inst_out, node_out, instance_id,
            time_limit, T_root, T_node, bound_mode,
            false, 0.1, max_depth_collect);

        if (res.optimal) {
            std::cout << std::fixed << std::setprecision(2)
                      << "OPTIMAL  obj=" << res.objective
                      << "  nodes=" << res.nodes
                      << "  t=" << res.solve_time_s << "s\n";
            ++solved;
            ++instance_id;
        } else {
            std::cout << res.status << "  (ignoré)\n";
            ++skipped;
        }
    }

    std::cout << std::string(60, '-') << "\n";
    std::cout << "Résumé : " << solved << " instance(s) écrites"
              << ", " << skipped << " ignorée(s)"
              << "  (id=" << start_id << ".." << instance_id - 1 << ")\n";

    return 0;
}
