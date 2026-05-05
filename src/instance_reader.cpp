#include "instance_reader.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <map>

// ============================================================
// Parse une ligne CSV → vecteur de doubles
// ============================================================
std::vector<double> InstanceReader::parse_line(const std::string& line) {
    std::vector<double> values;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty())
            values.push_back(std::stod(token));
    }
    return values;
}

// ============================================================
// Chargement principal
// ============================================================
std::vector<CCSPInstance> InstanceReader::load(
    const std::string& filepath,
    double epsilon,
    int max_episodes)
{
    std::ifstream file(filepath);
    if (!file.is_open())
        throw std::runtime_error("Impossible d'ouvrir le fichier : " + filepath);

    // On accumule les lignes par épisode dans un hashmap
    // clé = episode_id, valeur = liste de lignes (chacune = un scénario)
    std::map<int, std::vector<std::vector<double>>> raw;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        auto values = parse_line(line);
        if (values.size() < 6) continue; // ligne malformée

        int episode_id = static_cast<int>(values[0]);

        // Limite sur le nombre d'épisodes
        if (max_episodes > 0 &&
            static_cast<int>(raw.size()) >= max_episodes &&
            raw.find(episode_id) == raw.end())
            continue;

        raw[episode_id].push_back(values);
    }

    // Construction des CCSPInstance
    std::vector<CCSPInstance> instances;
    instances.reserve(raw.size()); // optimisation pour éviter les reallocations

    for (auto& [ep_id, lines] : raw) {

        // Toutes les lignes d'un épisode ont le même m, n, pA, pB
        // Format : [episode, scenario, m, n, pA, pB, A_flat(m*n), b_flat(m)]
        const auto& first = lines[0];

        int m   = static_cast<int>(first[2]);
        int n   = static_cast<int>(first[3]);
        double pA = first[4];
        double pB = first[5];

        int expected_len = 6 + m * n + m;
        if (static_cast<int>(first.size()) != expected_len) {
            throw std::runtime_error(
                "Episode " + std::to_string(ep_id) +
                " : longueur de ligne incorrecte. Attendu=" +
                std::to_string(expected_len) +
                " Reçu=" + std::to_string(first.size())
            );
        }

        int N = static_cast<int>(lines.size()); // nombre de scénarios

        CCSPInstance inst;
        inst.episode = ep_id;
        inst.n       = n;
        inst.m       = m;
        inst.pA      = pA;
        inst.pB      = pB;
        inst.epsilon = epsilon;

        // Probabilités uniformes p_i = 1/N
        inst.p.assign(N, 1.0 / N);

        // Construction des scénarios
        inst.scenarios.reserve(N);
        for (const auto& row : lines) {
            CCSPScenario scen;
            scen.index = static_cast<int>(row[1]);

            // Reconstruction de A^i (m lignes de n valeurs)
            scen.A.resize(m, std::vector<double>(n));
            int offset = 6;
            for (int k = 0; k < m; ++k)
                for (int j = 0; j < n; ++j)
                    scen.A[k][j] = row[offset + k * n + j];

            // Reconstruction de b^i (m valeurs)
            scen.b.resize(m);
            int b_offset = 6 + m * n;
            for (int k = 0; k < m; ++k)
                scen.b[k] = row[b_offset + k];

            inst.scenarios.push_back(std::move(scen));
        }

        instances.push_back(std::move(inst));
    }

    return instances;
}
