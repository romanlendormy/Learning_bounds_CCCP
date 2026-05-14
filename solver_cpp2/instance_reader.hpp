#pragma once
/*
 * instance_reader.hpp — Lecture des instances CCSP depuis le format CSV.
 *
 * Format attendu (identique à solver/instance_reader.py) :
 *   [episode, scenario, m, n, pA, pB, A_flat(m*n), b_flat(m)]
 *   Une ligne par scénario ; les lignes d'un même épisode forment une instance.
 *
 * c (objectif) n'est pas dans le fichier → ones(n) par défaut.
 */

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

struct CCSPInstance {
    int    episode;
    int    n;
    int    m;
    int    N;
    double epsilon;
    double pA;
    double pB;

    std::vector<double>                              c;  // (n,)
    std::vector<double>                              p;  // (N,)
    std::vector<std::vector<std::vector<double>>>    A;  // (N, m, n)
    std::vector<std::vector<double>>                 b;  // (N, m)

    std::string str() const {
        return "CCSPInstance(ep=" + std::to_string(episode)
             + ", n="  + std::to_string(n)
             + ", m="  + std::to_string(m)
             + ", N="  + std::to_string(N)
             + ", eps=" + std::to_string(epsilon) + ")";
    }
};

static std::vector<double> parse_csv_line(const std::string& line) {
    std::vector<double> vals;
    std::istringstream ss(line);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty())
            vals.push_back(std::stod(token));
    }
    return vals;
}

inline std::vector<CCSPInstance> load_instances(
    const std::string&          filepath,
    double                      epsilon      = 0.1,
    int                         max_episodes = -1,
    const std::vector<double>&  c_override   = {}
) {
    std::map<int, std::vector<std::vector<double>>> raw;

    std::ifstream f(filepath);
    if (!f.is_open())
        throw std::runtime_error("Impossible d'ouvrir : " + filepath);

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto vals = parse_csv_line(line);
        if (vals.empty()) continue;

        int ep_id = static_cast<int>(vals[0]);

        if (max_episodes > 0) {
            bool already_seen = (raw.count(ep_id) > 0);
            if (!already_seen && static_cast<int>(raw.size()) >= max_episodes)
                continue;
        }
        raw[ep_id].push_back(vals);
    }

    std::vector<CCSPInstance> instances;
    instances.reserve(raw.size());

    for (auto& [ep_id, rows] : raw) {
        const auto& first = rows[0];

        int m  = static_cast<int>(first[2]);
        int n  = static_cast<int>(first[3]);
        double pA = first[4];
        double pB = first[5];
        int N  = static_cast<int>(rows.size());

        int expected = 6 + m * n + m;
        if (static_cast<int>(first.size()) != expected)
            throw std::runtime_error(
                "Épisode " + std::to_string(ep_id)
                + " : longueur ligne incorrecte (attendu "
                + std::to_string(expected) + ", reçu "
                + std::to_string(first.size()) + ")"
            );

        std::vector<double> c_vec =
            c_override.empty() ? std::vector<double>(n, 1.0) : c_override;

        std::vector<double> p_vec(N, 1.0 / N);

        std::vector<std::vector<std::vector<double>>> A_arr(
            N, std::vector<std::vector<double>>(m, std::vector<double>(n)));
        std::vector<std::vector<double>> b_arr(N, std::vector<double>(m));

        for (int i = 0; i < N; ++i) {
            const auto& row = rows[i];
            int offset = 6;
            for (int j = 0; j < m; ++j)
                for (int k = 0; k < n; ++k)
                    A_arr[i][j][k] = row[offset + j * n + k];
            for (int j = 0; j < m; ++j)
                b_arr[i][j] = row[offset + m * n + j];
        }

        instances.push_back(CCSPInstance{
            ep_id, n, m, N, epsilon, pA, pB,
            c_vec, p_vec, A_arr, b_arr
        });
    }

    return instances;
}
