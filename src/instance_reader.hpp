#pragma once

#include <vector>
#include <string>
#include <iostream>

// ============================================================
// Structures de données pour le CCSP (Set Covering)
// ============================================================

// Un scénario i : matrice A^i (m x n) et vecteur b^i (m)
struct CCSPScenario {
    int index;                         // indice du scénario dans l'épisode
    std::vector<std::vector<double>> A; // shape (m, n)
    std::vector<double> b;              // shape (m,)
};

// Une instance complète = un épisode du fichier
struct CCSPInstance {
    int episode;          // identifiant de l'épisode
    int n;                // nombre de variables (items / capteurs)
    int m;                // nombre de contraintes (cibles)
    double pA;            // paramètre de densité (info uniquement)
    double pB;            // paramètre d'activation (info uniquement)
    double epsilon;       // tolérance de violation (à fixer globalement)
    std::vector<double> p;             // probabilités p_i (uniformes = 1/N ici)
    std::vector<CCSPScenario> scenarios; // liste des N scénarios

    // Affichage de résumé pour debug
    void print_summary() const {
        std::cout << "Episode " << episode
                  << " | n=" << n << " m=" << m
                  << " | N=" << scenarios.size()
                  << " scenarios | pA=" << pA << " pB=" << pB
                  << std::endl;
    }

    // Affichage détaillé du premier scénario
    void print_first_scenario() const {
        if (scenarios.empty()) return;
        const auto& s = scenarios[0];
        std::cout << "  Scenario 0 :" << std::endl;
        std::cout << "  A[0] = [";
        for (int j = 0; j < n; ++j)
            std::cout << s.A[0][j] << (j < n-1 ? ", " : "");
        std::cout << "]" << std::endl;
        std::cout << "  b    = [";
        for (int k = 0; k < m; ++k)
            std::cout << s.b[k] << (k < m-1 ? ", " : "");
        std::cout << "]" << std::endl;
    }
};

// ============================================================
// InstanceReader : lit un fichier .txt et retourne une liste
// d'instances (une par épisode)
// ============================================================
class InstanceReader {
public:
    // Charge tous les épisodes d'un fichier
    // epsilon : tolérance de violation de la contrainte de chance
    // max_episodes : limite le nombre d'épisodes lus (-1 = tous)
    static std::vector<CCSPInstance> load(
        const std::string& filepath,
        double epsilon = 0.1,
        int max_episodes = -1
    );

private:
    // Parse une ligne CSV en vecteur de doubles
    static std::vector<double> parse_line(const std::string& line);
};
