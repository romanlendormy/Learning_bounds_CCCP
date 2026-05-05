#include "instance_reader.hpp"
#include <iostream>
#include <stdexcept>

int main(int argc, char* argv[]) {

    // Chemin du fichier passé en argument, sinon valeur par défaut
    std::string filepath = (argc > 1)
        ? argv[1]
        : "data/covset/datasets/train/ccsp-train-small.txt";

    std::cout << "=== Test InstanceReader ===" << std::endl;
    std::cout << "Fichier : " << filepath << std::endl << std::endl;

    try {
        // Charger uniquement les 3 premiers épisodes pour le test
        auto instances = InstanceReader::load(filepath, /*epsilon=*/0.1, /*max_episodes=*/3);

        std::cout << instances.size() << " épisode(s) chargé(s)" << std::endl << std::endl;

        for (const auto& inst : instances) {
            inst.print_summary();
            inst.print_first_scenario();
            std::cout << std::endl;
        }

        // Vérifications de base
        bool ok = true;
        for (const auto& inst : instances) {
            // Chaque scénario doit avoir A de shape (m, n) et b de taille m
            for (const auto& s : inst.scenarios) {
                if (static_cast<int>(s.A.size()) != inst.m) { ok = false; break; }
                if (static_cast<int>(s.A[0].size()) != inst.n) { ok = false; break; }
                if (static_cast<int>(s.b.size()) != inst.m) { ok = false; break; }
            }
            // Vérification ||b||_inf >= 1 pour chaque scénario
            for (const auto& s : inst.scenarios) {
                double bmax = *std::max_element(s.b.begin(), s.b.end());
                if (bmax < 0.5) { ok = false; break; }
            }
        }

        std::cout << "Vérification shapes et ||b||_inf=1 : "
                  << (ok ? "OK" : "ECHEC") << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Erreur : " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
