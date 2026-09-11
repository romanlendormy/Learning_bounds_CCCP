#pragma once
/*
 * ssp_loader.hpp — Chargement d'une instance SSP depuis des fichiers texte.
 *
 * Format texte (4 fichiers dans un même répertoire) :
 *
 *   instance.txt
 *     E <int>
 *     T <int>
 *     N <int>
 *     EPSILON <double>
 *     INTERCHANGEABLE <0|1>   (réservé — non utilisé par le solveur SLD)
 *
 *   automaton.txt
 *     NUM_STATES <int>
 *     INITIAL <int>
 *     N_ACCEPTING <int>
 *     <q0> <q1> ...           (états acceptants, un par ligne ou sur une ligne)
 *     N_TRANSITIONS <int>
 *     <src> <sym_int> <dst>   (une transition par ligne)
 *     ...
 *     sym_int : O=0, P=1, R=2, W=3
 *
 *   cost.txt
 *     T <int>
 *     SALARY <s0> <s1> ... <sT-1>   (coûts salariaux, flottants)
 *
 *   demand.txt
 *     T <int>
 *     N <int>
 *     <d[0,0]> <d[1,0]> ... <d[T-1,0]>   (scénario 0 : T entiers)
 *     <d[0,1]> <d[1,1]> ... <d[T-1,1]>   (scénario 1)
 *     ...                                  (N lignes au total)
 *
 * Utilisation :
 *   SSPInstance inst = load_ssp_instance("chemin/vers/répertoire");
 *   SSPShiftProblem prob = make_ssp_problem(inst);
 *
 * Script Python d'export depuis HDF5 → texte : export_to_txt.py (dans ssp/).
 */

#include "ssp_problem.hpp"
#include <string>

struct SSPInstance {
    // Paramètres scalaires
    int    E, T, N;
    double epsilon;
    bool   interchangeable;

    // DFA
    int                                    initial;
    std::vector<int>                       accepting;
    std::vector<std::tuple<int,int,int>>   transitions;  // (src, sym, dst)

    // Coûts et demande
    Vec1d salary;    // (T,)
    Vec2i demand;    // (N, T)  — demand[scénario][période]
    Vec1d p;         // (N,)   — probabilités uniformes (1/N) par défaut
};

// Charge les 4 fichiers texte depuis un répertoire (chemin sans slash final)
SSPInstance load_ssp_instance(const std::string& dir);

// Construit SSPShiftProblem depuis une SSPInstance déjà chargée
SSPShiftProblem make_ssp_problem(const SSPInstance& inst);
