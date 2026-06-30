#pragma once
/*
 * gecode_solver.hpp — Solveur CP exact basé sur Gecode 6.
 *
 * Résout en one-shot :
 *   - CCMKP  : solve_gecode_mkp(...)
 *   - CCSSP  : solve_gecode_ssp(...)
 *
 * Les deux fonctions retournent un GecodeResult.
 */

#include "../solver_generic/types.hpp"
#include <string>

struct GecodeResult {
    double      objective;           // valeur objective (non scalée)
    bool        optimal;             // true si preuve d'optimalité
    bool        feasible;            // true si au moins une solution trouvée
    double      solve_time_s;        // temps total (construction + résolution)
    double      cold_start_time_s;   // temps de propagation initiale à la racine (construction modèle + init BAB)
    std::string status;              // "OPTIMAL" | "FEASIBLE" | "INFEASIBLE" | "TIMEOUT"
    int64_t     nodes = 0;           // nœuds traités (engine.statistics().node)
    int64_t     fails = 0;           // nœuds échoués (engine.statistics().fail)
    int64_t     depth = 0;           // profondeur max atteinte (engine.statistics().depth)
    int64_t     incumbent_improvements = 0;  // nombre de solutions améliorantes trouvées
};

/*
 * solve_gecode_mkp
 *   n        : nombre de variables binaires x[0..n-1]
 *   m        : nombre de contraintes par scénario
 *   N        : nombre de scénarios
 *   epsilon  : niveau de risque
 *   c        : (n,)     coûts déterministes
 *   p        : (N,)     probabilités
 *   A        : (N,m,n)  matrices de contraintes
 *   b        : (N,m)    membres droits
 *   time_lim : limite de temps en secondes (-1 = sans limite)
 */
GecodeResult solve_gecode_mkp(
    int n, int m, int N, double epsilon,
    const Vec1d& c, const Vec1d& p,
    const Vec3d& A, const Vec2d& b,
    double time_lim = 300.0);

/*
 * solve_gecode_ssp — SSP couverture multi-employés (chance-constraint joint)
 *
 *   E              : nombre d'employés
 *   T              : nombre de périodes
 *   m              : taille de l'alphabet Σ
 *   Q              : nombre d'états du DFA (partagé par tous les employés)
 *   N              : nombre de scénarios
 *   epsilon        : niveau de risque
 *   work_indicator : (m,)    1 si le symbole est "travail", 0 sinon
 *   salary         : (T,)    coût salarial par période (double, scalé en interne)
 *   delta          : [Q][m]  table de transition (-1 = transition morte)
 *   q0             : état initial du DFA
 *   accepting      : liste des états acceptants (état final légal)
 *   demand         : [N][T]  demande stochastique — ordre : scénario i, période t
 *   time_lim       : limite de temps en secondes (-1 = sans limite)
 *   break_symmetry : brisure de symétrie lex sur les lignes (défaut : true)
 */
GecodeResult solve_gecode_ssp(
    int E, int T, int m, int Q, int N, double epsilon,
    const std::vector<int>& work_indicator,
    const Vec1d&            salary,
    const std::vector<std::vector<int>>& delta,    // [Q][m]
    int                     q0,
    const std::vector<int>& accepting,
    const std::vector<std::vector<int>>& demand,   // [N][T]
    double time_lim       = 300.0,
    bool   break_symmetry = true);

/*
 * run_ssp_tests — Vérifie les propriétés de solve_gecode_ssp sur une instance
 *                 jouet (E=3, T=8, m=4, Q=3, N=20, ε=0.10).
 * Retourne true si tous les tests passent.
 */
bool run_ssp_tests();
