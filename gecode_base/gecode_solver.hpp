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
    double      objective;      // valeur objective (non scalée)
    bool        optimal;        // true si preuve d'optimalité
    bool        feasible;       // true si au moins une solution trouvée
    double      solve_time_s;   // temps total (construction + résolution)
    std::string status;         // "OPTIMAL" | "FEASIBLE" | "INFEASIBLE" | "TIMEOUT"
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
 * solve_gecode_ssp
 *   n        : nombre de périodes
 *   m        : nombre d'activités par période
 *   k        : nombre d'automates par scénario
 *   Q        : nombre d'états par automate
 *   N        : nombre de scénarios
 *   epsilon  : niveau de risque
 *   profit   : (n*m,)        matrice de profits aplatie (profit[t*m+a])
 *   p        : (N,)          probabilités
 *   delta    : [N][k][Q][m]  tables de transition (-1 = transition morte)
 *   accepting: [N][k][Q]     états acceptants (1=oui, 0=non)
 *   time_lim : limite de temps en secondes (-1 = sans limite)
 */
GecodeResult solve_gecode_ssp(
    int n, int m, int k, int Q, int N, double epsilon,
    const Vec1d& profit, const Vec1d& p,
    const std::vector<std::vector<std::vector<std::vector<int>>>>& delta,
    const std::vector<std::vector<std::vector<int>>>& accepting,
    double time_lim = 300.0);
