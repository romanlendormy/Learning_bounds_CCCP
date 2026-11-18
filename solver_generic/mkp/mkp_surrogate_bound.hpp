#pragma once
/*
 * mkp_surrogate_bound.hpp — Borne surrogate partagée pour ζ_i (MKP déterministe).
 *
 * La borne actuelle de MKPProblemBnB (mkp_problem_bnb.cpp) résout m relaxations
 * 1D SÉPARÉES (une par contrainte j, ignorant les autres) et prend le MINIMUM.
 * C'est une borne valide (chaque relaxation séparée majore la relaxation LP
 * jointe, donc leur min aussi) mais structurellement plus lâche que nécessaire.
 *
 * Ici : relaxation SURROGATE — agrège les m contraintes en UNE seule via des
 * poids u_j ≥ 0, ramenant le problème à un knapsack fractionnaire 1D (tri +
 * remplissage glouton, comme aujourd'hui, juste sur un poids agrégé au lieu
 * d'un poids par dimension). Résultat mathématique standard (Dyer 1980,
 * littérature MDKP) : pour u_j ≥ 0 quelconques, cette borne est valide
 * (Σ_j u_j·(A_j x) ≤ Σ_j u_j·b_j est impliquée par les m contraintes d'origine,
 * donc max sous cette contrainte agrégée ≥ max sous les m contraintes) — et
 * avec de bons poids (ex. u_j = 1/b_j, normalisant chaque dimension), souvent
 * STRICTEMENT plus serrée que min_j(borne_j).
 *
 * Poids u_j et ORDRE DE TRI calculés UNE FOIS par appel solve_zeta(i, cost),
 * à partir de la capacité ORIGINALE b_i[j] (pas recalculés à chaque nœud
 * résiduel — la validité ne dépend pas du choix de u_j, seulement la finesse ;
 * garder u_j fixe permet de trier une seule fois par appel).
 *
 * 'active' est physiquement RÉORDONNÉ par ratio cost/agg_w décroissant (pas
 * un index de permutation séparé) : le branchement ET le calcul de borne
 * utilisent alors la MÊME convention simple "position idx..fin = non décidé"
 * (cf. usage dans mkp_problem_bnb_surrogate.cpp et zeta_gecode_solver.cpp).
 *
 * Partagé entre :
 *   - MKPProblemBnBSurrogate (mkp_problem_bnb_surrogate.cpp) : même squelette
 *     de B&B récursif que MKPProblemBnB, borne différente.
 *   - Le brancher Gecode (solver_bab/mkp/zeta_gecode_solver.cpp) : même borne
 *     injectée à chaque nœud via rel(obj <= ub), pour une comparaison
 *     scientifiquement propre (seul le moteur de recherche diffère, pas la
 *     qualité de la borne ni l'ordre de branchement).
 */

#include "types.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

struct MkpSurrogatePrep {
    Vec1d            u;             // (m,) poids par dimension, 1/b_i[j] (garde epsilon)
    std::vector<int> active_sorted; // indices ORIGINAUX des items actifs, triés par ratio décroissant
    Vec1d            agg_w_sorted;  // poids agrégé, aligné sur active_sorted
};

// Prépare poids/ordre — appelé UNE FOIS par solve_zeta(i, cost), avant la
// récursion. 'active' : indices originaux des items non triviaux (cost>0,
// pas "libres" — cf. filtrage dans l'appelant, identique à MKPProblemBnB).
inline MkpSurrogatePrep prepare_mkp_surrogate(
    const std::vector<int>& active,
    const Vec1d&            cost,
    const Vec2d&             Ai,      // (m, n) — Ai[j][k], k = indice ORIGINAL
    const Vec1d&             bi,      // (m,) capacité ORIGINALE du scénario
    int                      m)
{
    constexpr double EPS = 1e-9;
    constexpr double LARGE_PENALTY = 1e12;

    MkpSurrogatePrep prep;
    prep.u.resize(m);
    for (int j = 0; j < m; ++j)
        prep.u[j] = (bi[j] > EPS) ? (1.0 / bi[j]) : LARGE_PENALTY;

    int na = static_cast<int>(active.size());
    Vec1d agg_w(na);
    for (int idx = 0; idx < na; ++idx) {
        int k = active[idx];
        double w = 0.0;
        for (int j = 0; j < m; ++j) w += prep.u[j] * Ai[j][k];
        agg_w[idx] = w;
    }

    std::vector<int> order(na);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        int ka = active[a], kb = active[b];
        double ra = (agg_w[a] > EPS) ? cost[ka] / agg_w[a] : 1e18;
        double rb = (agg_w[b] > EPS) ? cost[kb] / agg_w[b] : 1e18;
        return ra > rb;
    });

    prep.active_sorted.resize(na);
    prep.agg_w_sorted.resize(na);
    for (int idx = 0; idx < na; ++idx) {
        prep.active_sorted[idx] = active[order[idx]];
        prep.agg_w_sorted[idx]  = agg_w[order[idx]];
    }
    return prep;
}

// Borne sur le sous-problème résiduel : items de position >= idx dans
// prep.active_sorted (position < idx == déjà décidé, branchement séquentiel).
// Recalcule seulement la capacité agrégée résiduelle (O(m), pas de retri) et
// remplit glouton dans l'ordre pré-trié.
inline double mkp_surrogate_bound(
    const MkpSurrogatePrep& prep,
    const Vec1d&            cost,
    int                     idx,
    const Vec1d&            rem_b,   // (m,) capacité résiduelle courante
    double                  val,     // valeur déjà acquise (items fixés à 1)
    int                     m)
{
    constexpr double EPS = 1e-9;
    double cap = 0.0;
    for (int j = 0; j < m; ++j) cap += prep.u[j] * rem_b[j];

    double v = val;
    int na = static_cast<int>(prep.active_sorted.size());
    for (int pos = idx; pos < na; ++pos) {
        double a = prep.agg_w_sorted[pos];
        int    k = prep.active_sorted[pos];
        if (a < EPS)              { v += cost[k]; }
        else if (a <= cap + EPS)  { v += cost[k]; cap -= a; }
        else                      { v += cost[k] * cap / a; break; }
    }
    return v;
}
