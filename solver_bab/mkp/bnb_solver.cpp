/*
 * bnb_solver.cpp — Solveur B&B standalone pour le sous-problème ζ du MKP.
 *
 * Algorithme identique à mkp_problem_bnb.cpp (solver_generic/mkp/).
 * Aucune dépendance externe (pas d'OR-Tools, pas de pybind11).
 * Interface C exportée pour ctypes Python.
 *
 * Compilation :
 *   g++ -O3 -std=c++17 -shared -fPIC -o bnb_solver.so bnb_solver.cpp
 */

#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

// ── Algorithme B&B ────────────────────────────────────────────────────────────
//
// Pour un scénario i (n_g variables libres, m contraintes, capacités résiduelles b_i) :
//   max  cost^T x   s.t.  A_i x <= b_i,  x in {0,1}^n_g
// Borne supérieure LP par colonne (Dantzig) sur chaque dimension, min sur les m.

namespace {

struct MkpBnB {
    const int*    active;  // indices des variables actives dans [0, n_g)
    const double* cost;    // coûts lagrangiens (n_g,)
    const double* A_i;     // matrice (m, n_g) row-major pour le scénario i
    int na, m, n_g;
    double best_val;
    std::vector<int8_t> x_bits, best_x_bits;
    std::vector<std::vector<int>> sort_j;  // ordre des actifs par ratio c/A pour chaque dim j

    double lp_j(int j, int idx, double cap_j, double val) const {
        double v = val;
        for (int ki : sort_j[j]) {
            if (ki < idx) continue;
            int    k = active[ki];
            double a = A_i[j * n_g + k];
            if      (a < 1e-12)         { v += cost[k]; }
            else if (a <= cap_j + 1e-12){ v += cost[k]; cap_j -= a; }
            else                        { v += cost[k] * cap_j / a; break; }
        }
        return v;
    }

    double upper_bound(int idx, const double* rem_b, double val) const {
        double ub = 1e30;
        for (int j = 0; j < m; ++j) {
            double lj = lp_j(j, idx, rem_b[j], val);
            if (lj < ub) ub = lj;
        }
        return ub;
    }

    void run(int idx, double* rem_b, double val) {
        if (idx == na) {
            if (val > best_val + 1e-9) { best_val = val; best_x_bits = x_bits; }
            return;
        }
        if (upper_bound(idx, rem_b, val) <= best_val + 1e-9) return;

        int k = active[idx];
        bool feasible = true;
        for (int j = 0; j < m; ++j)
            if (A_i[j * n_g + k] > rem_b[j] + 1e-12) { feasible = false; break; }

        if (feasible) {
            for (int j = 0; j < m; ++j) rem_b[j] -= A_i[j * n_g + k];
            x_bits[idx] = 1;
            run(idx + 1, rem_b, val + cost[k]);
            x_bits[idx] = 0;
            for (int j = 0; j < m; ++j) rem_b[j] += A_i[j * n_g + k];
        }
        run(idx + 1, rem_b, val);
    }
};

// Résout le B&B pour un scénario.
// cost_i  : (n_g,) — coûts lagrangiens (déjà detach du graphe de calcul)
// A_i     : (m, n_g) row-major — contraintes du scénario
// b_i     : (m,) — capacités résiduelles
// x_out_i : (n_g,) — résultat binaire, pré-initialisé à 0
static void solve_one(
    int n_g, int m,
    const double* cost_i,
    const double* A_i,
    const double* b_i,
    double*       x_out_i
) {
    std::vector<int> active;
    active.reserve(n_g);
    double val_free = 0.0;

    for (int k = 0; k < n_g; ++k) {
        if (cost_i[k] <= 0.0) continue;
        bool has_constraint = false;
        for (int j = 0; j < m; ++j)
            if (A_i[j * n_g + k] >= 1e-9) { has_constraint = true; break; }
        if (!has_constraint) { x_out_i[k] = 1.0; val_free += cost_i[k]; }
        else                 { active.push_back(k); }
    }

    if (active.empty()) return;

    // Tri surrogate décroissant (ratio coût / utilisation normalisée)
    std::sort(active.begin(), active.end(), [&](int a, int b_idx) {
        double wa = 0.0, wb = 0.0;
        for (int j = 0; j < m; ++j) {
            if (b_i[j] > 1e-12) {
                wa += A_i[j * n_g + a]     / b_i[j];
                wb += A_i[j * n_g + b_idx] / b_i[j];
            }
        }
        double ra = (wa > 1e-12) ? cost_i[a]     / wa : 1e18;
        double rb = (wb > 1e-12) ? cost_i[b_idx] / wb : 1e18;
        return ra > rb;
    });

    int na = static_cast<int>(active.size());

    // Ordre des actifs par ratio c[k]/A[j,k] pour chaque dimension j (LP bound)
    std::vector<std::vector<int>> sort_j(m, std::vector<int>(na));
    for (int j = 0; j < m; ++j) {
        std::iota(sort_j[j].begin(), sort_j[j].end(), 0);
        std::sort(sort_j[j].begin(), sort_j[j].end(), [&](int a, int b_idx) {
            int ka = active[a], kb = active[b_idx];
            double ra = (A_i[j*n_g+ka] > 1e-12) ? cost_i[ka] / A_i[j*n_g+ka] : 1e18;
            double rb = (A_i[j*n_g+kb] > 1e-12) ? cost_i[kb] / A_i[j*n_g+kb] : 1e18;
            return ra > rb;
        });
    }

    MkpBnB bb;
    bb.active   = active.data();
    bb.cost     = cost_i;
    bb.A_i      = A_i;
    bb.na = na; bb.m = m; bb.n_g = n_g;
    bb.best_val = val_free;
    bb.x_bits.assign(na, 0);
    bb.best_x_bits.assign(na, 0);
    bb.sort_j = std::move(sort_j);

    std::vector<double> rem_b(b_i, b_i + m);
    bb.run(0, rem_b.data(), val_free);

    for (int ki = 0; ki < na; ++ki)
        if (bb.best_x_bits[ki]) x_out_i[active[ki]] = 1.0;
}

// ── Variante surrogate (borne agrégée Σ_j u_j·A_i[j][k], u_j = 1/b_i[j], au
// lieu du minimum de m relaxations séparées) — port fidèle de
// solver_generic/mkp/mkp_surrogate_bound.hpp dans le style tableaux plats de
// ce fichier. ζ_i retourne EXACTEMENT la même valeur que solve_one (seule la
// preuve d'optimalité interne diffère) — validé dans le vrai B&B SLD
// séquentiel (gecode_bab.hpp, MKPProblemBnBSurrogate) : mêmes nœuds/objectif,
// ×2.25 plus rapide (n=20/N=100).
//
// Simplification par rapport à solve_one : PAS de sort_j par dimension — un
// seul tri agrégé sert à la fois de borne ET d'ordre de branchement (comme
// mkp_problem_bnb_surrogate.cpp), donc pas de "if (ki < idx) continue" : les
// positions idx..na-1 dans active/agg_w (déjà triés) sont exactement les
// items non décidés.

struct MkpBnBSurr {
    const int*    active;  // indices des variables actives dans [0, n_g), triés par ratio agrégé décroissant
    const double* cost;    // coûts lagrangiens (n_g,), indexé par indice ORIGINAL
    const double* agg_w;   // poids agrégé, ALIGNÉ sur active (agg_w[pos] = poids de active[pos])
    const double* u;       // poids par dimension (m,), u[j] = 1/b_i[j]
    const double* A_i;     // matrice (m, n_g) row-major, comme MkpBnB
    int na, m, n_g;
    double best_val;
    std::vector<int8_t> x_bits, best_x_bits;

    double bound(int idx, const double* rem_b, double val) const {
        double cap = 0.0;
        for (int j = 0; j < m; ++j) cap += u[j] * rem_b[j];
        double v = val;
        for (int pos = idx; pos < na; ++pos) {
            double a = agg_w[pos];
            int    k = active[pos];
            if      (a < 1e-12)          { v += cost[k]; }
            else if (a <= cap + 1e-12)   { v += cost[k]; cap -= a; }
            else                          { v += cost[k] * cap / a; break; }
        }
        return v;
    }

    void run(int idx, double* rem_b, double val) {
        if (idx == na) {
            if (val > best_val + 1e-9) { best_val = val; best_x_bits = x_bits; }
            return;
        }
        if (bound(idx, rem_b, val) <= best_val + 1e-9) return;

        int k = active[idx];
        bool feasible = true;
        for (int j = 0; j < m; ++j)
            if (A_i[j * n_g + k] > rem_b[j] + 1e-12) { feasible = false; break; }

        if (feasible) {
            for (int j = 0; j < m; ++j) rem_b[j] -= A_i[j * n_g + k];
            x_bits[idx] = 1;
            run(idx + 1, rem_b, val + cost[k]);
            x_bits[idx] = 0;
            for (int j = 0; j < m; ++j) rem_b[j] += A_i[j * n_g + k];
        }
        run(idx + 1, rem_b, val);
    }
};

static void solve_one_surrogate(
    int n_g, int m,
    const double* cost_i,
    const double* A_i,
    const double* b_i,
    double*       x_out_i
) {
    std::vector<int> active;
    active.reserve(n_g);
    double val_free = 0.0;

    for (int k = 0; k < n_g; ++k) {
        if (cost_i[k] <= 0.0) continue;
        bool has_constraint = false;
        for (int j = 0; j < m; ++j)
            if (A_i[j * n_g + k] >= 1e-9) { has_constraint = true; break; }
        if (!has_constraint) { x_out_i[k] = 1.0; val_free += cost_i[k]; }
        else                 { active.push_back(k); }
    }

    if (active.empty()) return;

    constexpr double EPS = 1e-9;
    constexpr double LARGE_PENALTY = 1e12;
    std::vector<double> u(m);
    for (int j = 0; j < m; ++j) u[j] = (b_i[j] > EPS) ? (1.0 / b_i[j]) : LARGE_PENALTY;

    int na = static_cast<int>(active.size());
    std::vector<double> agg_w(na);
    for (int idx = 0; idx < na; ++idx) {
        int k = active[idx];
        double w = 0.0;
        for (int j = 0; j < m; ++j) w += u[j] * A_i[j * n_g + k];
        agg_w[idx] = w;
    }

    // Tri par ratio cost/agg_w décroissant — sert à la fois d'ordre de
    // branchement et d'ordre de remplissage glouton pour bound().
    std::vector<int> order(na);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b_idx) {
        double ra = (agg_w[a] > EPS)      ? cost_i[active[a]]      / agg_w[a]      : 1e18;
        double rb = (agg_w[b_idx] > EPS)  ? cost_i[active[b_idx]]  / agg_w[b_idx]  : 1e18;
        return ra > rb;
    });
    std::vector<int>    active_sorted(na);
    std::vector<double> agg_w_sorted(na);
    for (int idx = 0; idx < na; ++idx) {
        active_sorted[idx] = active[order[idx]];
        agg_w_sorted[idx]  = agg_w[order[idx]];
    }

    MkpBnBSurr bb;
    bb.active   = active_sorted.data();
    bb.cost     = cost_i;
    bb.agg_w    = agg_w_sorted.data();
    bb.u        = u.data();
    bb.A_i      = A_i;
    bb.na = na; bb.m = m; bb.n_g = n_g;
    bb.best_val = val_free;
    bb.x_bits.assign(na, 0);
    bb.best_x_bits.assign(na, 0);

    std::vector<double> rem_b(b_i, b_i + m);
    bb.run(0, rem_b.data(), val_free);

    for (int ki = 0; ki < na; ++ki)
        if (bb.best_x_bits[ki]) x_out_i[active_sorted[ki]] = 1.0;
}

} // namespace

// ── Interface C (ctypes) ──────────────────────────────────────────────────────

extern "C" {

/*
 * solve_zeta_batch — résout N_z sous-problèmes ζ indépendants.
 *
 * Paramètres (tous C-contiguous, row-major) :
 *   N_z    : nombre de scénarios libres
 *   m      : nombre de contraintes par scénario
 *   n_g    : nombre de variables libres
 *   cost   : (N_z, n_g) float64 — coûts lagrangiens
 *   A      : (N_z, m, n_g) float64 — matrices de contraintes
 *   b_res  : (N_z, m) float64 — capacités résiduelles
 *   x_out  : (N_z, n_g) float64 — solutions binaires (sortie, pré-initialisée à 0)
 */
void solve_zeta_batch(
    int N_z, int m, int n_g,
    const double* cost,
    const double* A,
    const double* b_res,
    double*       x_out
) {
    std::memset(x_out, 0, sizeof(double) * N_z * n_g);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N_z; ++i) {
        solve_one(
            n_g, m,
            cost  + i * n_g,
            A     + (size_t)i * m * n_g,
            b_res + i * m,
            x_out + i * n_g
        );
    }
}

/*
 * solve_zeta_batch_surrogate — même contrat que solve_zeta_batch (mêmes
 * paramètres, même valeur ζ_i retournée EXACTEMENT), mais borne de
 * branchement agrégée (surrogate) au lieu du minimum de m relaxations
 * séparées — cf. MkpBnBSurr ci-dessus. Fonction séparée, PAS un paramètre
 * ajouté à solve_zeta_batch : ne change rien pour les appelants existants
 * (scripts d'entraînement Python via ctypes, backends GNN natifs non batché/
 * AOTInductor).
 */
void solve_zeta_batch_surrogate(
    int N_z, int m, int n_g,
    const double* cost,
    const double* A,
    const double* b_res,
    double*       x_out
) {
    std::memset(x_out, 0, sizeof(double) * N_z * n_g);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N_z; ++i) {
        solve_one_surrogate(
            n_g, m,
            cost  + i * n_g,
            A     + (size_t)i * m * n_g,
            b_res + i * m,
            x_out + i * n_g
        );
    }
}

} // extern "C"
