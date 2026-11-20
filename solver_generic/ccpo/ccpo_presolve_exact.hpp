// ccpo_presolve_exact.hpp — presolve CCPO EXACT et CERTIFIÉ.
//
// Port C++ ligne à ligne de gnn/ccpo/presolve_box_exact.py::box_min_certified (validé sur 3000 cas
// aléatoires contre l'énumération des 3^n ensembles actifs, et sur 3,74 M scénarios de collect_campaign_v2).
//
// Problème (relaxation boîte d'un scénario) : v* = min q'Σq  s.c.  Σq = Q, 0 <= q <= U.
// Remplace presolve_min_variance_box (ccpo_sld.hpp), dont l'ensemble actif ne relâche jamais une variable
// fixée et peut SURESTIMER v* (scénarios satisfaisables écartés à tort).
//
//  1. Ensemble actif primal standard : un blocage par pas, relâchement du multiplicateur le plus négatif,
//     arrêt sur KKT -> optimum exact (QP strictement convexe).
//  2. Valeur renvoyée = valeur DUALE d(μ,α,β) = μQ - UΣβ - ¼ w'Σ⁻¹w (w = μ1+α-β, α,β >= 0) :
//     dualité faible => d <= v* pour TOUT μ et α,β >= 0, donc aucune surestimation possible, même si
//     l'étape 1 s'arrête trop tôt ; d = v* à l'optimum.
//  Si Σ (ou un bloc) n'est pas numériquement définie positive : borne = -inf (scénario jamais écarté : prudent).
//
// Sélection : presolve EXACT par défaut (campagne de l'article). L'ancien presolve (peut écarter à tort)
// reste accessible par CCPO_PRESOLVE=legacy, pour reproduire des résultats antérieurs.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace ccpo_presolve_exact {

using Mat = std::vector<std::vector<double>>;
using Vec = std::vector<double>;

// Cholesky STRICT : false si un pivot <= 0 (pas de garde-fou silencieux, contrairement à cholesky()).
inline bool chol_strict(const Mat& M, Mat& L) {
    const int m = static_cast<int>(M.size());
    L.assign(m, Vec(m, 0.0));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j <= i; ++j) {
            double s = M[i][j];
            for (int t = 0; t < j; ++t) s -= L[i][t] * L[j][t];
            if (i == j) {
                if (!(s > 0.0)) return false;
                L[i][i] = std::sqrt(s);
            } else {
                L[i][j] = s / L[j][j];
            }
        }
    return true;
}

inline Vec chol_solve(const Mat& L, const Vec& b) {
    const int m = static_cast<int>(b.size());
    Vec y(m), x(m);
    for (int i = 0; i < m; ++i) {
        double s = b[i];
        for (int t = 0; t < i; ++t) s -= L[i][t] * y[t];
        y[i] = s / L[i][i];
    }
    for (int i = m - 1; i >= 0; --i) {
        double s = y[i];
        for (int t = i + 1; t < m; ++t) s -= L[t][i] * x[t];
        x[i] = s / L[i][i];
    }
    return x;
}

inline Vec grad(const Mat& S, const Vec& q) {  // 2Σq
    const int n = static_cast<int>(q.size());
    Vec g(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int k = 0; k < n; ++k) s += S[i][k] * q[k];
        g[i] = 2.0 * s;
    }
    return g;
}

// tous fixés : μ au milieu de [max g (à U), min g (à 0)]
inline double mu_all_fixed(const Vec& g, const std::vector<signed char>& st) {
    double lo = -std::numeric_limits<double>::infinity(), hi = std::numeric_limits<double>::infinity();
    bool has_lo = false, has_hi = false;
    for (size_t i = 0; i < g.size(); ++i) {
        if (st[i] == 2) { lo = has_lo ? std::max(lo, g[i]) : g[i]; has_lo = true; }
        if (st[i] == 1) { hi = has_hi ? std::min(hi, g[i]) : g[i]; has_hi = true; }
    }
    if (has_lo && has_hi) return 0.5 * (lo + hi);
    return has_lo ? lo : hi;
}

// Borne inférieure certifiée (dualité faible) au point q, état st (0 libre, 1 à 0, 2 à U).
inline double dual_value(const Mat& S, double Q, double U, const Vec& q, const std::vector<signed char>& st) {
    const int n = static_cast<int>(q.size());
    Mat L;
    if (!chol_strict(S, L)) return -std::numeric_limits<double>::infinity();
    const Vec g = grad(S, q);
    auto d = [&](double mu) {
        Vec w(n);
        double sum_beta = 0.0;
        for (int i = 0; i < n; ++i) {
            const double alpha = (st[i] == 1) ? std::max(g[i] - mu, 0.0) : 0.0;
            const double beta = (st[i] == 2) ? std::max(mu - g[i], 0.0) : 0.0;
            sum_beta += beta;
            w[i] = mu + alpha - beta;
        }
        const Vec x = chol_solve(L, w);
        double wx = 0.0;
        for (int i = 0; i < n; ++i) wx += w[i] * x[i];
        return mu * Q - U * sum_beta - 0.25 * wx;
    };
    double mu0 = 0.0;
    int nfree = 0;
    for (int i = 0; i < n; ++i)
        if (st[i] == 0) { mu0 += g[i]; ++nfree; }
    mu0 = nfree ? mu0 / nfree : mu_all_fixed(g, st);
    double best = d(mu0);
    for (double scale : {1e-6, 1e-4, 1e-2})
        for (int s : {-1, 1}) best = std::max(best, d(mu0 * (1.0 + s * scale)));
    return best;
}

struct BoxMinResult {
    double lower;    // borne inférieure certifiée de v* (à comparer à B)
    double primal;   // q'Σq au point réalisable final (majorant de v*)
    int iterations;
    bool optimal;    // KKT atteint
};

inline BoxMinResult box_min_certified(int n, double Q, double U, const Mat& S) {
    const double inf = std::numeric_limits<double>::infinity();
    if (Q > n * U + 1e-12 || Q < -1e-12) return {inf, inf, 0, true};  // boîte vide : aucun q
    const int max_iter = 50 * (n + 1);
    Vec q(n, Q / n);
    std::vector<signed char> st(n, 0);
    bool optimal = false;
    int it = 0;
    for (; it < max_iter; ++it) {
        std::vector<int> fr, fx;
        for (int i = 0; i < n; ++i) (st[i] == 0 ? fr : fx).push_back(i);
        Vec target = q;
        if (!fr.empty()) {
            const int m = static_cast<int>(fr.size());
            Mat M(m, Vec(m));
            Vec r(m, 0.0), ones(m, 1.0);
            double R = Q;
            for (int i : fx) R -= q[i];
            for (int a = 0; a < m; ++a) {
                for (int b = 0; b < m; ++b) M[a][b] = S[fr[a]][fr[b]];
                for (int i : fx) r[a] += S[fr[a]][i] * q[i];
            }
            Mat L;
            if (!chol_strict(M, L)) break;  // non PD : on s'arrête, la borne duale reste valide
            const Vec e = chol_solve(L, ones), b = chol_solve(L, r);
            double se = 0.0, sb = 0.0;
            for (int a = 0; a < m; ++a) { se += e[a]; sb += b[a]; }
            const double mu = 2.0 * (R + sb) / se;
            for (int a = 0; a < m; ++a) target[fr[a]] = mu / 2.0 * e[a] - b[a];
        }
        Vec p(n);
        double pmax = 0.0;
        for (int i = 0; i < n; ++i) { p[i] = target[i] - q[i]; pmax = std::max(pmax, std::fabs(p[i])); }
        if (pmax <= 1e-12 * std::max(1.0, U)) {
            const Vec g = grad(S, q);
            double mu;
            if (!fr.empty()) { mu = 0.0; for (int i : fr) mu += g[i]; mu /= fr.size(); }
            else mu = mu_all_fixed(g, st);
            int j = 0; double mj = std::numeric_limits<double>::infinity(), gmax = 0.0;
            for (int i = 0; i < n; ++i) {
                const double mult = (st[i] == 1) ? g[i] - mu : (st[i] == 2) ? mu - g[i] : 0.0;
                if (mult < mj) { mj = mult; j = i; }
                gmax = std::max(gmax, std::fabs(g[i]));
            }
            if (fx.empty() || mj >= -1e-10 * std::max(1.0, gmax)) { optimal = true; break; }
            st[j] = 0;  // relâcher UNE contrainte
            continue;
        }
        double t = 1.0; int block = -1;
        for (int i : fr) {
            if (p[i] < 0 && q[i] + p[i] < 0) {
                const double ti = -q[i] / p[i];
                if (ti < t) { t = ti; block = i; }
            } else if (p[i] > 0 && q[i] + p[i] > U) {
                const double ti = (U - q[i]) / p[i];
                if (ti < t) { t = ti; block = i; }
            }
        }
        for (int i = 0; i < n; ++i) q[i] += t * p[i];
        if (block >= 0) {
            if (p[block] < 0) { q[block] = 0.0; st[block] = 1; }
            else { q[block] = U; st[block] = 2; }
        }
        for (int i = 0; i < n; ++i) q[i] = std::min(std::max(q[i], 0.0), U);
    }
    double primal = 0.0;
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k) primal += S[i][k] * q[i] * q[k];
    return {dual_value(S, Q, U, q, st), primal, it + 1, optimal};
}

// Mode : true = exact (DÉFAUT) ; false seulement si CCPO_PRESOLVE=legacy (ou "0"). Toute autre valeur -> exact.
inline bool use_exact_from_env() {
    const char* e = std::getenv("CCPO_PRESOLVE");
    return !(e && (std::strcmp(e, "legacy") == 0 || std::strcmp(e, "0") == 0));
}

// Tolérance relative de la règle d'élimination (borne > B·(1+tol)), identique au prototype Python.
constexpr double kRelTol = 1e-9;

}  // namespace ccpo_presolve_exact
