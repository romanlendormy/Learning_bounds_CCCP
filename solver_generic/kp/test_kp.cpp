/*
 * test_kp.cpp — Tests unitaires pour KPProblem / KPProblemDP.
 *
 * Instance jouet : n=8 items, N=3 scénarios, poids/capacités entiers (pour
 * que KPProblemDP soit exact et comparable au B&B sans erreur d'arrondi).
 *
 * Tests :
 *   T1. EXACTITUDE theta : solve_theta == argmax brute-force (x_k=1 ssi cost_k>0)
 *   T2. EXACTITUDE zeta (BB)  : solve_zeta (KPProblem) == brute-force 2^n par scénario
 *   T3. EXACTITUDE zeta (DP)  : solve_zeta (KPProblemDP) == brute-force 2^n par scénario
 *   T4. BB == DP sur instance entière : mêmes valeurs optimales (aux ties près)
 *   T5. δ = θ - ζ ≥ 0 (X^i ⊆ domaine de base, cf. mkp_subproblems.hpp)
 *   T6. COÛTS NÉGATIFS : pas de crash, brute-force toujours retrouvé
 *
 * Compilation et exécution (depuis solver_generic/) :
 *   cmake --build build_cc --target test_kp
 *   ./build_cc/test_kp
 */

#include "kp_problem.hpp"
#include "kp_problem_dp.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#define PASS "\033[32m[PASS]\033[0m"
#define FAIL "\033[31m[FAIL]\033[0m"

static int n_pass = 0, n_fail = 0;

static void check(bool cond, const std::string& name) {
    if (cond) { std::cout << PASS " " << name << "\n"; ++n_pass; }
    else      { std::cout << FAIL " " << name << "\n"; ++n_fail; }
}

static bool close(double a, double b, double tol = 1e-6) {
    return std::fabs(a - b) <= tol;
}

// ── Brute force : max cost^T x, x in {0,1}^n, a_row^T x <= b ────────────────
static double brute_zeta(const Vec1d& cost, const Vec1d& a_row, double b) {
    int n = (int)cost.size();
    double best = 0.0;   // x=0 toujours faisable
    for (int mask = 0; mask < (1 << n); ++mask) {
        double w = 0.0, v = 0.0;
        for (int k = 0; k < n; ++k)
            if (mask & (1 << k)) { w += a_row[k]; v += cost[k]; }
        if (w <= b + 1e-9 && v > best) best = v;
    }
    return best;
}

static double brute_theta(const Vec1d& cost) {
    double v = 0.0;
    for (double c : cost) if (c > 0.0) v += c;
    return v;
}

int main() {
    // n=8 items, poids/capacités entiers, N=3 scénarios
    Vec1d c = {5, 3, 8, 2, 7, 4, 6, 1};
    Vec1d p = {0.5, 0.3, 0.2};
    Vec2d A = {
        {3, 2, 5, 1, 4, 3, 4, 1},   // scénario 0
        {4, 3, 6, 2, 5, 3, 5, 2},   // scénario 1 (plus contraignant)
        {2, 1, 4, 1, 3, 2, 3, 1},   // scénario 2 (plus lâche)
    };
    Vec1d b = {10.0, 8.0, 12.0};
    double epsilon = 0.1;

    KPProblem   bb(c, p, A, b, epsilon);
    KPProblemDP dp(c, p, A, b, epsilon);

    // T1 — theta
    auto [theta_val, theta_x] = bb.solve_theta(c);
    check(close(theta_val, brute_theta(c)), "T1 theta == brute-force");

    // T2/T3/T4/T5 — zeta par scénario
    bool t2_ok = true, t3_ok = true, t4_ok = true, t5_ok = true;
    for (int i = 0; i < bb.N_scen(); ++i) {
        auto [zbb_val, zbb_x] = bb.solve_zeta(i, c);
        auto [zdp_val, zdp_x] = dp.solve_zeta(i, c);
        double zref = brute_zeta(c, A[i], b[i]);

        if (!close(zbb_val, zref)) t2_ok = false;
        if (!close(zdp_val, zref)) t3_ok = false;
        if (!close(zbb_val, zdp_val)) t4_ok = false;
        if (theta_val - zbb_val < -1e-9) t5_ok = false;
    }
    check(t2_ok, "T2 zeta (BB) == brute-force sur chaque scénario");
    check(t3_ok, "T3 zeta (DP) == brute-force sur chaque scénario");
    check(t4_ok, "T4 BB == DP (instance entière)");
    check(t5_ok, "T5 delta = theta - zeta >= 0");

    // T6 — coûts négatifs (typique en cours de sous-gradient : cost = c - lambda^T A)
    Vec1d c_neg = {-2, 3, -1, 5, -4, 2, -3, 1};
    auto [theta_neg, x_theta_neg] = bb.solve_theta(c_neg);
    bool t6_ok = close(theta_neg, brute_theta(c_neg));
    for (int i = 0; i < bb.N_scen(); ++i) {
        auto [zbb_neg, x_neg] = bb.solve_zeta(i, c_neg);
        if (!close(zbb_neg, brute_zeta(c_neg, A[i], b[i]))) t6_ok = false;
    }
    check(t6_ok, "T6 couts negatifs : BB toujours exact");

    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail == 0 ? 0 : 1;
}
