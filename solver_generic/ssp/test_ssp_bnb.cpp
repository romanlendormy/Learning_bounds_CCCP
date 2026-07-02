/*
 * test_ssp_bnb.cpp — Non-régression pour SSPProblemBnB / SSPCLDProblemBnB.
 *
 * T1 : masque entièrement libre (-1 partout) ⟹ résultats rigoureusement
 *      identiques à SSPShiftProblem / SSPCLDProblem (les classes de base,
 *      déjà validées) — garde-fou contre toute régression introduite par le
 *      refactoring solve_theta/solve_zeta -> solve_theta_masked/solve_zeta_masked.
 *
 * T2 : masque partiel, validé par énumération brute (E=1, petit T) — confirme
 *      que le masque restreint bien le domaine de maximisation au sous-ensemble
 *      attendu, pas juste "ne plante pas".
 */

#include "ssp_problem.hpp"
#include "ssp_problem_bnb.hpp"
#include "ssp_cld_problem.hpp"
#include "ssp_cld_problem_bnb.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

static int n_pass = 0, n_fail = 0;

static void check(bool cond, const std::string& name) {
    if (cond) { std::cout << "[PASS] " << name << "\n"; ++n_pass; }
    else      { std::cout << "[FAIL] " << name << "\n"; ++n_fail; }
}

static bool close(double a, double b, double tol = 1e-9) {
    return std::abs(a - b) <= tol;
}

// Automate jouet à 3 états (identique à bench_ssp_v1cld2.cpp) :
//   0 --O/P--> 0,  0 --W--> 1,  1 --W--> 1,  1 --R--> 2,  2 --R/O--> 2
//   Acceptants : {0,1,2}.  Symboles : O=0, P=1, R=2, W=3.
using T3 = std::tuple<int,int,int>;
static const std::vector<T3> TRANSITIONS = {
    {0, 0, 0}, {0, 1, 0}, {0, 3, 1},
    {1, 3, 1}, {1, 2, 2},
    {2, 2, 2}, {2, 0, 2},
};
static const std::vector<int> ACCEPTING = {0, 1, 2};
static const int INITIAL = 0;

int main() {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> pert(-2.0, 2.0);

    // ── T1 : masque libre ≡ classes de base ──────────────────────────────────
    {
        const int E = 3, T = 6, N = 5;
        const double eps = 0.2;
        Vec1d salary(T);
        for (int t = 0; t < T; ++t) salary[t] = 1.0 + t * 0.5;

        Vec2i demand(N, Vec1i(T));
        std::uniform_int_distribution<int> ddist(0, E);
        for (int i = 0; i < N; ++i)
            for (int t = 0; t < T; ++t) demand[i][t] = ddist(rng) % (E + 1);
        Vec1d p(N, 1.0 / N);

        SSPShiftProblem base_ld(TRANSITIONS, ACCEPTING, INITIAL, salary, demand, p, eps, E);
        Vec2i mask_free(E, Vec1i(T, -1));
        SSPProblemBnB bnb_ld(TRANSITIONS, ACCEPTING, INITIAL, salary, demand, p, eps, E, mask_free);

        // cost = c() + perturbation aléatoire (mime un cost Lagrangien avec poids négatifs)
        Vec1d cost = base_ld.c();
        for (auto& v : cost) v += pert(rng);

        auto [vt_base, xt_base] = base_ld.solve_theta(cost);
        auto [vt_bnb,  xt_bnb ] = bnb_ld.solve_theta(cost);
        check(close(vt_base, vt_bnb), "T1a v1LD: theta(mask=libre) == theta(base)");
        check(xt_base == xt_bnb, "T1b v1LD: x*(mask=libre) == x*(base)");

        bool all_zeta_ok = true;
        for (int i = 0; i < N; ++i) {
            auto [vz_base, xz_base] = base_ld.solve_zeta(i, cost);
            auto [vz_bnb,  xz_bnb ] = bnb_ld.solve_zeta(i, cost);
            if (!close(vz_base, vz_bnb)) all_zeta_ok = false;
        }
        check(all_zeta_ok, "T1c v1LD: zeta(mask=libre) == zeta(base) pour tous les scénarios");

        SSPCLDProblem    base_cld(TRANSITIONS, ACCEPTING, INITIAL, salary, demand, p, eps, E);
        SSPCLDProblemBnB bnb_cld (TRANSITIONS, ACCEPTING, INITIAL, salary, demand, p, eps, E, mask_free);

        Vec1d cost_cld = base_cld.c();
        for (auto& v : cost_cld) v += pert(rng);

        bool all_theta_ok = true, all_zeta_cld_ok = true;
        for (int j = 0; j < T; ++j) {
            auto [vt_b, xt_b] = base_cld.solve_theta(j, cost_cld);
            auto [vt_n, xt_n] = bnb_cld.solve_theta(j, cost_cld);
            if (!close(vt_b, vt_n)) all_theta_ok = false;
            for (int i = 0; i < N; ++i) {
                auto [vz_b, xz_b] = base_cld.solve_zeta(i, j, cost_cld);
                auto [vz_n, xz_n] = bnb_cld.solve_zeta(i, j, cost_cld);
                if (!close(vz_b, vz_n)) all_zeta_cld_ok = false;
            }
        }
        check(all_theta_ok, "T1d v1CLD: theta(mask=libre) == theta(base) pour tout j");
        check(all_zeta_cld_ok, "T1e v1CLD: zeta(mask=libre) == zeta(base) pour tout (i,j)");
    }

    // ── T2 : masque partiel validé par énumération brute (E=1) ───────────────
    {
        const int E = 1, T = 4, N = 1;
        Vec1d salary = {2.0, 3.0, 1.0, 4.0};
        Vec2i demand(N, Vec1i(T, 0));  // pas de couverture active : on teste solve_theta
        Vec1d p(N, 1.0);
        const double eps = 0.5;

        SSPShiftProblem base(TRANSITIONS, ACCEPTING, INITIAL, salary, demand, p, eps, E);
        Vec1d cost = base.c();
        // perturbation déterministe pour un test reproductible
        for (int k = 0; k < (int)cost.size(); ++k) cost[k] += 0.3 * (k % 5) - 0.6;

        // Masque : force la période t=1 au symbole W (=3)
        const int t_force = 1, sym_force = 3;
        Vec2i mask(E, Vec1i(T, -1));
        mask[0][t_force] = sym_force;
        SSPProblemBnB bnb(TRANSITIONS, ACCEPTING, INITIAL, salary, demand, p, eps, E, mask);

        // Table de transition pour l'énumération brute
        int delta[3][4];
        for (auto& row : delta) for (auto& v : row) v = -1;
        for (auto& [s, sym, d] : TRANSITIONS) delta[s][sym] = d;

        double brute_best = -1e30;
        for (int seq = 0; seq < 256; ++seq) {  // 4^4 séquences
            int a[4];
            int x = seq;
            for (int t = 0; t < 4; ++t) { a[t] = x % 4; x /= 4; }
            if (a[t_force] != sym_force) continue;  // respecte le masque

            int q = INITIAL;
            bool valid = true;
            double val = 0.0;
            for (int t = 0; t < T && valid; ++t) {
                int q_next = delta[q][a[t]];
                if (q_next < 0) { valid = false; break; }
                val += cost[t * SSP_SIGMA_SIZE + a[t]];
                q = q_next;
            }
            if (!valid) continue;
            // tous les états sont acceptants ici (ACCEPTING = {0,1,2})
            if (val > brute_best) brute_best = val;
        }

        auto [vt_bnb, xt_bnb] = bnb.solve_theta(cost);
        check(close(brute_best, vt_bnb),
              "T2a v1LD masqué (t_force=1 -> W) == optimum énuméré brute-force");
        check(xt_bnb[t_force * SSP_SIGMA_SIZE + sym_force] > 0.5,
              "T2b v1LD masqué: x* respecte bien le symbole imposé à t_force");
    }

    std::cout << "\n=== " << n_pass << " PASS, " << n_fail << " FAIL ===\n";
    return n_fail == 0 ? 0 : 1;
}
