/*
 * bench_ssp_v1cld.cpp — Benchmark de la descente de sous-gradient v1CLD
 *                        sur une instance SSP générée en mémoire.
 *
 * Analogue de bench_ssp.cpp (v1LD), même instance, même Gecode de référence.
 * Différences clés :
 *   - SSPCLDProblem  : m = T contraintes (une par période), solve_zeta en O(E·T·Q)
 *   - solve_v1cld    : λ_{i,j} ∈ R^d par (scénario i, période j)
 *   - Borne obtenue  : v1CLD_star ≤ v1LD_star  (plus serrée ou égale)
 *
 * Usage : ./bench_ssp_v1cld [E T N iters [out.csv]]
 *
 * Instance DFA identique à bench_ssp.cpp :
 *   3 états {0=pré-service, 1=en-service, 2=post-service}
 *   Transitions : 0-O->0, 0-P->0, 0-W->1, 1-W->1, 1-R->2, 2-R->2, 2-O->2
 *   Acceptants  : {0, 1, 2}
 */

#include "ssp_cld_problem.hpp"
#include "ssp_problem.hpp"
#include "v1cld_solver.hpp"
#include "v1ld_solver.hpp"
#include "../../gecode_base/gecode_solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <vector>

// ── Génération de l'instance (identique à bench_ssp.cpp) ─────────────────────

struct BenchInstance {
    int E, T, N;
    int Q, m;
    double epsilon;
    std::vector<std::tuple<int,int,int>> transitions;
    std::vector<int> accepting;
    int initial;
    Vec1d salary;
    Vec2i demand;
    Vec1d p;
};

static BenchInstance make_instance(int E, int T, int N, double eps, int seed = 42)
{
    BenchInstance inst;
    inst.E = E; inst.T = T; inst.N = N; inst.epsilon = eps;
    inst.Q = 3; inst.m = 4;
    inst.initial = 0;
    inst.accepting = {0, 1, 2};

    inst.transitions = {
        {0, 0, 0},   // 0 --O--> 0
        {0, 1, 0},   // 0 --P--> 0
        {0, 3, 1},   // 0 --W--> 1
        {1, 3, 1},   // 1 --W--> 1
        {1, 2, 2},   // 1 --R--> 2
        {2, 2, 2},   // 2 --R--> 2
        {2, 0, 2},   // 2 --O--> 2
    };

    inst.salary.resize(T);
    for (int t = 0; t < T; ++t) {
        double x = 2.0 * t / (T - 1) - 1.0;
        inst.salary[t] = 1.0 + 4.0 * std::exp(-4.0 * x * x);
    }

    std::mt19937 rng(seed);
    int max_d = std::max(1, E / 2);
    std::uniform_int_distribution<int> dist(0, max_d);
    inst.demand.assign(N, Vec1i(T));
    for (int i = 0; i < N; ++i)
        for (int t = 0; t < T; ++t)
            inst.demand[i][t] = dist(rng);

    inst.p.assign(N, 1.0 / N);
    return inst;
}

// ── Affichage de la courbe de convergence ─────────────────────────────────────

static void print_convergence(const V1CLDResult& res, int log_interval)
{
    std::cout << "\n--- Courbe de convergence (log toutes les "
              << log_interval << " iter) ---\n";
    std::cout << std::setw(6)  << "iter"
              << std::setw(10) << "t (s)"
              << std::setw(16) << "borne (courant)"
              << std::setw(14) << "meilleure"
              << std::setw(12) << "||g||"
              << "\n";
    std::cout << std::string(58, '-') << "\n";

    double running_best = -std::numeric_limits<double>::infinity();
    for (int k = 0; k < (int)res.convergence.size(); ++k) {
        double cost_bound = -res.convergence[k];
        running_best = std::max(running_best, cost_bound);
        std::cout << std::setw(6)  << k * log_interval
                  << std::fixed << std::setprecision(3)
                  << std::setw(10) << res.times_s[k]
                  << std::setw(16) << cost_bound
                  << std::setw(14) << running_best
                  << std::setw(12) << res.gnorm_history[k]
                  << "\n";
    }
}

// ── Écriture du CSV de convergence ────────────────────────────────────────────

static void write_csv(const std::string& path, const V1CLDResult& res,
                      int log_interval, int E, int T, int N, double eps,
                      double optimal_cost, double gecode_time_s)
{
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "Impossible d'écrire " << path << "\n";
        return;
    }
    f << "# bench_ssp_v1cld  E=" << E << " T=" << T << " N=" << N
      << " eps=" << eps << " iters=" << res.iterations_run
      << " time_s=" << std::fixed << std::setprecision(3) << res.solve_time_s
      << " cost_lower=" << std::setprecision(6) << -res.v1CLD_star
      << " optimal=" << std::setprecision(6) << optimal_cost
      << " gecode_time_s=" << std::setprecision(3) << gecode_time_s << "\n";
    f << "iter,time_s,cost_bound,best_cost,gnorm\n";

    double best = -std::numeric_limits<double>::infinity();
    for (int k = 0; k < (int)res.convergence.size(); ++k) {
        double cost_bound = -res.convergence[k];
        best = std::max(best, cost_bound);
        f << k * log_interval << ","
          << std::fixed << std::setprecision(6) << res.times_s[k] << ","
          << cost_bound << ","
          << best << ","
          << res.gnorm_history[k] << "\n";
    }
    std::cout << "CSV écrit : " << path << "\n";
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    int         E       = 5;
    int         T       = 8;
    int         N       = 15;
    int         iters   = 5000;
    double      eps     = 0.10;
    std::string csvfile = "bench_ssp_v1cld.csv";

    if (argc >= 2) E       = std::atoi(argv[1]);
    if (argc >= 3) T       = std::atoi(argv[2]);
    if (argc >= 4) N       = std::atoi(argv[3]);
    if (argc >= 5) iters   = std::atoi(argv[4]);
    if (argc >= 6) csvfile = argv[5];

    std::cout << "=== Benchmark SSP v1CLD ===\n";
    std::cout << "Instance  : E=" << E << "  T=" << T
              << "  N=" << N << "  ε=" << eps << "\n";
    std::cout << "n_vars    : " << E * T * 4 << "  (E×T×4)\n";
    std::cout << "m (contraintes) : " << T << "  (une par période)\n";
    std::cout << "λ shape   : N×m×d = " << N << "×" << T
              << "×" << E * T * 4 << "\n\n";

    auto inst = make_instance(E, T, N, eps);

    std::cout << "Salary[t] :";
    for (double s : inst.salary)
        std::cout << "  " << std::fixed << std::setprecision(2) << s;
    std::cout << "\n";

    double mean_d = 0.0; int max_d = 0;
    for (int i = 0; i < N; ++i)
        for (int t = 0; t < T; ++t) {
            mean_d += inst.demand[i][t];
            max_d   = std::max(max_d, inst.demand[i][t]);
        }
    mean_d /= N * T;
    std::cout << "Demande   : moyenne=" << std::fixed << std::setprecision(2)
              << mean_d << "  max=" << max_d << "\n\n";

    // ── Optimal exact Gecode ─────────────────────────────────────────────────
    std::vector<std::vector<int>> delta_gecode(inst.Q, std::vector<int>(inst.m, -1));
    for (auto [src, sym, dst] : inst.transitions)
        delta_gecode[src][sym] = dst;
    std::vector<int> work_ind(inst.m, 0);
    work_ind[3] = 1;

    std::cout << "Résolution exacte Gecode (BAB)...\n";
    GecodeResult gres = solve_gecode_ssp(
        inst.E, inst.T, inst.m, inst.Q, inst.N, inst.epsilon,
        work_ind, inst.salary, delta_gecode, inst.initial,
        inst.accepting, inst.demand, /*time_lim=*/300.0, /*break_symmetry=*/true);
    std::cout << "  Status  : " << gres.status << "\n";
    std::cout << "  Optimal : " << std::fixed << std::setprecision(6) << gres.objective
              << "  (" << std::setprecision(3) << gres.solve_time_s << " s, "
              << gres.nodes << " nœuds, " << gres.fails << " fails)\n\n";

    // ── Construction du problème v1CLD ───────────────────────────────────────
    SSPCLDProblem prob_cld(
        inst.transitions, inst.accepting, inst.initial,
        inst.salary, inst.demand, inst.p, inst.epsilon, inst.E);

    // ── Sanity-check rapide : θ et ζ à λ=0 ──────────────────────────────────
    {
        auto [vt, xt] = prob_cld.solve_theta(0, prob_cld.c());
        auto [vz, xz] = prob_cld.solve_zeta(0, 0, prob_cld.c());
        std::cout << "Sanity (λ=0, période j=0) :\n";
        std::cout << "  θ(j=0, c) = " << std::fixed << std::setprecision(4) << vt << "\n";
        std::cout << "  ζ(i=0, j=0, c) = " << vz
                  << "   δ = " << std::max(0.0, vt - vz) << "\n";

        // Cross-check avec v1LD : θ_CLD(j=0, c) doit = θ_LD(c)
        SSPShiftProblem prob_ld(
            inst.transitions, inst.accepting, inst.initial,
            inst.salary, inst.demand, inst.p, inst.epsilon, inst.E);
        auto [vt_ld, xt_ld] = prob_ld.solve_theta(prob_ld.c());
        std::cout << "  θ_LD(c)   = " << std::setprecision(4) << vt_ld
                  << "  (doit ≈ θ_CLD ci-dessus)\n\n";
    }

    // ── Descente de sous-gradient v1CLD ──────────────────────────────────────
    const int log_interval = 1;
    double v_lower_polyak = gres.feasible ? -gres.objective : -1e18;

    std::cout << "Lancement solve_v1cld (max " << iters << " iter"
              << ", Polyak v_lower=" << std::fixed << std::setprecision(4)
              << v_lower_polyak
              << (gres.feasible ? " [Gecode réalisable]" : " [FALLBACK 1/√t]")
              << ")...\n";

    V1CLDResult res = solve_v1cld(
        prob_cld,
        iters,
        log_interval,
        /*verbose=*/false,
        /*time_limit_s=*/3600.0,
        /*rel_tol=*/-1.0,
        /*tol_window=*/iters,
        /*v_lower=*/v_lower_polyak,
        /*lambda_init=*/{}
    );

    // ── Résumé ────────────────────────────────────────────────────────────────
    const double cost_lower = -res.v1CLD_star;
    std::cout << "\n=== Résultat v1CLD ===\n";
    std::cout << "  Borne inf. coût salarial : " << std::fixed << std::setprecision(6)
              << cost_lower << "  (= −v1CLD*)\n";
    std::cout << "  Itérations  : " << res.iterations_run << "\n";
    std::cout << "  Temps total : " << std::fixed << std::setprecision(3)
              << res.solve_time_s << " s\n";
    std::cout << "  Temps/iter  : " << std::fixed << std::setprecision(4)
              << res.solve_time_s / res.iterations_run << " s\n";

    if (!res.convergence.empty()) {
        double first_cost = -res.convergence.front();
        double tightening = cost_lower - first_cost;
        std::cout << "  Borne iter 0 : " << std::fixed << std::setprecision(4)
                  << first_cost << "\n";
        std::cout << "  Resserrement : +" << std::fixed << std::setprecision(4)
                  << tightening << "  (+" << std::setprecision(1)
                  << 100.0 * tightening / std::abs(first_cost) << " %)\n";
    }

    if (gres.feasible) {
        double gap     = gres.objective - cost_lower;
        double gap_pct = 100.0 * gap / gres.objective;
        std::cout << "  Gap v1CLD : " << std::fixed << std::setprecision(4)
                  << gap << "  (" << std::setprecision(1) << gap_pct << " %)\n";
    }

    print_convergence(res, log_interval);
    write_csv(csvfile, res, log_interval, E, T, N, eps,
              gres.objective, gres.solve_time_s);

    return 0;
}
