/*
 * bench_ssp.cpp — Benchmark de la descente de sous-gradient v1LD sur une
 *                 instance SSP générée en mémoire.
 *
 * Usage : ./bench_ssp [E T N iters]   (paramètres optionnels)
 *
 * Instance générée :
 *   DFA  : 3 états  {0=pré-service, 1=en-service, 2=post-service}
 *          Alphabet {O=0, P=1, R=2, W=3}
 *          Transitions : 0-O->0, 0-P->0, 0-W->1, 1-W->1, 1-R->2,
 *                        2-R->2, 2-O->2
 *          Acceptants : {2}  (l'employé doit avoir travaillé puis reposé)
 *   Salary[t]  : pic au milieu, généré de façon déterministe
 *   demand[i][t]: tiré aléatoirement (seed fixé), ≤ E/2
 */

#include "ssp_problem.hpp"
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

// ── Génération de l'instance ──────────────────────────────────────────────────

struct BenchInstance {
    int E, T, N;
    int Q, m;       // nombre d'états DFA, taille alphabet
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
    inst.Q = 3; inst.m = 4;  // DFA fixe : 3 états, alphabet {O=0,P=1,R=2,W=3}
    inst.initial = 0;
    // Acceptants : {0, 1, 2} — évite l'infaisabilité structurelle à la dernière période
    // (un employé travaillant à t=T-1 finit en état 1, non-acceptant si on n'inclut que {2})
    inst.accepting = {0, 1, 2};

    // DFA à 3 états (identique au DFA des tests unitaires)
    inst.transitions = {
        {0, 0, 0},   // 0 --O--> 0  (rester hors-service)
        {0, 1, 0},   // 0 --P--> 0  (préparation)
        {0, 3, 1},   // 0 --W--> 1  (début de service)
        {1, 3, 1},   // 1 --W--> 1  (continuer à travailler)
        {1, 2, 2},   // 1 --R--> 2  (pause/repos)
        {2, 2, 2},   // 2 --R--> 2  (continuer à reposer)
        {2, 0, 2},   // 2 --O--> 2  (hors-service après repos)
    };

    // Salaire : profil en cloche centré sur T/2
    inst.salary.resize(T);
    for (int t = 0; t < T; ++t) {
        double x = 2.0 * t / (T - 1) - 1.0;  // ∈ [-1, 1]
        inst.salary[t] = 1.0 + 4.0 * std::exp(-4.0 * x * x);
    }

    // Demande : aléatoire, seed fixé, ≤ E/2  par période et scénario
    std::mt19937 rng(seed);
    int max_d = std::max(1, E / 2);
    std::uniform_int_distribution<int> dist(0, max_d);
    inst.demand.assign(N, Vec1i(T));
    for (int i = 0; i < N; ++i)
        for (int t = 0; t < T; ++t)
            inst.demand[i][t] = dist(rng);

    // Probabilités uniformes
    inst.p.assign(N, 1.0 / N);

    return inst;
}

// ── Affichage de la courbe de convergence ─────────────────────────────────────

static void print_convergence(const V1LDResult& res, int log_interval)
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

    // Borne inf. sur le coût = −L1  : s'améliore en augmentant → running max
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

static void write_csv(const std::string& path, const V1LDResult& res,
                      int log_interval, int E, int T, int N, double eps,
                      double optimal_cost, double gecode_time_s)
{
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "Impossible d'écrire " << path << "\n";
        return;
    }
    // En-tête avec métadonnées en commentaire
    // Les colonnes cost_bound et best_cost sont −L1 : borne sur le coût min original (positive).
    // optimal = coût optimal exact Gecode (BAB sans borne ajoutée).
    f << "# bench_ssp  E=" << E << " T=" << T << " N=" << N
      << " eps=" << eps << " iters=" << res.iterations_run
      << " time_s=" << std::fixed << std::setprecision(3) << res.solve_time_s
      << " cost_lower=" << std::setprecision(6) << -res.v1LD_star
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
    // Paramètres par défaut (raisonnables, ~30-90s selon la machine)
    int         E        = 5;
    int         T        = 8;
    int         N        = 15;
    int         iters    = 5000;    // max itérations ; arrêt anticipé par patience
    double      eps      = 0.10;
    double      abs_tol  = 0.01;   // précision absolue sur la borne (cost_lower)
    int         patience = 20;     // patience : nb iter stagnantes avant arrêt
    std::string csvfile  = "bench_ssp.csv";

    if (argc >= 2) E        = std::atoi(argv[1]);
    if (argc >= 3) T        = std::atoi(argv[2]);
    if (argc >= 4) N        = std::atoi(argv[3]);
    if (argc >= 5) iters    = std::atoi(argv[4]);
    if (argc >= 6) csvfile  = argv[5];

    std::cout << "=== Benchmark SSP v1LD ===\n";
    std::cout << "Instance  : E=" << E << "  T=" << T
              << "  N=" << N << "  ε=" << eps << "\n";
    std::cout << "n_vars    : " << E * T * 4 << "  (E×T×4)\n";
    std::cout << "Arrêt     : précision=" << abs_tol
              << "  patience=" << patience
              << "  (max " << iters << " iter)\n\n";

    auto inst = make_instance(E, T, N, eps);

    // Affichage du salary pour vérification
    std::cout << "Salary[t] :";
    for (double s : inst.salary)
        std::cout << "  " << std::fixed << std::setprecision(2) << s;
    std::cout << "\n";

    // Quelques stats sur la demande
    double mean_d = 0.0;
    int    max_d  = 0;
    for (int i = 0; i < N; ++i)
        for (int t = 0; t < T; ++t) {
            mean_d += inst.demand[i][t];
            max_d   = std::max(max_d, inst.demand[i][t]);
        }
    mean_d /= N * T;
    std::cout << "Demande   : moyenne=" << std::fixed << std::setprecision(2)
              << mean_d << "  max=" << max_d << "\n\n";

    // ── Optimal exact Gecode (BAB sans borne ajoutée) ────────────────────────
    // Convertit les transitions {src,sym,dst} en table delta[Q][m].
    std::vector<std::vector<int>> delta(inst.Q, std::vector<int>(inst.m, -1));
    for (auto [src, sym, dst] : inst.transitions)
        delta[src][sym] = dst;
    std::vector<int> work_ind(inst.m, 0);
    work_ind[3] = 1;  // W=3 est le seul symbole de travail dans cette instance

    std::cout << "Résolution exacte Gecode (BAB)...\n";
    GecodeResult gres = solve_gecode_ssp(
        inst.E, inst.T, inst.m, inst.Q, inst.N, inst.epsilon,
        work_ind, inst.salary, delta, inst.initial,
        inst.accepting, inst.demand, /*time_lim=*/300.0, /*break_symmetry=*/true);
    std::cout << "  Status  : " << gres.status << "\n";
    std::cout << "  Optimal : " << std::fixed << std::setprecision(6) << gres.objective
              << "  (" << std::setprecision(3) << gres.solve_time_s << " s, "
              << gres.nodes << " nœuds, " << gres.fails << " fails)\n\n";

    SSPShiftProblem prob(
        inst.transitions, inst.accepting, inst.initial,
        inst.salary, inst.demand, inst.p, inst.epsilon, inst.E);

    // ── Sanity-check rapide : θ et ζ à λ=0 ──────────────────────────────────
    {
        auto [vt, xt] = prob.solve_theta(prob.c());
        auto [vz, xz] = prob.solve_zeta(0, prob.c());
        std::cout << "Sanity (λ=0) :\n";
        std::cout << "  θ(c) = " << std::fixed << std::setprecision(4) << vt << "\n";
        std::cout << "  ζ_0(c) = " << vz << "   δ_0 = " << std::max(0.0, vt - vz) << "\n\n";
    }

    // ── Descente de sous-gradient ─────────────────────────────────────────────
    const int log_interval = 1;   // une ligne par itération dans le CSV

    // Référence de Polyak : v_lower = −obj_gecode (borne inférieure sur v1LD*).
    // Valide dès qu'une solution réalisable existe (pas besoin de preuve d'optimalité) :
    //   obj_gecode ≥ optimal  →  −obj_gecode ≤ −optimal ≤ v1LD*  (dualité faible).
    // Si Gecode timeout sans solution : fallback 1/√t — mais le pas 1/√t est
    // dangereux avec N grand (amplification ×N dans cost_i = c + λ_i/p_i).
    double v_lower_polyak = gres.feasible ? -gres.objective : -1e18;

    std::cout << "Lancement solve_v1ld (précision=" << abs_tol
              << ", patience=" << patience << ", max=" << iters << " iter"
              << ", Polyak v_lower=" << std::fixed << std::setprecision(4)
              << v_lower_polyak
              << (gres.feasible ? " [Gecode réalisable]" : " [FALLBACK 1/√t]")
              << ")...\n";

    V1LDResult res = solve_v1ld(
        prob,
        iters,
        log_interval,
        /*verbose=*/false,
        /*time_limit_s=*/3600.0,
        /*rel_tol=*/-1.0,
        /*tol_window=*/iters,
        /*v_lower=*/v_lower_polyak,
        /*lambda_init=*/{},
        /*abs_tol=*/abs_tol,
        /*abs_patience=*/patience
    );

    // ── Résumé ────────────────────────────────────────────────────────────────
    // Le solveur est max-natif (c[e,t,W] = −salary), donc v1LD_star ≤ 0.
    // La borne sur le coût min original est −v1LD_star ≥ 0.
    const double cost_lower = -res.v1LD_star;
    std::cout << "\n=== Résultat ===\n";
    std::cout << "  Borne inf. coût salarial : " << std::fixed << std::setprecision(6)
              << cost_lower << "  (= −v1LD*)\n";
    std::cout << "  Itérations  : " << res.iterations_run << "\n";
    std::cout << "  Temps total : " << std::fixed << std::setprecision(3)
              << res.solve_time_s << " s\n";
    std::cout << "  Temps/iter  : " << std::fixed << std::setprecision(4)
              << res.solve_time_s / res.iterations_run << " s\n";

    if (!res.convergence.empty()) {
        // first/last sur la borne coût (positive, décroissante vers le bas)
        double first_cost = -res.convergence.front();
        double best_cost  = cost_lower;
        // Une borne inférieure s'améliore en augmentant (se rapproche du vrai min par dessous)
        double tightening = best_cost - first_cost;   // > 0 si la borne est plus haute
        std::cout << "  Borne iter 0 : " << std::fixed << std::setprecision(4)
                  << first_cost << "\n";
        std::cout << "  Resserrement : +" << std::fixed << std::setprecision(4)
                  << tightening << "  (+" << std::setprecision(1)
                  << 100.0 * tightening / std::abs(first_cost) << " %)\n";
    }

    std::cout << "  Gap v1LD : " << std::fixed << std::setprecision(4)
              << (gres.objective - cost_lower) << "  ("
              << std::setprecision(1)
              << 100.0 * (gres.objective - cost_lower) / gres.objective << " %)\n";

    print_convergence(res, log_interval);
    write_csv(csvfile, res, log_interval, E, T, N, eps,
              gres.objective, gres.solve_time_s);

    return 0;
}
