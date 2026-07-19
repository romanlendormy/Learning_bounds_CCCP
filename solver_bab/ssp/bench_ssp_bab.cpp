/*
 * bench_ssp_bab.cpp — Comparaison Gecode base vs B&B v1CLD vs B&B v1LD sur SSP.
 *
 * Usage :
 *   bench_ssp_bab [E T N eps T_node time_lim]
 *
 * Instance synthétique identique à bench_ssp_v1cld2.cpp (automate jouet à 3
 * états, salaire en cloche, demande aléatoire uniforme).
 *
 * Structure alignée sur MKP (solver_bab/mkp/) : tout tourne sur Gecode::BAB<>
 * natif, pas de DFS manuelle (cf. gecode_bab_ssp.hpp pour pourquoi une
 * version DFS maison a été abandonnée — elle élaguait de façon non
 * équivalente à Gecode natif, pas fiable à l'échelle).
 *
 *   - Gecode base : solve_gecode_ssp — CP pur, heuristique générique Gecode
 *                   (size_min), sans borne Lagrangienne.
 *   - B&B v1CLD   : solve_gecode_bab_ssp(bound_mode=V1CLD) — heuristique
 *                   size_min (défaut, cf. gecode_bab_ssp.hpp) + borne v1CLD
 *                   (glouton O(E·T·Q), rapide) à chaque nœud.
 *   - B&B v1LD    : solve_gecode_bab_ssp(bound_mode=V1LD) — même heuristique
 *                   size_min + borne v1LD (ζ exact via CP-SAT, coûteux) à
 *                   chaque nœud.
 *
 * Heuristique RATIO (score demande/salaire) disponible via
 * SSPBranchHeuristic::RATIO mais plus exercée par défaut dans ce bench —
 * à réintroduire au besoin si on veut la comparer de nouveau à size_min.
 *
 * Chaque résultat est imprimé (et le flux vidé) dès qu'il est disponible, pas
 * seulement à la toute fin — utile pour suivre un run sbatch long dans le
 * fichier .out au fur et à mesure plutôt que d'attendre la fin du job.
 *
 * ATTENTION — v1LD : ζ_i est résolu EXACTEMENT via CP-SAT à CHAQUE appel
 * more() (NP-difficile) — ne sera praticable que sur de petites instances
 * (E, T, N faibles). C'est voulu : l'écart de vitesse est en soi la donnée
 * d'intérêt pour décider si v1LD est exploitable sur les instances visées.
 */

#include "gecode_bab_ssp.hpp"
#include "../../gecode_base/gecode_solver.hpp"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <tuple>
#include <vector>

// ── Génération de l'instance (identique à bench_ssp_v1cld2.cpp) ─────────────

struct BenchInstance {
    int E, T, N, Q, m;
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

// tightenings/fails_bound = -1 : borne non applicable (Gecode base) → affiché "-".
static void print_row(const std::string& variant, double t_total, double t_bound,
                      double t_pure, int64_t nodes, double obj, const std::string& status,
                      int64_t tightenings = -1, int64_t fails_bound = -1)
{
    std::cout << std::left  << std::setw(28) << variant
              << std::right << std::setw(12) << std::fixed << std::setprecision(4) << t_total
              << std::setw(12) << t_bound
              << std::setw(12) << t_pure
              << std::setw(14) << nodes
              << std::setw(14) << std::setprecision(6) << obj
              << std::setw(12);
    if (tightenings < 0) std::cout << "-";
    else                 std::cout << tightenings;
    std::cout << std::setw(14);
    if (fails_bound < 0) std::cout << "-";
    else                 std::cout << fails_bound;
    std::cout << "  " << status << "\n";
}

int main(int argc, char** argv)
{
    int    E        = (argc > 1) ? std::atoi(argv[1]) : 3;
    int    T        = (argc > 2) ? std::atoi(argv[2]) : 6;
    int    N        = (argc > 3) ? std::atoi(argv[3]) : 8;
    double eps      = (argc > 4) ? std::atof(argv[4]) : 0.10;
    int    T_node   = (argc > 5) ? std::atoi(argv[5]) : 30;
    // time_lim  : Gecode base / repro / v1CLD (0 = illimité).
    // time_lim_ld : v1LD uniquement — gardé séparé car ζ exact via CP-SAT à
    // chaque nœud le rend impraticable à convergence sur des instances non
    // triviales ; on veut pouvoir retirer le timeout des autres méthodes
    // (comparaison de nœuds à l'optimum réel) sans laisser v1LD tourner
    // indéfiniment.
    double time_lim    = (argc > 6) ? std::atof(argv[6]) : 60.0;
    double time_lim_ld = (argc > 7) ? std::atof(argv[7]) : time_lim;

    std::cout << "=== bench_ssp_bab : Gecode base vs B&B v1CLD vs B&B v1LD (heuristique size_min) ===\n";
    std::cout << "Instance : E=" << E << "  T=" << T << "  N=" << N
              << "  eps=" << eps << "  T_node=" << T_node
              << "  time_lim=" << time_lim << "s"
              << "  time_lim_v1LD=" << time_lim_ld << "s\n\n";
    std::cout << "ATTENTION : le B&B v1LD résout ζ exactement via CP-SAT à CHAQUE "
                 "nœud (NP-difficile) — ne sera praticable que sur de petites "
                 "instances.\n" << std::flush;

    auto inst = make_instance(E, T, N, eps);

    std::cout << "\n=== Résultats (imprimés au fur et à mesure) ===\n";
    std::cout << std::left  << std::setw(28) << "variante"
              << std::right << std::setw(12) << "t_total(s)"
              << std::setw(12) << "t_bound(s)"
              << std::setw(12) << "t_pure(s)"
              << std::setw(14) << "nœuds"
              << std::setw(14) << "objectif"
              << std::setw(12) << "resserr."
              << std::setw(14) << "fails_borne"
              << "  statut\n";
    std::cout << std::string(136, '-') << "\n" << std::flush;

    // ── Gecode base (CP pur, heuristique générique Gecode, sans borne) ────────
    std::vector<std::vector<int>> delta_gecode(inst.Q, std::vector<int>(inst.m, -1));
    for (auto& [src, sym, dst] : inst.transitions) delta_gecode[src][sym] = dst;
    std::vector<int> work_ind(inst.m, 0);
    work_ind[3] = 1;

    GecodeResult g_base = solve_gecode_ssp(
        inst.E, inst.T, inst.m, inst.Q, inst.N, inst.epsilon,
        work_ind, inst.salary, delta_gecode, inst.initial,
        inst.accepting, inst.demand, time_lim, /*break_symmetry=*/true);
    print_row("Gecode base (size_min)", g_base.solve_time_s, 0.0, g_base.solve_time_s,
              g_base.nodes, g_base.objective, g_base.status);
    std::cout << std::flush;

    // ── Gecode base (size_min, repro) — même modèle CP que SSPSpaceBnB (utilisé
    // par le B&B), avec notre reproduction maison de size_min/val_min dans
    // BoundBrancher (SSPBranchHeuristic::SIZE_MIN), SANS jamais appeler more().
    // Sert de test de fidélité : si notre reproduction est correcte, le nombre
    // de nœuds doit être ~identique à "Gecode base (size_min)" ci-dessus (brancher
    // natif de Gecode) — un écart signalerait un tie-break ou détail
    // d'implémentation différent entre les deux, avant de faire confiance à
    // "B&B v1CLD (size_min)" plus bas.
    GecodeSSPBabResult g_sm_repro = solve_gecode_bab_ssp(
        inst.E, inst.T, inst.N, inst.epsilon,
        inst.transitions, inst.accepting, inst.initial,
        inst.salary, inst.demand, inst.p,
        time_lim, /*T_root=*/200, T_node, SSPBoundMode::V1CLD,
        /*rel_tol=*/1e-4, /*tol_window=*/30, /*use_lagrangian_bound=*/false,
        SSPBranchHeuristic::SIZE_MIN);
    print_row("Gecode base (size_min repro)", g_sm_repro.solve_time_s, 0.0, g_sm_repro.solve_time_s,
              g_sm_repro.nodes, g_sm_repro.objective, g_sm_repro.status);
    std::cout << std::flush;

    // ── B&B v1CLD (size_min, défaut + borne glouton O(E·T·Q), rapide) ─────────
    // Comparé à "Gecode base (size_min repro)" ci-dessus (même heuristique,
    // sans borne), l'écart de nœuds isole l'effet de la borne v1CLD seule.
    GecodeSSPBabResult r_cld = solve_gecode_bab_ssp(
        inst.E, inst.T, inst.N, inst.epsilon,
        inst.transitions, inst.accepting, inst.initial,
        inst.salary, inst.demand, inst.p,
        time_lim, /*T_root=*/200, T_node, SSPBoundMode::V1CLD);
    print_row("B&B v1CLD", r_cld.solve_time_s, r_cld.t_bound_total_s,
              r_cld.t_bab_pure_s, r_cld.nodes, r_cld.objective, r_cld.status,
              r_cld.bound_tightenings, r_cld.fails_from_bound);
    std::cout << std::flush;

    // ── B&B v1LD (size_min, défaut + borne exacte CP-SAT, coûteux) ────────────
    // T_root/T_node volontairement bas : chaque itération de sous-gradient fait
    // N résolutions CP-SAT (une par scénario) — même 10 itérations à la racine
    // peuvent représenter des dizaines/centaines de résolutions NP-difficiles.
    GecodeSSPBabResult r_ld = solve_gecode_bab_ssp(
        inst.E, inst.T, inst.N, inst.epsilon,
        inst.transitions, inst.accepting, inst.initial,
        inst.salary, inst.demand, inst.p,
        time_lim_ld, /*T_root=*/10, std::min(T_node, 3), SSPBoundMode::V1LD);
    print_row("B&B v1LD", r_ld.solve_time_s, r_ld.t_bound_total_s,
              r_ld.t_bab_pure_s, r_ld.nodes, r_ld.objective, r_ld.status,
              r_ld.bound_tightenings, r_ld.fails_from_bound);
    std::cout << std::flush;

    // ── Sanity check : les 3 méthodes doivent converger vers le même optimum ──
    // Une méthode qui n'a pas atteint l'optimalité prouvée (time-out) n'est
    // comparée qu'à titre informatif — ce n'est pas une incohérence, juste
    // une recherche incomplète (attendu pour v1LD/v1CLD sur une grande instance).
    std::cout << "\n=== Sanity check (même optimum ?) ===\n";
    bool ok = true;
    auto report = [&](const char* label, const GecodeSSPBabResult& r) {
        if (!g_base.feasible || !r.feasible) return;
        double diff = std::abs(g_base.objective - r.objective);
        bool both_optimal = g_base.optimal && r.optimal;
        std::string tag = !both_optimal ? "[TIMEOUT — comparaison informative]"
                          : (diff < 1e-6 ? "[PASS]" : "[FAIL]");
        std::cout << "  |Gecode base - " << label << "| = " << diff << "  " << tag << "\n";
        if (both_optimal) ok = ok && (diff < 1e-6);
    };
    report("Gecode base (size_min repro)", g_sm_repro);
    report("B&B v1CLD", r_cld);
    report("B&B v1LD ", r_ld);
    std::cout << (ok ? "\n=== OK ===\n" : "\n=== ATTENTION : incohérence détectée ===\n")
              << std::flush;

    return ok ? 0 : 1;
}
