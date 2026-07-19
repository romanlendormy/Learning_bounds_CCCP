/*
 * bench_ssp_bab_file.cpp — Comparaison Gecode base vs B&B base_ref vs B&B
 *                           v1CLD vs B&B v1LD sur une instance SSP chargée
 *                           depuis fichiers texte (générée par ssp_stochastic,
 *                           cf. export_to_txt.py).
 *
 * Contrairement à bench_ssp_bab.cpp (automate jouet à 3 états, demande i.i.d.
 * sans structure temporelle), cette instance vient du générateur calibré
 * (demande bimodale AR(1), automate à contraintes réalistes, E calibré par
 * quantile de demande) — cf. discussion sur le déséquilibre de tightness de
 * l'instance jouet.
 *
 * Usage :
 *   bench_ssp_bab_file <répertoire instance> [T_node time_lim time_lim_v1LD run_gecode_base]
 *
 * time_lim_v1LD < 0 désactive v1LD (coûteux — CP-SAT à chaque nœud) ; noter
 * que 0 est déjà pris ailleurs dans le code pour "illimité", d'où la
 * sentinelle négative dédiée.
 *
 * run_gecode_base = 0 désactive la variante "Gecode base" (coûteuse sur les
 * instances difficiles, redondante avec B&B base_ref qui prouve déjà
 * l'optimum sans borne) — dans ce cas B&B base_ref sert de référence pour le
 * sanity check et le calcul du gap racine. Défaut 1 (activé).
 *
 * <répertoire instance> doit contenir instance.txt, automaton.txt, cost.txt,
 * demand.txt (cf. ssp_loader.hpp).
 *
 * En plus du tableau lisible (print_row), chaque méthode exécutée imprime une
 * ligne "CSV_ROW,..." (cf. print_csv_row) pour un parsing automatisé par un
 * script de sweep, sans dépendre du format tabulaire aligné.
 */

#include "gecode_bab_ssp.hpp"
#include "ssp_loader.hpp"
#include "../../gecode_base/gecode_solver.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

// Gap relatif (%) entre la borne calculée au nœud racine (avant tout
// branchement) et l'optimum réel (donné par Gecode base, référence fiable
// vérifiée par le sanity check plus bas) : 100 * (true_opt - root_bound) / true_opt.
// >= 0 pour une borne valide (root_bound <= true_opt) ; une valeur négative
// signale une borne racine déjà invalide (root_bound > true_opt).
static double root_gap_pct(double true_opt, double root_bound_value)
{
    if (std::isnan(root_bound_value) || !(true_opt > 0.0)) return std::nan("");
    return 100.0 * (true_opt - root_bound_value) / true_opt;
}

static void print_row(const std::string& variant, double t_total, double t_bound,
                      double t_pure, int64_t nodes, double obj, const std::string& status,
                      int64_t tightenings = -1, int64_t fails_bound = -1,
                      double gap_root = std::nan(""))
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
    std::cout << std::setw(14);
    if (std::isnan(gap_root)) std::cout << "-";
    else                      std::cout << std::setprecision(4) << gap_root;
    std::cout << "  " << status << "\n";
}

// Ligne machine-lisible, une par méthode exécutée — cf. en-tête du fichier.
// Champs : CSV_ROW,dir,E,T,N,eps,variant,status,optimal,feasible,nodes,fails,
//          t_total,t_bound,t_pure,tightenings,fails_bound,objective,gap_root_pct
static void print_csv_row(const std::string& dir, int E, int T, int N, double eps,
                          const std::string& variant,
                          double t_total, double t_bound, double t_pure,
                          int64_t nodes, int64_t fails, double obj,
                          const std::string& status, bool optimal, bool feasible,
                          int64_t tightenings = -1, int64_t fails_bound = -1,
                          double gap_root = std::nan(""))
{
    std::cout << "CSV_ROW," << dir << "," << E << "," << T << "," << N << "," << eps << ","
              << variant << "," << status << ","
              << (optimal ? 1 : 0) << "," << (feasible ? 1 : 0) << ","
              << nodes << "," << fails << ","
              << std::fixed << std::setprecision(6)
              << t_total << "," << t_bound << "," << t_pure << ","
              << tightenings << "," << fails_bound << "," << obj << ","
              << gap_root << "\n";
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: bench_ssp_bab_file <répertoire instance> "
                     "[T_node time_lim time_lim_v1LD run_gecode_base]\n";
        return 1;
    }
    std::string dir = argv[1];
    int    T_node       = (argc > 2) ? std::atoi(argv[2]) : 30;
    double time_lim      = (argc > 3) ? std::atof(argv[3]) : 60.0;
    double time_lim_ld   = (argc > 4) ? std::atof(argv[4]) : time_lim;
    bool   run_gecode_base = (argc > 5) ? (std::atoi(argv[5]) != 0) : true;

    SSPInstance inst = load_ssp_instance(dir);

    // Q = nombre d'états du DFA (déduit comme dans SSPCLDProblem/SSPShiftProblem)
    int Q = inst.initial + 1;
    for (auto& [src, sym, dst] : inst.transitions)
        Q = std::max({Q, src + 1, dst + 1});
    for (int q : inst.accepting)
        Q = std::max(Q, q + 1);
    const int m = 4;  // alphabet {O,P,R,W}

    std::cout << "=== bench_ssp_bab_file : " << dir << " ===\n";
    std::cout << "Instance : E=" << inst.E << "  T=" << inst.T << "  N=" << inst.N
              << "  eps=" << inst.epsilon << "  Q=" << Q
              << "  n_trans=" << inst.transitions.size()
              << "  T_node=" << T_node
              << "  time_lim=" << time_lim << "s"
              << "  time_lim_v1LD=" << time_lim_ld << "s\n\n";

    std::cout << std::left  << std::setw(28) << "variante"
              << std::right << std::setw(12) << "t_total(s)"
              << std::setw(12) << "t_bound(s)"
              << std::setw(12) << "t_pure(s)"
              << std::setw(14) << "nœuds"
              << std::setw(14) << "objectif"
              << std::setw(12) << "resserr."
              << std::setw(14) << "fails_borne"
              << std::setw(14) << "gap_racine(%)"
              << "  statut\n";
    std::cout << std::string(150, '-') << "\n" << std::flush;

    // ── Gecode base (CP pur, heuristique générique Gecode, sans borne) ────────
    // Désactivable (run_gecode_base=0) — cf. en-tête. B&B base_ref (ci-dessous)
    // sert alors de référence pour le sanity check / gap racine.
    GecodeResult g_base{};
    if (run_gecode_base) {
        std::vector<std::vector<int>> delta_gecode(Q, std::vector<int>(m, -1));
        for (auto& [src, sym, dst] : inst.transitions) delta_gecode[src][sym] = dst;
        std::vector<int> work_ind(m, 0);
        work_ind[3] = 1;

        g_base = solve_gecode_ssp(
            inst.E, inst.T, m, Q, inst.N, inst.epsilon,
            work_ind, inst.salary, delta_gecode, inst.initial,
            inst.accepting, inst.demand, time_lim, /*break_symmetry=*/true);
        print_row("Gecode base (size_min)", g_base.solve_time_s, 0.0, g_base.solve_time_s,
                  g_base.nodes, g_base.objective, g_base.status);
        print_csv_row(dir, inst.E, inst.T, inst.N, inst.epsilon, "gecode_base_native",
                      g_base.solve_time_s, 0.0, g_base.solve_time_s,
                      g_base.nodes, g_base.fails, g_base.objective, g_base.status,
                      g_base.optimal, g_base.feasible);
        std::cout << std::flush;
    } else {
        std::cout << "Gecode base : désactivé (run_gecode_base=0) — référence = B&B base_ref\n"
                  << std::flush;
    }

    // ── B&B base_ref (size_min repro, SANS borne) — référence isolant l'effet ─
    // de la borne v1CLD seule : même B&B custom (Gecode::BAB<> + BoundBrancher
    // SIZE_MIN), même branchement que v1CLD ci-dessous, mais more() ne renvoie
    // jamais de borne (use_lagrangian_bound=false). Cf. g_sm_repro dans
    // bench_ssp_bab.cpp.
    GecodeSSPBabResult r_base_ref = solve_gecode_bab_ssp(
        inst.E, inst.T, inst.N, inst.epsilon,
        inst.transitions, inst.accepting, inst.initial,
        inst.salary, inst.demand, inst.p,
        time_lim, /*T_root=*/200, T_node, SSPBoundMode::V1CLD,
        /*rel_tol=*/1e-4, /*tol_window=*/30, /*use_lagrangian_bound=*/false,
        SSPBranchHeuristic::SIZE_MIN);
    print_row("B&B base_ref (size_min)", r_base_ref.solve_time_s, 0.0, r_base_ref.solve_time_s,
              r_base_ref.nodes, r_base_ref.objective, r_base_ref.status);
    print_csv_row(dir, inst.E, inst.T, inst.N, inst.epsilon, "base_ref",
                  r_base_ref.solve_time_s, 0.0, r_base_ref.solve_time_s,
                  r_base_ref.nodes, r_base_ref.fails, r_base_ref.objective, r_base_ref.status,
                  r_base_ref.optimal, r_base_ref.feasible);
    std::cout << std::flush;

    // Référence "optimum vrai" pour gap racine / sanity check : Gecode base si
    // exécuté, sinon B&B base_ref (preuve d'optimalité elle aussi sans borne).
    const double  ref_objective = run_gecode_base ? g_base.objective : r_base_ref.objective;
    const bool    ref_optimal   = run_gecode_base ? g_base.optimal   : r_base_ref.optimal;
    const bool    ref_feasible  = run_gecode_base ? g_base.feasible  : r_base_ref.feasible;
    const char*   ref_label     = run_gecode_base ? "Gecode base" : "B&B base_ref";

    // ── B&B v1CLD (size_min, borne glouton O(E·T·Q)) ──────────────────────────
    GecodeSSPBabResult r_cld = solve_gecode_bab_ssp(
        inst.E, inst.T, inst.N, inst.epsilon,
        inst.transitions, inst.accepting, inst.initial,
        inst.salary, inst.demand, inst.p,
        time_lim, /*T_root=*/200, T_node, SSPBoundMode::V1CLD);
    double gap_cld = root_gap_pct(ref_objective, r_cld.root_bound_value);
    print_row("B&B v1CLD", r_cld.solve_time_s, r_cld.t_bound_total_s,
              r_cld.t_bab_pure_s, r_cld.nodes, r_cld.objective, r_cld.status,
              r_cld.bound_tightenings, r_cld.fails_from_bound, gap_cld);
    print_csv_row(dir, inst.E, inst.T, inst.N, inst.epsilon, "v1cld",
                  r_cld.solve_time_s, r_cld.t_bound_total_s,
                  r_cld.t_bab_pure_s, r_cld.nodes, r_cld.fails, r_cld.objective, r_cld.status,
                  r_cld.optimal, r_cld.feasible, r_cld.bound_tightenings, r_cld.fails_from_bound,
                  gap_cld);
    std::cout << std::flush;

    // ── B&B v1LD (size_min, borne exacte CP-SAT, coûteux) — sentinelle
    // time_lim_ld < 0 : désactivé (cf. en-tête du fichier).
    bool ran_ld = (time_lim_ld >= 0.0);
    GecodeSSPBabResult r_ld{};
    if (ran_ld) {
        r_ld = solve_gecode_bab_ssp(
            inst.E, inst.T, inst.N, inst.epsilon,
            inst.transitions, inst.accepting, inst.initial,
            inst.salary, inst.demand, inst.p,
            time_lim_ld, /*T_root=*/10, std::min(T_node, 3), SSPBoundMode::V1LD);
        double gap_ld = root_gap_pct(ref_objective, r_ld.root_bound_value);
        print_row("B&B v1LD", r_ld.solve_time_s, r_ld.t_bound_total_s,
                  r_ld.t_bab_pure_s, r_ld.nodes, r_ld.objective, r_ld.status,
                  r_ld.bound_tightenings, r_ld.fails_from_bound, gap_ld);
        print_csv_row(dir, inst.E, inst.T, inst.N, inst.epsilon, "v1ld",
                      r_ld.solve_time_s, r_ld.t_bound_total_s,
                      r_ld.t_bab_pure_s, r_ld.nodes, r_ld.fails, r_ld.objective, r_ld.status,
                      r_ld.optimal, r_ld.feasible, r_ld.bound_tightenings, r_ld.fails_from_bound,
                      gap_ld);
        std::cout << std::flush;
    } else {
        std::cout << "B&B v1LD : désactivé (time_lim_v1LD < 0)\n" << std::flush;
    }

    // ── Sanity check ───────────────────────────────────────────────────────────
    std::cout << "\n=== Sanity check (même optimum ?) ===\n";
    bool ok = true;
    auto report = [&](const char* label, const GecodeSSPBabResult& r) {
        if (!ref_feasible || !r.feasible) return;
        double diff = std::abs(ref_objective - r.objective);
        bool both_optimal = ref_optimal && r.optimal;
        std::string tag = !both_optimal ? "[TIMEOUT — comparaison informative]"
                          : (diff < 1e-6 ? "[PASS]" : "[FAIL]");
        std::cout << "  |" << ref_label << " - " << label << "| = " << diff << "  " << tag << "\n";
        if (both_optimal) ok = ok && (diff < 1e-6);
    };
    if (run_gecode_base) report("B&B base_ref", r_base_ref);
    report("B&B v1CLD", r_cld);
    if (ran_ld) report("B&B v1LD ", r_ld);
    std::cout << (ok ? "\n=== OK ===\n" : "\n=== ATTENTION : incohérence détectée ===\n")
              << std::flush;

    return ok ? 0 : 1;
}
