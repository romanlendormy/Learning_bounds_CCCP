/*
 * campaign_sld_dld_vs_ref.cpp — Campagne de benchmark CPU comparant SLD
 * et DLD (bornes lagrangiennes par sous-gradient, à la racine — mask
 * entièrement libre, PAS de B&B déroulé) sur les instances SSP de
 * test_paper_trillium/ssp/, avec un B&B REF exact (pruning CP pur, moteur
 * natif Gecode::BAB) comme référence d'optimalité pour calculer le gap.
 *
 * Ces instances n'ont AUCUN optimum connu au préalable (calibrées par un
 * probe Gecode de 5s visant une fenêtre de nombre de nœuds, pas résolues à
 * l'optimalité — cf. test_paper_trillium/ssp/README.md) : c'est pourquoi le
 * B&B REF est intégré ici plutôt que supposé disponible ailleurs.
 *
 * Usage :
 *   campaign_sld_dld_vs_ref <instance_dir> [T=100] [time_limit_ref_s=1800] [time_limit_sg_s=1800]
 *
 * Ordre des phases et écriture CSV EN DEUX TEMPS (décision utilisateur
 * 2026-08-22, suite à la découverte ci-dessous) : REF puis DLD sont
 * calculés et ÉCRITS (ligne CSV_ROW imprimée + stdout flushé) AVANT de
 * tenter SLD, qui passe donc en DERNIER. Si SLD reste bloqué (cf.
 * ci-dessous) et que la tâche SLURM se fait tuer par son time limit, la
 * ligne CSV_ROW déjà écrite (REF+DLD, SLD marqué "PENDING") survit sur
 * disque -- seule la partie SLD de cette instance manque, pas tout. Si
 * SLD termine normalement (ou est sauté), une SECONDE ligne CSV_ROW
 * (mêmes colonnes, sld_status="RUN"/"SKIPPED_N_GT_100" à la place de
 * "PENDING") est imprimée à la toute fin : le script de fusion doit garder
 * la DERNIÈRE ligne CSV_ROW par fichier (la plus complète).
 *

 * <instance_dir> doit contenir instance.txt/automaton.txt/cost.txt/demand.txt
 * (cf. ssp_loader.hpp), typiquement test_paper_trillium/ssp/E{E}_T{T}_N{N}/rep{r}/.
 *
 * Convention de signe (vérifiée dans gecode_bab_ssp.hpp:28-31,
 * debug_root_bound.cpp, debug_root_bound_sld.cpp) : le problème est interne
 * en max(−salaire) ⇒ lb_salary = −SLD_star (resp. −DLD_star) est une
 * borne INFÉRIEURE valide sur le coût salarial réel minimisé. gap_pct =
 * 100·(ref_objective − lb_salary)/ref_objective.
 *
 * DLD utilise solve_dld_parallel (solver_generic/dld_solver_parallel.hpp,
 * ζ/θ threadé OpenMP, validé bit-identique à solve_dld séquentiel par
 * debug_dld_parallel_check) plutôt que solve_dld (original, séquentiel,
 * partagé avec MKP) — pour bénéficier du threading sur cette campagne sans
 * toucher au fichier partagé.
 *
 * Sortie : une ligne lisible par phase (REF/SLD/DLD) + une ligne
 * "CSV_ROW,..." unique (même convention que bench_ssp_bab_file.cpp) :
 *
 *   instance_dir,E,T,N,rep,
 *   ref_status,ref_proven,ref_objective,ref_time_s,ref_nodes,
 *   sld_status,sld_time_s,sld_iterations_run,sld_lb_salary,sld_gap_pct,sld_n_stationary,
 *   dld_time_s,dld_iterations_run,dld_lb_salary,dld_gap_pct,dld_n_stationary,dld_t_zeta_theta_s
 *
 * SLD est SAUTÉ (sld_status=SKIPPED_N_GT_100) si N > SLD_MAX_N : décision
 * prise le 2026-08-22 après avoir constaté empiriquement (smoke test,
 * E8_T12_N500/rep0) que solve_zeta pour SLD (solver_generic/ssp/ssp_problem.cpp)
 * appelle OR-Tools CP-SAT SANS limite de temps interne ("Exact... NP-hard
 * exact. Validité de la borne SLD exige l'optimalité exacte.",
 * ssp_problem.hpp) -- une seule itération peut alors rester bloquée
 * indéfiniment (constaté >90s sans terminer la 1ère itération sur N=500,
 * alors que REF prouve l'optimal en 10s), et le contrôle time_limit_s de
 * solve_sld ne vérifie l'horloge qu'ENTRE deux itérations complètes, donc ne
 * peut pas interrompre un appel CP-SAT en cours.
 *
 * MAIS la restriction N<=100 seule ne suffit PAS : reconstaté empiriquement
 * (smoke test, E4_T10_N100/rep0, N=100) qu'un combo "dur" (fenêtre de
 * calibration relevée [1e6,1e8] pour E=4,T=10/12) peut aussi bloquer SLD
 * bien au-delà du budget demandé, même à N=100 -- d'où la réorganisation
 * "écrire REF+DLD avant de tenter SLD" ci-dessus : la restriction
 * SLD_MAX_N réduit le risque (et le coût moyen) mais ne l'élimine pas,
 * l'écriture anticipée est ce qui protège contre la perte de données.
 * DLD reste évalué sur toute la grille (sa propre boucle ζ/θ est un DP
 * polynomial, pas un solve exact sans borne de temps).
 */
#include "gecode_bab_ssp.hpp"
#include "ssp_loader.hpp"
#include "../../solver_generic/ssp/ssp_problem_bnb.hpp"
#include "../../solver_generic/ssp/ssp_dld_problem_bnb.hpp"
#include "../../solver_generic/sld_solver.hpp"
#include "../../solver_generic/dld_solver_parallel.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>

// Cf. docstring en tête de fichier : au-delà de cette taille, solve_zeta
// (SLD, CP-SAT exact sans limite de temps interne) risque de bloquer
// indéfiniment sur une seule itération -- SLD n'est donc évalué que pour
// N <= SLD_MAX_N (correspond exactement aux tailles N=50/100 de la grille
// test_paper_trillium ; N=500 est sauté).
constexpr int SLD_MAX_N = 100;

// Nombre d'itérations t≥1 telles que |conv[t]-conv[t-1]| / max(|conv[t]|,1)
// < rel_thresh -- seuil validé avec l'utilisateur (2026-08-22) pour détecter
// le plateau typique de la borne SSP.
static int count_stationary(const Vec1d& conv, double rel_thresh = 1e-5) {
    int cnt = 0;
    for (size_t t = 1; t < conv.size(); ++t) {
        double denom = std::max(std::abs(conv[t]), 1.0);
        if (std::abs(conv[t] - conv[t - 1]) / denom < rel_thresh) ++cnt;
    }
    return cnt;
}

// rep{N} -> N ; -1 si le nom du dossier ne suit pas ce format.
static int parse_rep_index(const std::string& dirname) {
    if (dirname.rfind("rep", 0) != 0) return -1;
    try { return std::stoi(dirname.substr(3)); } catch (...) { return -1; }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "Usage: %s <instance_dir> [T=100] [time_limit_ref_s=1800] [time_limit_sg_s=1800]\n",
            argv[0]);
        return 2;
    }
    std::string instance_dir   = argv[1];
    int    T                   = (argc > 2) ? std::atoi(argv[2]) : 100;
    // 30 min PAR PHASE (REF, SLD, DLD indépendamment -- PAS un budget total
    // partagé) : décision utilisateur 2026-08-22. Ne couvre PAS le cas d'un
    // seul appel CP-SAT SLD bloqué en cours d'itération (cf. docstring) --
    // c'est l'écriture anticipée de REF+DLD (voir plus bas) qui protège
    // contre la perte de données dans ce cas, pas ce budget.
    double time_limit_ref_s    = (argc > 3) ? std::atof(argv[3]) : 1800.0;
    double time_limit_sg_s     = (argc > 4) ? std::atof(argv[4]) : 1800.0;

    SSPInstance inst = load_ssp_instance(instance_dir);
    // p.filename() est vide si instance_dir se termine par '/' -- on retombe
    // alors sur le composant précédent (nom du dossier "repN" réel).
    std::filesystem::path p(instance_dir);
    std::string rep_dirname = p.filename().empty()
                              ? p.parent_path().filename().string()
                              : p.filename().string();
    int rep = parse_rep_index(rep_dirname);

    std::printf("[campaign] instance_dir=%s  E=%d T=%d N=%d eps=%g  rep=%d  T=%d  time_limit_ref_s=%.0f  time_limit_sg_s=%.0f\n",
                instance_dir.c_str(), inst.E, inst.T, inst.N, inst.epsilon, rep, T,
                time_limit_ref_s, time_limit_sg_s);

    // ── Phase 1 : REF (B&B exact, pruning CP pur, moteur Gecode::BAB natif) ──
    // Mirroir exact de la branche "ref" de test_gnn_dld_bab.cpp : REF
    // court-circuite more() (aucun calcul/pose de borne), use_lagrangian_bound
    // n'a donc aucun effet de coût ici, gnn_client=nullptr jamais déréférencé.
    GecodeSSPBabResult ref_res = solve_gecode_bab_ssp(
        inst.E, inst.T, inst.N, inst.epsilon,
        inst.transitions, inst.accepting, inst.initial,
        inst.salary, inst.demand, inst.p,
        /*time_lim=*/time_limit_ref_s,
        /*T_root=*/200, /*T_node=*/30,
        /*bound_mode=*/SSPBoundMode::REF,
        /*rel_tol=*/1e-4, /*tol_window=*/30,
        /*use_lagrangian_bound=*/true,
        /*branch_heuristic=*/SSPBranchHeuristic::SIZE_MIN,
        /*gnn_client=*/nullptr);

    bool   ref_has_incumbent = ref_res.feasible;
    bool   ref_proven        = (ref_res.status == "OPTIMAL");
    double ref_objective     = ref_has_incumbent ? ref_res.objective
                                                  : std::numeric_limits<double>::quiet_NaN();
    std::printf("[campaign][REF]   status=%s  objective=%.6f  proven=%d  nodes=%lld  time_s=%.2f\n",
                ref_res.status.c_str(), ref_res.objective, (int)ref_proven,
                (long long)ref_res.nodes, ref_res.solve_time_s);
    std::fflush(stdout);

    auto gap_pct_of = [&](double lb_salary) -> double {
        if (!ref_has_incumbent) return std::numeric_limits<double>::quiet_NaN();
        double denom = std::max(std::abs(ref_objective), 1e-9);
        return 100.0 * (ref_objective - lb_salary) / denom;
    };

    Vec2i mask(inst.E, Vec1i(inst.T, -1));

    // ── Phase 2 : DLD (racine, même mask) — variante parallèle OpenMP ─────
    // Calculée AVANT SLD (cf. docstring) : sa boucle ζ/θ est un DP
    // polynomial, jamais bloquant comme le CP-SAT de SLD.
    SSPDLDProblemBnB dld_prob(inst.transitions, inst.accepting, inst.initial,
                                 inst.salary, inst.demand, inst.p, inst.epsilon,
                                 inst.E, mask);
    DLDResult dld_res = solve_dld_parallel(dld_prob, T, 1, false,
                                                  time_limit_sg_s, -1.0, 30,
                                                  -1e18, /*lambda_init=*/{});
    double dld_lb_salary    = -dld_res.DLD_star;
    double dld_gap_pct      = gap_pct_of(dld_lb_salary);
    int    dld_n_stationary = count_stationary(dld_res.convergence);
    std::printf("[campaign][DLD] iterations_run=%d  lb_salary=%.6f  gap_pct=%.4f  n_stationary=%d  time_s=%.3f  t_zeta_theta_s=%.3f\n",
                dld_res.iterations_run, dld_lb_salary, dld_gap_pct, dld_n_stationary,
                dld_res.solve_time_s, dld_res.t_zeta_theta_s);

    // ── Ligne CSV (même schéma imprimé deux fois -- cf. docstring) ──────────
    auto print_csv_row = [&](const std::string& sld_status, double sld_time_s,
                              int sld_iters_run, double sld_lb_salary,
                              double sld_gap_pct, int sld_n_stationary) {
        std::printf("CSV_ROW,%s,%d,%d,%d,%d,"
                    "%s,%d,%.6f,%.3f,%lld,"
                    "%s,%.3f,%d,%.6f,%.4f,%d,"
                    "%.3f,%d,%.6f,%.4f,%d,%.3f\n",
                    instance_dir.c_str(), inst.E, inst.T, inst.N, rep,
                    ref_res.status.c_str(), (int)ref_proven, ref_objective, ref_res.solve_time_s,
                    (long long)ref_res.nodes,
                    sld_status.c_str(), sld_time_s, sld_iters_run, sld_lb_salary, sld_gap_pct,
                    sld_n_stationary,
                    dld_res.solve_time_s, dld_res.iterations_run, dld_lb_salary, dld_gap_pct,
                    dld_n_stationary, dld_res.t_zeta_theta_s);
        std::fflush(stdout);
    };

    // Ligne PARTIELLE (REF+DLD déjà sûrs, SLD pas encore tenté) --
    // survit sur disque si la tâche se fait tuer pendant SLD.
    print_csv_row("PENDING", 0.0, 0, std::numeric_limits<double>::quiet_NaN(),
                   std::numeric_limits<double>::quiet_NaN(), 0);

    // ── Phase 3 : SLD (racine, même mask) — EN DERNIER (cf. docstring) ─────
    // Sauté si N > SLD_MAX_N.
    std::string sld_status       = "SKIPPED_N_GT_100";
    double      sld_time_s       = 0.0;
    int         sld_iters_run    = 0;
    double      sld_lb_salary    = std::numeric_limits<double>::quiet_NaN();
    double      sld_gap_pct      = std::numeric_limits<double>::quiet_NaN();
    int         sld_n_stationary = 0;

    if (inst.N <= SLD_MAX_N) {
        SSPProblemBnB sld_prob(inst.transitions, inst.accepting, inst.initial,
                                 inst.salary, inst.demand, inst.p, inst.epsilon,
                                 inst.E, mask);
        SLDResult sld_res = solve_sld(sld_prob, T, /*log_interval=*/1, /*verbose=*/false,
                                          time_limit_sg_s, /*rel_tol=*/-1.0, /*tol_window=*/30,
                                          /*v_lower=*/-1e18, /*lambda_init=*/{});
        sld_status       = "RUN";
        sld_time_s       = sld_res.solve_time_s;
        sld_iters_run    = sld_res.iterations_run;
        sld_lb_salary    = -sld_res.SLD_star;
        sld_gap_pct      = gap_pct_of(sld_lb_salary);
        sld_n_stationary = count_stationary(sld_res.convergence);
        std::printf("[campaign][SLD]  iterations_run=%d  lb_salary=%.6f  gap_pct=%.4f  n_stationary=%d  time_s=%.3f\n",
                    sld_iters_run, sld_lb_salary, sld_gap_pct, sld_n_stationary, sld_time_s);
    } else {
        std::printf("[campaign][SLD]  SKIPPED (N=%d > SLD_MAX_N=%d -- cf. docstring : CP-SAT "
                     "exact par scenario sans limite de temps interne, risque de blocage)\n",
                     inst.N, SLD_MAX_N);
    }

    // Ligne FINALE (complète) -- le script de fusion garde la dernière
    // ligne CSV_ROW par fichier.
    print_csv_row(sld_status, sld_time_s, sld_iters_run, sld_lb_salary,
                   sld_gap_pct, sld_n_stationary);

    return 0;
}
