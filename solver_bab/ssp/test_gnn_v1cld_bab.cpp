/*
 * test_gnn_v1cld_bab.cpp — Validation d'intégration bout-en-bout (étape 3) :
 * un vrai B&B Gecode (solve_gecode_bab_ssp, bound_mode=GNN) sur une instance
 * réelle. Contrairement à test_gnn_v1cld_native.cpp (étape 2, un seul
 * query() standalone), ceci exerce more() DANS la recherche complète —
 * masques reconstruits par le B&B lui-même, GNN appelé à répétition,
 * potentiellement (cf. discussion 2026-08-18) après le branchement complet
 * de x (zone jamais vue à l'entraînement).
 *
 * Le critère de succès n'est PAS que la borne soit serrée partout : la
 * recherche Gecode est complète, donc même une borne GNN lâche (mais
 * valide -- déjà établi séparément) doit converger vers le VRAI optimum, au
 * prix de plus de nœuds explorés. status="OPTIMAL" + objective == opt
 * attendu (fichier instance.txt/opt connu par ailleurs) est le test qui
 * compte ici.
 *
 * Usage :
 *   test_gnn_v1cld_bab <test_dir> <checkpoint_path> <expected_opt> [time_lim_s] [mode] [device]
 * mode : "gnn" (défaut), "v1cld" ou "ref" -- "v1cld"/"ref" ignorent
 * checkpoint_path (peut être n'importe quelle chaîne, jamais lu).
 * device (mode "gnn" seulement) : "cpu" (défaut) ou "cuda" -- ajouté
 * 2026-08-20 pour isoler l'effet du device seul (binaire compilé avec
 * -DGNN_DEVICE=cuda requis, cf. solver_bab/setup_gpu_env.sh).
 * "v1cld" compare à la borne sous-gradient EXISTANTE (déjà en production) sur
 * la MÊME instance, même budget, même heuristique -- cf. discussion
 * 2026-08-19 : jamais fait avant ce test, alors que c'est la comparaison qui
 * dit si le mode GNN apporte quelque chose vs le statu quo.
 * "ref" (2026-08-19, mirroir de BoundMode::REF côté MKP) : squelette nu, AUCUNE
 * borne calculée/postée (more() court-circuite) -- isole l'effet de la borne
 * seule (écart ref vs gnn/v1cld) de celui du branchement/moteur.
 */
#include "gecode_bab_ssp.hpp"
#include "gnn_v1cld_native.hpp"
#include "../../solver_generic/ssp/ssp_loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <sys/resource.h>

// Temps CPU cumulé du processus (user+sys, tous threads -- cf. man getrusage,
// RUSAGE_SELF agrège les threads), en secondes. Ajouté 2026-08-20 : le
// wall-clock mesuré sur cpubase_bycore_* (Rorqual, nœud partagé) est pollué
// par de la contention externe (voisin bruyant, confirmé job 19302252 --
// loadavg nœud >> nb de cœurs sans que notre cgroup soit throttlé). Le temps
// CPU réellement consommé par NOS threads est lui insensible à cette latence
// d'ordonnancement, donc plus fiable pour comparer REF/V1CLD/GNN à taille
// croissante sans relancer sur un autre cluster.
static double cpu_time_now_s() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6
         + ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <test_dir> <checkpoint_path> <expected_opt> [time_lim_s] [mode]\n", argv[0]);
        return 2;
    }
    std::string test_dir = argv[1];
    std::string checkpoint_path = argv[2];
    double expected_opt = std::atof(argv[3]);
    double time_lim = (argc > 4) ? std::atof(argv[4]) : 60.0;
    std::string mode = (argc > 5) ? argv[5] : "gnn";
    std::string device = (argc > 6) ? argv[6] : "cpu";

    try {
        SSPInstance inst = load_ssp_instance(test_dir);
        std::printf("[test] instance : E=%d T=%d N=%d epsilon=%g  opt attendu=%.6f  time_lim=%.0fs  mode=%s  device=%s\n",
                    inst.E, inst.T, inst.N, inst.epsilon, expected_opt, time_lim, mode.c_str(), device.c_str());

        GecodeSSPBabResult res;
        double cpu_t0 = cpu_time_now_s();
        if (mode == "gnn") {
            GNNV1CLDNative gnn(checkpoint_path, device, inst);
            res = solve_gecode_bab_ssp(
                inst.E, inst.T, inst.N, inst.epsilon,
                inst.transitions, inst.accepting, inst.initial,
                inst.salary, inst.demand, inst.p,
                /*time_lim=*/time_lim,
                /*T_root=*/200, /*T_node=*/30,
                /*bound_mode=*/SSPBoundMode::GNN,
                /*rel_tol=*/1e-4, /*tol_window=*/30,
                /*use_lagrangian_bound=*/true,
                /*branch_heuristic=*/SSPBranchHeuristic::SIZE_MIN,
                /*gnn_client=*/&gnn);

            SSPGNNQueryTiming qt = gnn.cumulative_timing();
            double t_query_sum = qt.t_graph_build_s + qt.t_forward_s + qt.t_zeta_theta_s + qt.t_lagrangian_rest_s;
            std::printf("[gnn_timing] n_queries=%lld\n", (long long)qt.n_queries);
            std::printf("[gnn_timing] graph_build   : %8.3fs (%.1f%%)\n",
                        qt.t_graph_build_s, 100.0 * qt.t_graph_build_s / std::max(t_query_sum, 1e-9));
            std::printf("[gnn_timing] forward_gnn   : %8.3fs (%.1f%%)\n",
                        qt.t_forward_s, 100.0 * qt.t_forward_s / std::max(t_query_sum, 1e-9));
            std::printf("[gnn_timing] zeta_theta    : %8.3fs (%.1f%%)\n",
                        qt.t_zeta_theta_s, 100.0 * qt.t_zeta_theta_s / std::max(t_query_sum, 1e-9));
            std::printf("[gnn_timing] lagrangian_rest: %8.3fs (%.1f%%)\n",
                        qt.t_lagrangian_rest_s, 100.0 * qt.t_lagrangian_rest_s / std::max(t_query_sum, 1e-9));
            std::printf("[gnn_timing] somme query()  : %8.3fs\n", t_query_sum);
        } else if (mode == "v1cld") {
            res = solve_gecode_bab_ssp(
                inst.E, inst.T, inst.N, inst.epsilon,
                inst.transitions, inst.accepting, inst.initial,
                inst.salary, inst.demand, inst.p,
                /*time_lim=*/time_lim,
                /*T_root=*/200, /*T_node=*/30,
                /*bound_mode=*/SSPBoundMode::V1CLD,
                /*rel_tol=*/1e-4, /*tol_window=*/30,
                /*use_lagrangian_bound=*/true,
                /*branch_heuristic=*/SSPBranchHeuristic::SIZE_MIN,
                /*gnn_client=*/nullptr);

            // Répartition du coût V1CLD (2026-08-20), miroir de [gnn_timing] :
            // t_v1cld_problem_build_s (analogue graph_build), zeta_theta
            // (étape B, calcul θ/ζ répété sur T_iters itérations), et le
            // reste par soustraction (Λ_sum, knapsack, sous-gradient, pas de
            // descente, logging -- analogue de ce que "forward" produit côté
            // GNN : la mise à jour de λ).
            double t_rest = res.t_bound_total_s
                           - res.t_v1cld_problem_build_s - res.t_v1cld_zeta_theta_s;
            double t_sum  = std::max(res.t_bound_total_s, 1e-9);
            std::printf("[v1cld_timing] problem_build : %10.6fs (%.1f%%)\n",
                        res.t_v1cld_problem_build_s, 100.0 * res.t_v1cld_problem_build_s / t_sum);
            std::printf("[v1cld_timing] zeta_theta    : %10.6fs (%.1f%%)\n",
                        res.t_v1cld_zeta_theta_s, 100.0 * res.t_v1cld_zeta_theta_s / t_sum);
            std::printf("[v1cld_timing] lambda_update(reste) : %10.6fs (%.1f%%)\n",
                        t_rest, 100.0 * t_rest / t_sum);
            std::printf("[v1cld_timing] t_bound_total_s : %10.6fs\n", res.t_bound_total_s);
            std::printf("[v1cld_timing] iters_total=%lld  (theta+zeta calls = iters_total * N * m)\n",
                        (long long)res.v1cld_iters_total);
            std::printf("[v1cld_timing] zeta_theta_us_per_iter = %.4f\n",
                        1e6 * res.t_v1cld_zeta_theta_s / std::max(res.v1cld_iters_total, (int64_t)1));
        } else if (mode == "ref") {
            res = solve_gecode_bab_ssp(
                inst.E, inst.T, inst.N, inst.epsilon,
                inst.transitions, inst.accepting, inst.initial,
                inst.salary, inst.demand, inst.p,
                /*time_lim=*/time_lim,
                /*T_root=*/200, /*T_node=*/30,
                /*bound_mode=*/SSPBoundMode::REF,
                /*rel_tol=*/1e-4, /*tol_window=*/30,
                /*use_lagrangian_bound=*/true,
                /*branch_heuristic=*/SSPBranchHeuristic::SIZE_MIN,
                /*gnn_client=*/nullptr);
        } else {
            std::fprintf(stderr, "mode inconnu : %s (attendu gnn|v1cld|ref)\n", mode.c_str());
            return 2;
        }

        double cpu_time_s = cpu_time_now_s() - cpu_t0;

        std::printf("[test] status=%s  objective=%.6f  feasible=%d  optimal=%d\n",
                    res.status.c_str(), res.objective, res.feasible, res.optimal);
        std::printf("[test] nodes=%lld  fails=%lld  more_calls=%lld  bound_tightenings=%lld  fails_from_bound=%lld\n",
                    (long long)res.nodes, (long long)res.fails, (long long)res.more_calls,
                    (long long)res.bound_tightenings, (long long)res.fails_from_bound);
        std::printf("[test] solve_time_s=%.2f  t_bound_total_s=%.2f  t_bab_pure_s=%.2f\n",
                    res.solve_time_s, res.t_bound_total_s, res.t_bab_pure_s);
        // root_bound_value (2026-08-20) : borne au tout premier nœud (racine),
        // convention coût réel -- pour comparer la TIGHTNESS pure de la borne
        // (GNN un coup vs V1CLD/V1LD sous-gradient convergé) indépendamment du
        // nombre de nœuds explorés ensuite (question 2026-08-20 : la borne est-elle
        // trop faible, et si oui à cause du GNN ou de la difficulté de la descente
        // de sous-gradient elle-même ?).
        std::printf("[test] root_bound_value=%.6f  root_bound_time_s=%.3f\n",
                    res.root_bound_value, res.root_bound_time_s);
        // cpu_time_s (getrusage, user+sys, tous threads) : à comparer à
        // solve_time_s -- si proches, le run n'a pas été pollué par de la
        // contention externe (cf. discussion 2026-08-20, nœud partagé
        // cpubase_bycore_*) ; si cpu_time_s << solve_time_s, une partie du
        // wall-clock a été passée à attendre plutôt qu'à calculer.
        std::printf("[test] cpu_time_s=%.2f  (wall-clock/cpu ratio=%.2f)\n",
                    cpu_time_s, res.solve_time_s / std::max(cpu_time_s, 1e-9));

        if (!res.feasible) {
            std::printf("[test] === FAIL (pas de solution trouvée) ===\n");
            return 1;
        }
        double diff = res.objective - expected_opt;
        std::printf("[test] diff vs opt attendu = %.6e\n", diff);

        if (res.optimal && std::fabs(diff) < 1e-3) {
            std::printf("[test] === PASS (optimum retrouvé, mode GNN correct) ===\n");
            return 0;
        } else if (!res.optimal) {
            std::printf("[test] === INCOMPLET (timeout avant preuve d'optimalité, "
                        "augmenter time_lim) ===\n");
            return 1;
        } else {
            std::printf("[test] === FAIL (objective != opt attendu malgré status OPTIMAL -- "
                        "signe d'un bug, PAS juste une borne lâche) ===\n");
            return 1;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[test] EXCEPTION: %s\n", e.what());
        return 1;
    }
}
