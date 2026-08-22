/*
 * test_gnn_v1cld_bab_batched.cpp — Étape B3 : intégration bout-en-bout du
 * batching multi-nœuds SSP dans un vrai B&B (SSPSpaceGNNBatched +
 * GNNV1CLDNativeBatched, cf. gecode_bab_ssp_gnn_batched.hpp et
 * gnn_v1cld_native_batched.hpp) sur une instance réelle -- backend GNN réel
 * cette fois (Étape B1 utilisait un stub sans borne pour valider la DFS
 * seule ; Étape B2 a validé query_batch() isolément).
 *
 * Le critère de succès premier est la CORRECTION (converge vers le vrai
 * optimum comme les modes gnn/v1cld/ref du moteur séquentiel), le second est
 * la PERFORMANCE (le batching du forward GNN doit réduire t_forward_s et
 * donc, si tout va bien, le temps de résolution total -- cf. discussion
 * 2026-08-19 : le forward domine à 95% d'un more() en mode séquentiel).
 *
 * Usage :
 *   test_gnn_v1cld_bab_batched <test_dir> <checkpoint_path> <expected_opt> <time_lim_s> <K> [device]
 * device : "cpu" (défaut) ou "cuda" -- ajouté 2026-08-20, même motivation que
 * pour test_gnn_v1cld_bab.cpp : jamais testé en GPU jusqu'ici, alors que
 * c'est précisément en GPU que le batching amortit le mieux le coût fixe par
 * appel (cf. discussion 2026-08-20, MKP gnn_batched K=64 sur H100).
 */
#include "gecode_bab_ssp_gnn_batched.hpp"
#include "gnn_v1cld_native_batched.hpp"
#include "../../solver_generic/ssp/ssp_loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <sys/resource.h>

static double cpu_time_now_s() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6
         + ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr, "usage: %s <test_dir> <checkpoint_path> <expected_opt> <time_lim_s> <K> [device]\n", argv[0]);
        return 2;
    }
    std::string test_dir = argv[1];
    std::string checkpoint_path = argv[2];
    double expected_opt = std::atof(argv[3]);
    double time_lim = std::atof(argv[4]);
    int K = std::atoi(argv[5]);
    std::string device = (argc > 6) ? argv[6] : "cpu";

    try {
        SSPInstance inst = load_ssp_instance(test_dir);
        std::printf("[test] instance : E=%d T=%d N=%d epsilon=%g  opt attendu=%.6f  time_lim=%.0fs  K=%d  device=%s\n",
                    inst.E, inst.T, inst.N, inst.epsilon, expected_opt, time_lim, K, device.c_str());
        double cpu_t0 = cpu_time_now_s();

        GNNV1CLDNativeBatched backend(checkpoint_path, device, inst);
        GNNBatchedSSPBabResult res = solve_gecode_bab_ssp_gnn_batched(
            inst.E, inst.T, inst.N, inst.epsilon,
            inst.transitions, inst.accepting, inst.initial,
            inst.salary, inst.demand, inst.p,
            time_lim, K, &backend,
            /*branch_heuristic=*/SSPBranchHeuristic::SIZE_MIN);

        SSPGNNBatchQueryTiming qt = backend.cumulative_timing();
        double t_query_sum = qt.t_graph_build_s + qt.t_forward_s + qt.t_zeta_theta_s + qt.t_lagrangian_rest_s;
        std::printf("[gnn_timing] n_batch_calls=%lld  n_nodes_total=%lld  (moyenne %.2f nœuds/appel)\n",
                    (long long)qt.n_batch_calls, (long long)qt.n_nodes_total,
                    qt.n_batch_calls > 0 ? (double)qt.n_nodes_total / qt.n_batch_calls : 0.0);
        std::printf("[gnn_timing] graph_build   : %8.3fs (%.1f%%)\n",
                    qt.t_graph_build_s, 100.0 * qt.t_graph_build_s / std::max(t_query_sum, 1e-9));
        std::printf("[gnn_timing] forward_gnn   : %8.3fs (%.1f%%)\n",
                    qt.t_forward_s, 100.0 * qt.t_forward_s / std::max(t_query_sum, 1e-9));
        std::printf("[gnn_timing] zeta_theta    : %8.3fs (%.1f%%)\n",
                    qt.t_zeta_theta_s, 100.0 * qt.t_zeta_theta_s / std::max(t_query_sum, 1e-9));
        std::printf("[gnn_timing] lagrangian_rest: %8.3fs (%.1f%%)\n",
                    qt.t_lagrangian_rest_s, 100.0 * qt.t_lagrangian_rest_s / std::max(t_query_sum, 1e-9));
        std::printf("[gnn_timing] somme query_batch: %8.3fs\n", t_query_sum);

        std::printf("[test] status=%s  objective=%.6f  feasible=%d  optimal=%d\n",
                    res.status.c_str(), res.objective, res.feasible, res.optimal);
        std::printf("[test] nodes=%lld  fails=%lld  depth=%lld  incumbent_improvements=%lld\n",
                    (long long)res.nodes, (long long)res.fails, (long long)res.depth,
                    (long long)res.incumbent_improvements);
        std::printf("[test] solve_time_s=%.2f  K=%d  n_bucket_flushes=%lld  mean_fill=%.2f  max_pending_live=%lld\n",
                    res.solve_time_s, res.K, (long long)res.n_bucket_flushes,
                    res.mean_bucket_fill_at_flush, (long long)res.max_pending_live);
        double cpu_time_s = cpu_time_now_s() - cpu_t0;
        std::printf("[test] cpu_time_s=%.2f  (wall-clock/cpu ratio=%.2f)\n",
                    cpu_time_s, res.solve_time_s / std::max(cpu_time_s, 1e-9));

        if (!res.feasible) {
            std::printf("[test] === FAIL (pas de solution trouvée) ===\n");
            return 1;
        }
        double diff = res.objective - expected_opt;
        std::printf("[test] diff vs opt attendu = %.6e\n", diff);

        if (res.optimal && std::fabs(diff) < 1e-3) {
            std::printf("[test] === PASS (optimum retrouvé, batching correct) ===\n");
            return 0;
        } else if (!res.optimal) {
            std::printf("[test] === INCOMPLET (timeout avant preuve d'optimalité) ===\n");
            return 1;
        } else {
            std::printf("[test] === FAIL (objective != opt attendu malgré status OPTIMAL) ===\n");
            return 1;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[test] EXCEPTION: %s\n", e.what());
        return 1;
    }
}
