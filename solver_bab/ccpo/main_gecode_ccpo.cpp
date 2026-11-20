/*
 * main_gecode_ccpo.cpp — CLI pour le B&B du CCPO (modes REF et SLD).
 *
 * Charge un épisode généré par generators/ccpo (ep_XXXXXX/{instance.txt,
 * scenarios.txt}, cf. generators/ccpo/README.md §Format de sortie).
 *
 * Lecteur écrit indépendamment de generators/ccpo/writers.py (au-delà du
 * format texte lui-même) -- même esprit que le "lecteur indépendant" testé
 * côté Python (test_writers.py::test_roundtrip_*).
 *
 * Usage :
 *   bab_ccpo <répertoire épisode> <time_lim_s> [mode] [T_root] [T_node] [K] [omp_threads] [scripted_model.pt] [torch_threads] [cpu|cuda]
 *   mode : "ref" (défaut) | "sld" | "feas" | "gnn_zero" (driver batché, backend factice λ=0) | "gnn" (backend libtorch, requiert CCPO_WITH_TORCH)
 *
 * Sortie (CSV, une ligne) :
 *   dir,n,N,k,Q,mode,status,nodes,fails,fails_from_bound,depth,time_s,
 *   objective,greedy_value,more_calls,ub_tightened,root_bound_time_s,
 *   root_bound_value,t_bound_total_s
 */

#include "gecode_bab_ccpo.hpp"
#include "gecode_bab_ccpo_gnn_batched.hpp"
#ifdef CCPO_WITH_TORCH
#include "gnn_ccpo_native_batched.hpp"
#endif

#include <fstream>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

struct CCPOInstance {
    int n, N, k, Q, L, U;
    long long B;
    int eps_num, eps_den;
    std::vector<int> c_int;
    std::vector<std::vector<SigmaTriplet>> sigma;  // (N,) de (i,k,val), base 0, i<=k
};

static CCPOInstance load_ccpo_episode(const std::string& dir)
{
    CCPOInstance inst;

    {
        std::ifstream f(dir + "/instance.txt");
        if (!f) throw std::runtime_error("impossible d'ouvrir " + dir + "/instance.txt");
        f >> inst.n >> inst.N >> inst.k >> inst.Q >> inst.L >> inst.U
          >> inst.eps_num >> inst.eps_den;
        f >> inst.B;
        inst.c_int.resize(inst.n);
        for (int i = 0; i < inst.n; ++i) f >> inst.c_int[i];
        if (!f) throw std::runtime_error(dir + "/instance.txt : format invalide ou tronqué");
    }

    {
        std::ifstream f(dir + "/scenarios.txt");
        if (!f) throw std::runtime_error("impossible d'ouvrir " + dir + "/scenarios.txt");

        const int n_triplets = inst.n * (inst.n + 1) / 2;
        inst.sigma.assign(inst.N, {});

        for (int j = 0; j < inst.N; ++j) {
            int scen_id;
            f >> scen_id;
            if (!f || scen_id != j + 1)
                throw std::runtime_error(
                    dir + "/scenarios.txt : bloc " + std::to_string(j + 1) +
                    " attendu, lu " + std::to_string(scen_id) +
                    " -- fichier tronqué ou corrompu");

            // psi_int (n valeurs) : pas utilisé par REF (contrainte de risque
            // posée directement sur Sigma_int, dense), mais doit être lu pour
            // avancer correctement dans le flux.
            long long psi_dummy;
            for (int i = 0; i < inst.n; ++i) f >> psi_dummy;

            inst.sigma[j].reserve(n_triplets);
            for (int t = 0; t < n_triplets; ++t) {
                int r, cc, v;
                f >> r >> cc >> v;
                inst.sigma[j].push_back({r - 1, cc - 1, v});  // base 1 -> base 0
            }
        }
        if (!f) throw std::runtime_error(dir + "/scenarios.txt : format invalide ou tronqué");
    }

    return inst;
}

// ⌈(1-ε)N⌉ en arithmétique entière exacte -- même formule que
// generators/ccpo/txt_format.py::saa_threshold, pas de division flottante.
static int saa_threshold(int eps_num, int eps_den, int N)
{
    const long long num = static_cast<long long>(eps_den - eps_num) * N;
    return static_cast<int>((num + eps_den - 1) / eps_den);
}

#ifdef CCPO_WITH_SCIP
void ccpo_scip_install_hook();   // scip_zeta.cpp
#endif

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "Usage: bab_ccpo <répertoire épisode> <time_lim_s> "
                     "[mode=ref|sld|feas|gnn_zero|gnn] [T_root] [T_node] [K] [omp] [model.pt] [torch_threads] [cpu|cuda]\n";
        return 1;
    }
    // Réglage de zeta_solve : défaut de production (2,1), surchargeable par CCPO_ZETA_WARM / CCPO_ZETA_BRANCH
    // (lues par ccpo_zeta_config() à la première utilisation) ; sans effet sur le mode ref (pas de zeta_solve).
#ifdef CCPO_WITH_SCIP
    ccpo_scip_install_hook();
#endif
    if (ccpo_zeta_config().scip_mode == 1 && ccpo_scip_hook() == nullptr)
        std::cerr << "# ATTENTION : CCPO_ZETA_SCIP=1 demandé mais binaire compilé sans SCIP (CCPO_WITH_SCIP) -> hybride ignoré\n";
    std::cerr << "# zeta_cfg warm=" << ccpo_zeta_config().warm_mode << " branch=" << ccpo_zeta_config().branching
              << " infeas_prune=" << ccpo_zeta_config().infeas_prune << " inherit=" << ccpo_zeta_config().inherit_infeas
              << " scip=" << ccpo_zeta_config().scip_mode << " scip_nodes=" << ccpo_zeta_config().scip_nodes
              << " scip_nf0=" << ccpo_zeta_config().scip_nf0 << " scip_hook=" << (ccpo_scip_hook() ? 1 : 0) << "\n";
    const std::string dir      = argv[1];
    const double      time_lim = std::atof(argv[2]);
    std::string mode_str = (argc > 3) ? argv[3] : "ref";
    // Interrupteur : REF + détection de faisabilité résiduelle EXACTE (= mode "feas", cf. FEAS_ONLY dans gecode_bab_ccpo.hpp) pour comparer
    // à armes égales avec les modes à borne lagrangienne (qui incluent cette détection). Désactivé par défaut (REF = CP pur).
    if (mode_str == "ref")
        if (const char* e = std::getenv("CCPO_REF_FEAS"))
            if (std::atoi(e) == 1) {
                mode_str = "feas";
                std::cerr << "# CCPO_REF_FEAS=1 : REF avec détection exacte de faisabilité résiduelle (mode feas)\n";
            }
    const int T_root = (argc > 4) ? std::atoi(argv[4]) : 100;
    const int T_node = (argc > 5) ? std::atoi(argv[5]) : 20;

    const int K_bucket   = (argc > 6) ? std::atoi(argv[6]) : 8;
    const int omp_threads = (argc > 7) ? std::atoi(argv[7]) : 32;
    const bool gnn_zero = (mode_str == "gnn_zero");
    const bool gnn_real = (mode_str == "gnn");
    const std::string scripted = (argc > 8) ? argv[8] : "";
    const int torch_threads = (argc > 9) ? std::atoi(argv[9]) : 8;
    const std::string device = (argc > 10) ? argv[10] : "cpu";

    BoundMode mode = BoundMode::REF;
    if (gnn_zero || gnn_real) mode = BoundMode::GNN_BATCHED;
    else if (mode_str == "ref") mode = BoundMode::REF;
    else if (mode_str == "sld") mode = BoundMode::SLD;
    else if (mode_str == "feas") mode = BoundMode::FEAS_ONLY;
    else {
        std::cerr << "mode inconnu \"" << mode_str << "\" (attendu \"ref\", \"sld\" ou \"feas\")\n";
        return 1;
    }

    CCPOInstance inst;
    try {
        inst = load_ccpo_episode(dir);
    } catch (const std::exception& e) {
        std::cerr << "Erreur de chargement (" << dir << "): " << e.what() << "\n";
        return 1;
    }

    const int min_feas = saa_threshold(inst.eps_num, inst.eps_den, inst.N);

    if (gnn_real) {
#ifdef CCPO_WITH_TORCH
        CCPOGNNNativeBatched backend(scripted, torch_threads, device);
        CCPOGNNBatchedResult g = solve_gecode_bab_ccpo_gnn_batched(
            inst.n, inst.N, inst.k, inst.Q, inst.L, inst.U, inst.B, min_feas,
            inst.c_int, inst.sigma, time_lim, K_bucket, backend, omp_threads);
        std::cout << dir << "," << inst.n << "," << inst.N << "," << inst.k << "," << inst.Q << ","
                  << mode_str << "," << g.status << "," << g.nodes << "," << g.fails << ","
                  << g.fails_from_bound << "," << g.depth << "," << g.solve_time_s << ","
                  << g.objective << "," << g.greedy_value << "," << g.more_calls << ","
                  << g.ub_tightened << ",0,0," << g.t_bound_total_s
                  << ",flushes=" << g.n_bucket_flushes << ",fill=" << g.mean_bucket_fill
                  << ",max_pending=" << g.max_pending_live << ",leaf=" << g.nodes_bypassed_leaf
                  << ",infeas_inh=" << g.infeas_inherited << ",scip_calls=" << ccpo_zeta_stats().calls.load()
                  << ",scip_aborted=" << ccpo_zeta_stats().aborted.load() << ",scip_ok=" << ccpo_zeta_stats().scip_ok.load()
                  << ",scip_fail=" << ccpo_zeta_stats().scip_fail.load() << ",scip_noinc=" << ccpo_zeta_stats().no_incumbent.load()
                  << ",t_scip=" << ccpo_zeta_stats().scip_ns.load() / 1e9 << ",t_backend=" << g.t_backend_s << ",t_eval=" << g.t_eval_s
                  << ",t_graph=" << backend.t_graph_s << ",t_forward=" << backend.t_forward_s
                  << ",t_post=" << backend.t_post_s << ",members=" << backend.n_members
                  << ",first_level_bound=" << g.first_level_bound << ",first_level_n=" << g.first_level_children
                  << ",inc_impr=" << g.incumbent_improvements << ",K=" << K_bucket << ",omp=" << omp_threads
                  << ",torch_threads=" << torch_threads << ",device=" << device << ",tlim=" << time_lim
                  << ",scip_mode=" << ccpo_zeta_config().scip_mode << ",scip_budget=" << ccpo_zeta_config().scip_nodes
                  << ",scip_nf0=" << ccpo_zeta_config().scip_nf0
                  << ",model=" << scripted.substr(scripted.find_last_of('/') == std::string::npos ? 0 : scripted.find_last_of('/') + 1) << "\n";
        { auto& zs = ccpo_zeta_stats();   // compteurs de l'évaluation pilotée par l'élagage (stderr)
          std::cerr << "# lazy=" << ccpo_zeta_config().lazy << " evals=" << zs.lazy_evals.load() << " prune_cheap=" << zs.lazy_prune_cheap.load()
                    << " keep_cheap=" << zs.lazy_keep_cheap.load() << " refined=" << zs.lazy_refined.load()
                    << " exact_solves=" << zs.lazy_exact_solves.load() << " scen=" << zs.lazy_scen.load()
                    << " check_n=" << zs.lazy_check_n.load() << " check_bad=" << zs.lazy_check_bad.load() << " fallback=" << zs.lazy_fallback.load() << "\n"; }
        return 0;
#else
        std::cerr << "mode gnn : binaire compilé sans CCPO_WITH_TORCH\n";
        return 1;
#endif
    }
    if (gnn_zero) {
        CCPOZeroBackend backend;
        CCPOGNNBatchedResult g = solve_gecode_bab_ccpo_gnn_batched(
            inst.n, inst.N, inst.k, inst.Q, inst.L, inst.U, inst.B, min_feas,
            inst.c_int, inst.sigma, time_lim, K_bucket, backend, omp_threads);
        std::cout << dir << "," << inst.n << "," << inst.N << "," << inst.k << "," << inst.Q << ","
                  << mode_str << "," << g.status << "," << g.nodes << "," << g.fails << ","
                  << g.fails_from_bound << "," << g.depth << "," << g.solve_time_s << ","
                  << g.objective << "," << g.greedy_value << "," << g.more_calls << ","
                  << g.ub_tightened << ",0,0," << g.t_bound_total_s
                  << ",flushes=" << g.n_bucket_flushes << ",fill=" << g.mean_bucket_fill
                  << ",max_pending=" << g.max_pending_live << ",leaf=" << g.nodes_bypassed_leaf
                  << ",t_backend=" << g.t_backend_s << ",t_eval=" << g.t_eval_s
                  << ",first_level_bound=" << g.first_level_bound << ",first_level_n=" << g.first_level_children
                  << ",inc_impr=" << g.incumbent_improvements << ",K=" << K_bucket << ",omp=" << omp_threads << ",tlim=" << time_lim << "\n";
        { auto& zs = ccpo_zeta_stats();   // compteurs de l'évaluation pilotée par l'élagage (stderr)
          std::cerr << "# lazy=" << ccpo_zeta_config().lazy << " evals=" << zs.lazy_evals.load() << " prune_cheap=" << zs.lazy_prune_cheap.load()
                    << " keep_cheap=" << zs.lazy_keep_cheap.load() << " refined=" << zs.lazy_refined.load()
                    << " exact_solves=" << zs.lazy_exact_solves.load() << " scen=" << zs.lazy_scen.load()
                    << " check_n=" << zs.lazy_check_n.load() << " check_bad=" << zs.lazy_check_bad.load() << " fallback=" << zs.lazy_fallback.load() << "\n"; }
        return 0;
    }

    CCPOResult r = solve_gecode_bab_ccpo(
        inst.n, inst.N, inst.k, inst.Q, inst.L, inst.U, inst.B, min_feas,
        inst.c_int, inst.sigma, time_lim, mode, T_root, T_node, omp_threads);

    std::cout << dir << ","
              << inst.n << "," << inst.N << "," << inst.k << "," << inst.Q << ","
              << mode_str << ","
              << r.status << "," << r.nodes << "," << r.fails << "," << r.fails_from_bound << ","
              << r.depth << "," << r.solve_time_s << "," << r.objective << ","
              << r.greedy_value << "," << r.more_calls << "," << r.ub_tightened << ","
              << r.root_bound_time_s << "," << r.root_bound_value << ","
              << r.t_bound_total_s << ",inc_impr=" << r.incumbent_improvements << ",tlim=" << time_lim << "\n";

    return 0;
}
