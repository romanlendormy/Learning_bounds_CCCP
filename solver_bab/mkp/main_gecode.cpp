/*
 * main_gecode.cpp — CLI pour le B&B Gecode + borne V1LD.
 *
 * Usage :
 *   bab_mkp_gecode  <dataset.txt>  [epsilon]  [T_node]  [time_limit_s]  [replication]
 *
 * Exemple :
 *   bab_mkp_gecode  ../data/knapsack/datasets-song/ccmk-song-mknap1_4-S20.txt
 *   bab_mkp_gecode  ../data/knapsack/datasets-song/ccmk-song-mknap1_4-S20.txt 0.1 30 120 2
 */

#include "gecode_bab.hpp"
#include "gecode_bab_gnn_batched.hpp"
#include "gecode_bab_v1cld_gnn_batched.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ── Loader (copie locale pour éviter la dépendance sur bab_mkp.cpp) ──────────

struct MKPInst {
    Vec1d c; Vec1d p; Vec3d A; Vec2d b;
    int n, m, N;
};

static MKPInst load_song_local(const std::string& fp, int rep = 0)
{
    std::ifstream f(fp);
    if (!f) { std::cerr << "Erreur : " << fp << "\n"; std::exit(1); }

    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string tok;
        std::vector<double> vals;
        while (std::getline(iss, tok, ',')) vals.push_back(std::stod(tok));
        if (static_cast<int>(vals[4]) != rep) continue;
        rows.push_back(std::move(vals));
    }
    if (rows.empty()) { std::cerr << "Aucun scénario pour rep=" << rep << "\n"; std::exit(1); }

    int m = static_cast<int>(rows[0][2]);
    int n = static_cast<int>(rows[0][3]);
    int N = static_cast<int>(rows.size());

    Vec3d A(N, Vec2d(m, Vec1d(n, 0.0)));
    Vec2d b(N, Vec1d(m, 0.0));
    Vec1d c(n, 0.0);

    for (int i = 0; i < N; ++i) {
        const auto& v = rows[i];
        for (int j = 0; j < m; ++j)
            for (int k = 0; k < n; ++k)
                A[i][j][k] = v[5 + j * n + k];
        for (int j = 0; j < m; ++j)
            b[i][j] = v[5 + m * n + j];
        if (i == 0)
            for (int k = 0; k < n; ++k)
                c[k] = v[5 + m * n + m + k];
    }
    Vec1d p(N, 1.0 / N);
    return {c, p, A, b, n, m, N};
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage : bab_mkp_gecode <dataset.txt> "
                     "[epsilon=0.1] [T_node=30] [time_limit=120]\n";
        return 1;
    }

    std::string dataset     = argv[1];
    double      epsilon     = (argc > 2) ? std::stod(argv[2]) : 0.10;
    int         T_node      = (argc > 3) ? std::stoi(argv[3]) : 30;
    double      time_limit  = (argc > 4) ? std::stod(argv[4]) : 120.0;
    int         replication = (argc > 5) ? std::stoi(argv[5]) : 0;
    int         T_root      = (argc > 6) ? std::stoi(argv[6])  : 200;
    // mode : "v1ld" (défaut), "v1cld", "gnn" (borne one-shot pré-entraînée à
    // CHAQUE nœud), "gnn_root" (même borne one-shot, mais UNE SEULE FOIS à la
    // racine — REF ensuite), "gnn_batched" (borne GNN one-shot sd_edit/native,
    // mais calculée par LOTS de K nœuds en un seul forward — cf.
    // gecode_bab_gnn_batched.hpp, validé sur le sweep de robustesse job
    // 16406490 : mêmes optimaux que REF/gnn sur tout n∈{10,20}×N∈{20,50,100,500})
    // "gnn_batched_v1cld" (équivalent V1CLD de gnn_batched — cf.
    // gecode_bab_v1cld_gnn_batched.hpp, backend natif uniquement) ou "ref"
    // (squelette V1LD nu, sans borne — référence pour isoler l'effet du
    // branchement/incumbent/moteur DFS de l'effet de la borne elle-même).
    std::string mode_str    = (argc > 7) ? argv[7] : "v1ld";
    BoundMode   bound_mode  = (mode_str == "v1cld")          ? BoundMode::V1CLD
                            : (mode_str == "gnn")            ? BoundMode::GNN
                            : (mode_str == "gnn_root")       ? BoundMode::GNN_ROOT
                            : (mode_str == "gnn_root_v1cld") ? BoundMode::GNN_ROOT_V1CLD
                            : (mode_str == "ref")            ? BoundMode::REF
                                                             : BoundMode::V1LD;
    // patience : "patience" active le mode convergence, sinon itérations fixes
    // (ignoré en mode gnn : pas de sous-gradient à faire converger)
    bool        patience    = (argc > 8) && std::string(argv[8]) == "patience";
    double      patience_tol = (argc > 9) ? std::stod(argv[9]) : 0.1;
    // Mode gnn uniquement : checkpoint requis, device/python_exe optionnels.
    std::string gnn_checkpoint = (argc > 10) ? argv[10] : "";
    std::string gnn_device     = (argc > 11) ? argv[11] : "cpu";
    std::string gnn_python_exe = (argc > 12) ? argv[12] : "python3";
    // gnn_backend : "ipc" (défaut, process Python persistant) ou "native"
    // (build_graph+forward+lagrangian_v1ld réimplémentés en C++/LibTorch,
    // pas d'IPC par nœud — cf. gnn_v1ld_native.hpp). Ignoré hors mode gnn.
    std::string gnn_backend    = (argc > 13) ? argv[13] : "ipc";
    // gnn_variant : "homogene" (défaut, arêtes scénario = clique) ou "sd_edit"
    // (arêtes scénario = étoile, cf. gnn_sd_edit/train_v1LD_sd_edit.py et
    // gnn_sd_edit_native.hpp) — les deux backends (ipc, native) supportent
    // les deux variantes. Ignoré hors mode gnn.
    std::string gnn_variant    = (argc > 14) ? argv[14] : "homogene";
    // gnn_min_depth_frac : mode gnn uniquement (ignoré ailleurs, y compris
    // gnn_root) — saute l'appel GNN tant que depth/n < ce seuil (cf. doc sur
    // MKPSharedData::gnn_min_depth_frac, gecode_bab.hpp). Défaut 0.0 =
    // comportement historique (borne calculée à chaque nœud dès la racine).
    double      gnn_min_depth_frac = (argc > 15) ? std::stod(argv[15]) : 0.0;
    // gnn_min_gap_frac : mode gnn uniquement — saute l'appel GNN si le proxy
    // local gap_rel = (obj.max()-obj.min())/obj.max() dépasse ce seuil (cf.
    // doc sur MKPSharedData::gnn_min_gap_frac, gecode_bab.hpp). Défaut 0.0 =
    // jamais sauté (comportement historique).
    double      gnn_min_gap_frac   = (argc > 16) ? std::stod(argv[16]) : 0.0;
    // gnn_max_depth_frac : mode gnn uniquement — symétrique de
    // gnn_min_depth_frac, saute l'appel GNN dès que depth/n dépasse ce seuil
    // (cf. doc sur MKPSharedData::gnn_max_depth_frac, gecode_bab.hpp). Défaut
    // 1.0 = jamais sauté (comportement historique).
    double      gnn_max_depth_frac = (argc > 17) ? std::stod(argv[17]) : 1.0;
    // K : mode gnn_batched uniquement (ignoré ailleurs) — taille de bucket
    // pour le forward GNN groupé. Argument 18 (PAS 17 : gnn_max_depth_frac
    // occupe déjà cette position pour le mode gnn/gnn_root). Défaut 64 :
    // valeur intermédiaire choisie entre les deux K effectivement mesurés sur
    // le sweep de robustesse (job 16406490, n∈{10,20}×N∈{20,50,100,500}) —
    // K=32 (meilleur à N=500 : 131s vs 137s pour K=128) et K=128 (légèrement
    // meilleur à N≤100). K=64 lui-même N'A PAS été mesuré — extrapolation,
    // pas une valeur validée (cf. mémoire "gnn_batched_robustness_sweep_result").
    int         gnn_batched_K      = (argc > 18) ? std::stoi(argv[18]) : 64;
    // gnn_batched_max_depth_frac : mode gnn_batched uniquement — symétrique de
    // gnn_max_depth_frac (mode gnn), mais appliqué bucket par bucket (chaque
    // pending[ng_depth] est déjà un bucket de profondeur homogène) : au-delà
    // de ce seuil, l'enfant est traité immédiatement (obj.max() post-CP, pas
    // de borne GNN), cf. doc gecode_bab_gnn_batched.hpp. Argument 19 (pas 18,
    // déjà pris par K). Défaut 1.0 = jamais de bypass (comportement historique).
    double      gnn_batched_max_depth_frac = (argc > 19) ? std::stod(argv[19]) : 1.0;
    // zeta_method : modes v1ld ET gnn_batched (ignoré ailleurs — gnn/gnn_root
    // non-batché pas encore câblés) — méthode utilisée pour résoudre ζ_i.
    // "bnb" (défaut, historique/prod) ou "surrogate" (borne de branchement
    // agrégée au lieu du min de m relaxations séparées) :
    //   - mode v1ld : MKPProblemBnB vs MKPProblemBnBSurrogate (gecode_bab.hpp
    //     / MKPSharedData::use_surrogate_zeta, solver_generic/mkp/).
    //   - mode gnn_batched : ζ_i résolu APRÈS la prédiction λ du GNN (calcul
    //     de L1(λ)) via solve_zeta_batch vs solve_zeta_batch_surrogate
    //     (gnn/bnb_solver.cpp, cf. GNNSdEditNativeBatched::use_surrogate_zeta).
    // Même valeur ζ_i exacte dans les deux méthodes (seule la preuve
    // d'optimalité interne diffère) — validé ×2.25 plus rapide dans le B&B
    // V1LD séquentiel (n=20/N=100), cf. mémoire
    // project_compare_v1ld_zeta_bnb_surrogate. Argument 21 (20 déjà pris par
    // gnn_batched_max_depth_frac, mode gnn_batched).
    std::string zeta_method_str = (argc > 20) ? argv[20] : "bnb";
    bool        use_surrogate_zeta = (zeta_method_str == "surrogate");
    // gnn_node_stride : mode gnn_batched uniquement (ignoré ailleurs) — 1
    // candidat GNN sur gnn_node_stride obtient réellement sa borne, les
    // autres sont bypassés comme gnn_max_depth_frac (cf. doc sur ce
    // paramètre, gecode_bab_gnn_batched.hpp). Défaut 1 = jamais de bypass
    // (comportement historique). Argument 22.
    int         gnn_node_stride = (argc > 21) ? std::stoi(argv[21]) : 1;
    // gnn_batched_backend : mode gnn_batched uniquement — "native" (défaut,
    // TorchScript, GNNSdEditNativeBatched), "native_aoti" (AOTInductor B
    // dynamique, ABANDONNÉ — bug compilateur CUDA/Inductor, cf. mémoire
    // "aoti-batched-gru-decomposed-abandoned") ou "native_aoti_static"
    // (AOTInductor B figé = K, contourne ce bug, cf.
    // gnn_sd_edit_native_aoti_batched_static.hpp). Argument 23.
    std::string gnn_batched_backend = (argc > 22) ? argv[22] : "native";

    if ((mode_str == "gnn_batched" || mode_str == "gnn_batched_v1cld") && gnn_checkpoint.empty()) {
        std::cerr << "Erreur : mode=" << mode_str << " requiert un chemin de checkpoint "
                     "(argument 10) — usage :\n"
                     "  bab_mkp_gecode <dataset> [eps] [T_node] [time_lim] [rep] [T_root] "
                     "gnn_batched|gnn_batched_v1cld <ckpt.pt> [device=cpu] [-] [-] [-] [-] [-] [-] [K=64] "
                     "[gnn_max_depth_frac=1.0]\n"
                     "  (arguments 12-17 ignorés en mode gnn_batched/gnn_batched_v1cld : pas d'IPC, pas de "
                     "variante autre que sd_edit, pas de gating profondeur/gap min)\n";
        return 1;
    }

    if ((bound_mode == BoundMode::GNN || bound_mode == BoundMode::GNN_ROOT ||
         bound_mode == BoundMode::GNN_ROOT_V1CLD) && gnn_checkpoint.empty()) {
        std::cerr << "Erreur : mode=gnn/gnn_root/gnn_root_v1cld requiert un chemin de checkpoint "
                     "(argument 10) — usage :\n"
                     "  bab_mkp_gecode <dataset> [eps] [T_node] [time_lim] "
                     "[rep] [T_root] gnn|gnn_root|gnn_root_v1cld <ckpt.pt> [device=cpu] "
                     "[python_exe=python3] [backend=ipc|native] [variant=homogene|sd_edit] "
                     "[gnn_min_depth_frac=0.0] [gnn_min_gap_frac=0.0] [gnn_max_depth_frac=1.0]\n"
                     "  (mode gnn_root_v1cld : backend=native uniquement, variant/python_exe "
                     "ignorés — cf. GNNV1CLDSdEditNative, gnn_v1cld_sd_edit_native.hpp)\n";
        return 1;
    }

    MKPInst inst = load_song_local(dataset, replication);

    // ── mode=gnn_batched : moteur DFS + backend GNN entièrement séparés
    // (solve_gecode_bab_mkp_gnn_batched, gecode_bab_gnn_batched.hpp) — ne
    // partage ni BoundMode ni MKPSpaceV1LD/solve_gecode_bab_mkp avec les
    // autres modes, donc traité à part ici et retourne avant le chemin V1LD/
    // V1CLD/GNN/GNN_ROOT/REF ci-dessous (intact, non modifié par ce merge).
    if (mode_str == "gnn_batched") {
        std::cout << "═══════════════════════════════════════════════════\n";
        std::cout << "  CCMKP — Gecode BAB + borne GNN BATCHÉE (K=" << gnn_batched_K
                  << ", sd_edit, " << gnn_batched_backend << ")\n";
        std::cout << "═══════════════════════════════════════════════════\n";
        std::cout << "  Fichier : " << dataset << "\n";
        std::cout << "  n="   << inst.n << "  m=" << inst.m
                  << "  N="   << inst.N << "  ε=" << epsilon
                  << "  rep=" << replication << "\n";
        std::cout << "  time_limit=" << time_limit << "s  device=" << gnn_device
                  << "  K=" << gnn_batched_K
                  << (gnn_batched_max_depth_frac < 1.0 ? ("  gnn_max_depth_frac=" + std::to_string(gnn_batched_max_depth_frac)) : "")
                  << (use_surrogate_zeta ? "  zeta_method=surrogate" : "")
                  << (gnn_node_stride > 1 ? ("  gnn_node_stride=" + std::to_string(gnn_node_stride)) : "")
                  << "\n";
        std::cout << "───────────────────────────────────────────────────\n";

        GNNBatchedBabResult res = solve_gecode_bab_mkp_gnn_batched(
            inst.n, inst.m, inst.N, epsilon, inst.c, inst.p, inst.A, inst.b,
            time_limit, gnn_batched_K, gnn_checkpoint, gnn_device, gnn_batched_max_depth_frac,
            use_surrogate_zeta, gnn_node_stride, gnn_batched_backend);

        std::cout << "\n═══════════════════════════════════════════════════\n";
        std::cout << "  RÉSULTATS\n";
        std::cout << "═══════════════════════════════════════════════════\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  Statut                    : " << res.status    << "\n";
        std::cout << "  Valeur optimale           : " << res.objective << "\n";
        if (std::isnan(res.first_level_bound_value)) {
            std::cout << "  Borne GNN 1er niveau      : N/A (jamais flushée avant timeout)\n";
        } else {
            std::cout << "  Borne GNN 1er niveau      : " << res.first_level_bound_value
                       << "  (proxy \"gap racine\" : enfants directs de la racine, "
                       << res.first_level_bound_n_members << " membre(s) ; PAS la racine "
                       << "elle-même, cf. doc GNNBatchedBabResult::first_level_bound_value)"
                       << "  (gap vs optimal : " << std::setprecision(2)
                       << (res.objective > 0 ? 100.0 * (res.first_level_bound_value - res.objective) / res.objective : 0.0)
                       << "%)\n" << std::setprecision(4);
        }
        std::cout << "  Incumbent initial (greedy): " << res.greedy_value << "\n\n";

        std::cout << std::setprecision(0);
        std::cout << "  Nœuds                     : " << res.nodes << "\n";
        std::cout << "  Échecs totaux             : " << res.fails << "\n";
        std::cout << "  Améliorations incumbent   : " << res.incumbent_improvements << "\n\n";

        std::cout << "  Flushes de bucket         : " << res.n_bucket_flushes << "\n";
        std::cout << std::setprecision(2);
        std::cout << "  Remplissage moyen au flush: " << res.mean_bucket_fill_at_flush
                  << " / K=" << gnn_batched_K << "\n";
        std::cout << std::setprecision(0);
        std::cout << "  Pic de nœuds en attente   : " << res.max_pending_live << "\n";
        if (gnn_batched_max_depth_frac < 1.0)
            std::cout << "  Bypass (depth>seuil)      : " << res.nodes_bypassed_depth_cutoff << "\n";
        if (gnn_node_stride > 1)
            std::cout << "  Bypass (stride)           : " << res.nodes_bypassed_stride << "\n";
        std::cout << "\n";

        std::cout << std::setprecision(3);
        std::cout << "  Temps total               : " << res.solve_time_s << " s\n";
        std::cout << "  Temps borne (GNN batché)  : " << res.t_bound_total_s << " s"
                  << "  (" << std::setprecision(1) << std::fixed
                  << (res.solve_time_s > 0 ? 100.0 * res.t_bound_total_s / res.solve_time_s : 0.0)
                  << "% du total)\n" << std::setprecision(3);
        std::cout << "    dont construction graphe: " << res.t_gnn_graph_build_s << " s\n";
        std::cout << "    dont forward GNN        : " << res.t_gnn_forward_s << " s\n";
        std::cout << "    dont calcul borne (ζ)   : " << res.t_gnn_bound_compute_s << " s\n";
        std::cout << "  Temps BAB pur             : " << res.t_bab_pure_s << " s\n";
        if (res.nodes > 0)
            std::cout << "  Temps moyen par nœud      : "
                      << (res.solve_time_s / res.nodes * 1000.0) << " ms\n";
        std::cout << "═══════════════════════════════════════════════════\n";

        return 0;
    }

    // ── mode=gnn_batched_v1cld : équivalent V1CLD de gnn_batched ci-dessus —
    // même moteur DFS+bucketing (solve_gecode_bab_mkp_v1cld_gnn_batched,
    // gecode_bab_v1cld_gnn_batched.hpp), même CLI, seul le backend GNN et la
    // borne changent (ζ_{i,j} par (scénario,contrainte) au lieu de ζ_i par
    // scénario). Seul gnn_batched_backend="native" supporté (pas d'AOTInductor
    // V1CLD pour l'instant, cf. doc de header).
    if (mode_str == "gnn_batched_v1cld") {
        if (gnn_batched_backend != "native") {
            std::cerr << "Erreur : mode=gnn_batched_v1cld ne supporte que "
                         "gnn_batched_backend=native pour l'instant (reçu \""
                      << gnn_batched_backend << "\")\n";
            return 1;
        }

        std::cout << "═══════════════════════════════════════════════════\n";
        std::cout << "  CCMKP — Gecode BAB + borne V1CLD GNN BATCHÉE (K=" << gnn_batched_K
                  << ", sd_edit, " << gnn_batched_backend << ")\n";
        std::cout << "═══════════════════════════════════════════════════\n";
        std::cout << "  Fichier : " << dataset << "\n";
        std::cout << "  n="   << inst.n << "  m=" << inst.m
                  << "  N="   << inst.N << "  ε=" << epsilon
                  << "  rep=" << replication << "\n";
        std::cout << "  time_limit=" << time_limit << "s  device=" << gnn_device
                  << "  K=" << gnn_batched_K
                  << (gnn_batched_max_depth_frac < 1.0 ? ("  gnn_max_depth_frac=" + std::to_string(gnn_batched_max_depth_frac)) : "")
                  << (use_surrogate_zeta ? "  zeta_method=surrogate" : "")
                  << (gnn_node_stride > 1 ? ("  gnn_node_stride=" + std::to_string(gnn_node_stride)) : "")
                  << "\n";
        std::cout << "───────────────────────────────────────────────────\n";

        GNNBatchedBabResult res = solve_gecode_bab_mkp_v1cld_gnn_batched(
            inst.n, inst.m, inst.N, epsilon, inst.c, inst.p, inst.A, inst.b,
            time_limit, gnn_batched_K, gnn_checkpoint, gnn_device, gnn_batched_max_depth_frac,
            use_surrogate_zeta, gnn_node_stride, gnn_batched_backend);

        std::cout << "\n═══════════════════════════════════════════════════\n";
        std::cout << "  RÉSULTATS\n";
        std::cout << "═══════════════════════════════════════════════════\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  Statut                    : " << res.status    << "\n";
        std::cout << "  Valeur optimale           : " << res.objective << "\n";
        if (std::isnan(res.first_level_bound_value)) {
            std::cout << "  Borne GNN 1er niveau      : N/A (jamais flushée avant timeout)\n";
        } else {
            std::cout << "  Borne GNN 1er niveau      : " << res.first_level_bound_value
                       << "  (proxy \"gap racine\" : enfants directs de la racine, "
                       << res.first_level_bound_n_members << " membre(s) ; PAS la racine "
                       << "elle-même, cf. doc GNNBatchedBabResult::first_level_bound_value)"
                       << "  (gap vs optimal : " << std::setprecision(2)
                       << (res.objective > 0 ? 100.0 * (res.first_level_bound_value - res.objective) / res.objective : 0.0)
                       << "%)\n" << std::setprecision(4);
        }
        std::cout << "  Incumbent initial (greedy): " << res.greedy_value << "\n\n";

        std::cout << std::setprecision(0);
        std::cout << "  Nœuds                     : " << res.nodes << "\n";
        std::cout << "  Échecs totaux             : " << res.fails << "\n";
        std::cout << "  Améliorations incumbent   : " << res.incumbent_improvements << "\n\n";

        std::cout << "  Flushes de bucket         : " << res.n_bucket_flushes << "\n";
        std::cout << std::setprecision(2);
        std::cout << "  Remplissage moyen au flush: " << res.mean_bucket_fill_at_flush
                  << " / K=" << gnn_batched_K << "\n";
        std::cout << std::setprecision(0);
        std::cout << "  Pic de nœuds en attente   : " << res.max_pending_live << "\n";
        if (gnn_batched_max_depth_frac < 1.0)
            std::cout << "  Bypass (depth>seuil)      : " << res.nodes_bypassed_depth_cutoff << "\n";
        if (gnn_node_stride > 1)
            std::cout << "  Bypass (stride)           : " << res.nodes_bypassed_stride << "\n";
        std::cout << "\n";

        std::cout << std::setprecision(3);
        std::cout << "  Temps total               : " << res.solve_time_s << " s\n";
        std::cout << "  Temps borne (GNN batché)  : " << res.t_bound_total_s << " s"
                  << "  (" << std::setprecision(1) << std::fixed
                  << (res.solve_time_s > 0 ? 100.0 * res.t_bound_total_s / res.solve_time_s : 0.0)
                  << "% du total)\n" << std::setprecision(3);
        std::cout << "    dont construction graphe: " << res.t_gnn_graph_build_s << " s\n";
        std::cout << "    dont forward GNN        : " << res.t_gnn_forward_s << " s\n";
        std::cout << "    dont calcul borne (ζ)   : " << res.t_gnn_bound_compute_s << " s\n";
        std::cout << "  Temps BAB pur             : " << res.t_bab_pure_s << " s\n";
        if (res.nodes > 0)
            std::cout << "  Temps moyen par nœud      : "
                      << (res.solve_time_s / res.nodes * 1000.0) << " ms\n";
        std::cout << "═══════════════════════════════════════════════════\n";

        return 0;
    }

    std::cout << "═══════════════════════════════════════════════════\n";
    std::cout << "  CCMKP — Gecode BAB + borne "
              << (bound_mode == BoundMode::V1LD          ? "V1LD (ζ exact B&B)"
                : bound_mode == BoundMode::V1CLD         ? "V1CLD (ζ DP 1D exact)"
                : bound_mode == BoundMode::REF           ? "REF (squelette nu, sans borne)"
                : bound_mode == BoundMode::GNN_ROOT      ? "GNN_ROOT (one-shot V1LD, racine seulement, REF ensuite)"
                : bound_mode == BoundMode::GNN_ROOT_V1CLD ? "GNN_ROOT_V1CLD (one-shot V1CLD, racine seulement, REF ensuite)"
                                                          : "GNN (one-shot, pas de sous-gradient)")
              << "\n";
    std::cout << "═══════════════════════════════════════════════════\n";
    std::cout << "  Fichier : " << dataset << "\n";
    std::cout << "  n="   << inst.n << "  m=" << inst.m
              << "  N="   << inst.N << "  ε=" << epsilon
              << "  rep=" << replication << "\n";
    std::cout << "  T_root=" << T_root << "  T_node=" << T_node
              << "  time_limit=" << time_limit << "s"
              << "  mode=" << mode_str
              << (patience ? ("  patience=oui(tol=" + std::to_string(patience_tol) + ")") : "  patience=non")
              << ((bound_mode == BoundMode::V1LD && use_surrogate_zeta) ? "  zeta_method=surrogate" : "")
              << ((bound_mode == BoundMode::GNN || bound_mode == BoundMode::GNN_ROOT ||
                   bound_mode == BoundMode::GNN_ROOT_V1CLD) ? ("  gnn_backend=" + gnn_backend + "  gnn_variant=" + gnn_variant) : "")
              << ((bound_mode == BoundMode::GNN && gnn_min_depth_frac > 0.0) ? ("  gnn_min_depth_frac=" + std::to_string(gnn_min_depth_frac)) : "")
              << ((bound_mode == BoundMode::GNN && gnn_min_gap_frac > 0.0) ? ("  gnn_min_gap_frac=" + std::to_string(gnn_min_gap_frac)) : "")
              << ((bound_mode == BoundMode::GNN && gnn_max_depth_frac < 1.0) ? ("  gnn_max_depth_frac=" + std::to_string(gnn_max_depth_frac)) : "")
              << "\n";
    std::cout << "───────────────────────────────────────────────────\n";

    GecodeV1LDResult res = solve_gecode_bab_mkp(
        inst.n, inst.m, inst.N, epsilon,
        inst.c, inst.p, inst.A, inst.b,
        time_limit, T_root, T_node, bound_mode,
        patience, patience_tol,
        gnn_checkpoint, gnn_python_exe, gnn_device,
        "gnn/gnn_inference_server.py", gnn_backend, gnn_variant,
        gnn_min_depth_frac, gnn_min_gap_frac, gnn_max_depth_frac,
        use_surrogate_zeta);

    std::cout << "\n═══════════════════════════════════════════════════\n";
    std::cout << "  RÉSULTATS\n";
    std::cout << "═══════════════════════════════════════════════════\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Statut                    : " << res.status        << "\n";
    std::cout << "  Valeur optimale           : " << res.objective     << "\n";
    std::cout << "  UB finale                 : " << res.best_open_ub   << "\n";
    if (bound_mode == BoundMode::REF) {
        std::cout << "  Borne UB à la racine      : N/A (mode ref, pas de borne)\n";
    } else {
        // Avec gnn_min_depth_frac > 0, le premier appel qui calcule réellement
        // la borne n'est plus à la racine (depth 0) mais au premier nœud
        // atteignant le seuil — le libellé le précise pour éviter de lire
        // "à la racine" à tort.
        const bool gated_root = (bound_mode == BoundMode::GNN &&
                                  (gnn_min_depth_frac > 0.0 || gnn_min_gap_frac > 0.0));
        std::cout << "  Borne UB à la racine      : " << res.root_bound_value
                  << (gated_root ? "  (PAS la racine : 1er appel réel, seuil gnn_min_depth_frac/gnn_min_gap_frac atteint)" : "")
                  << "  (gap vs optimal : " << std::setprecision(2)
                  << (res.objective > 0 ? 100.0 * (res.root_bound_value - res.objective) / res.objective : 0.0)
                  << "%)\n" << std::setprecision(4);
    }
    std::cout << "  Incumbent initial (greedy): " << res.greedy_value  << "\n\n";

    // Nombre de nœuds dans l'arbre naïf (énumération complète sur x) : 2^n
    long long naive_nodes = 1LL << inst.n;  // 2^n feuilles

    std::cout << std::setprecision(0);
    std::cout << "  Nœuds Gecode              : " << res.nodes
              << " / " << naive_nodes << " naïf"
              << "  (" << std::fixed << std::setprecision(2)
              << static_cast<double>(res.nodes) / naive_nodes * 100.0 << "%)\n";

    std::cout << std::setprecision(0);
    std::cout << "  Profondeur max            : " << res.depth
              << " / " << (inst.n + inst.N) << " max théorique\n\n";

    std::cout << "  Échecs totaux             : " << res.fails      << "\n";
    std::cout << "    dont borne V1LD         : " << res.fails_from_bound
              << "  (" << std::fixed << std::setprecision(1)
              << (res.fails > 0 ? 100.0 * res.fails_from_bound / res.fails : 0.0)
              << "% des fails)\n";
    std::cout << "    dont propagation CP     : " << res.fails_from_cp
              << "  (" << std::fixed << std::setprecision(1)
              << (res.fails > 0 ? 100.0 * res.fails_from_cp / res.fails : 0.0)
              << "% des fails)\n\n";

    std::cout << "  Améliorations incumbent             : " << res.incumbent_improvements << "\n";
    std::cout << "  Appels more() (commits réussis)     : " << res.more_calls << "\n";
    std::cout << "  UB V1LD utile (réduit obj.max)      : " << res.ub_tightened
              << " / " << res.more_calls
              << "  (" << std::fixed << std::setprecision(1)
              << (res.more_calls > 0 ? 100.0 * res.ub_tightened / res.more_calls : 0.0)
              << "% des appels)\n\n";

    std::cout << std::setprecision(3);
    std::cout << "  Temps total               : " << res.solve_time_s     << " s\n";
    if (res.t_gnn_warmup_s > 0.0) {
        std::cout << "    dont pré-chauffage GNN (formes n_g) : " << std::setprecision(3)
                  << res.t_gnn_warmup_s << " s"
                  << "  (" << std::fixed << std::setprecision(1)
                  << (res.solve_time_s > 0 ? 100.0 * res.t_gnn_warmup_s / res.solve_time_s : 0.0)
                  << "% du total — hors chrono B&B/borne ci-dessous)\n";
    }
    std::cout << "  Temps borne root          : " << res.root_bound_time_s << " s"
              << "  (T_root=" << T_root << " iters)\n";
    std::cout << "  Temps borne total         : " << res.t_bound_total_s   << " s"
              << "  (" << std::fixed << std::setprecision(1)
              << (res.solve_time_s > 0 ? 100.0 * res.t_bound_total_s / res.solve_time_s : 0.0)
              << "% du total)\n";
    std::cout << "  Temps BAB pur             : " << std::setprecision(3) << res.t_bab_pure_s << " s"
              << "  (" << std::fixed << std::setprecision(1)
              << (res.solve_time_s > 0 ? 100.0 * res.t_bab_pure_s / res.solve_time_s : 0.0)
              << "% du total)\n";
    std::cout << "    dont clonage complet    : " << std::setprecision(4) << res.t_clone_total_s << " s"
              << "  (" << std::fixed << std::setprecision(1)
              << (res.t_bab_pure_s > 0 ? 100.0 * res.t_clone_total_s / res.t_bab_pure_s : 0.0)
              << "% du BAB pur, sur " << res.clone_count << " clonages)\n";
    std::cout << "      dont copie état (λ/b_res) : " << std::setprecision(4) << res.t_clone_state_s << " s"
              << "  (" << std::fixed << std::setprecision(1)
              << (res.t_clone_total_s > 0 ? 100.0 * res.t_clone_state_s / res.t_clone_total_s : 0.0)
              << "% du clonage total)\n"
              << "      dont clonage natif Gecode : " << std::setprecision(4)
              << (res.t_clone_total_s - res.t_clone_state_s) << " s\n"
              << "    dont dispatch brancher/CP hors clone : " << std::setprecision(4)
              << (res.t_bab_pure_s - res.t_clone_total_s) << " s\n";
    if (bound_mode == BoundMode::GNN || bound_mode == BoundMode::GNN_ROOT ||
        bound_mode == BoundMode::GNN_ROOT_V1CLD) {
        std::cout << std::setprecision(4);
        std::cout << "  --- décomposition borne GNN (cumulée sur " << res.more_calls << " appels) ---\n";
        std::cout << "  Construction de graphe    : " << res.t_gnn_graph_build_s   << " s\n";
        std::cout << "    dont features (tenseur nœuds)   : " << res.t_gnn_graph_features_s << " s"
                  << "  (" << std::fixed << std::setprecision(1)
                  << (res.t_gnn_graph_build_s > 0 ? 100.0 * res.t_gnn_graph_features_s / res.t_gnn_graph_build_s : 0.0)
                  << "%)\n" << std::setprecision(4);
        std::cout << "    dont arêtes (index+poids)       : " << res.t_gnn_graph_edges_s << " s"
                  << "  (" << std::fixed << std::setprecision(1)
                  << (res.t_gnn_graph_build_s > 0 ? 100.0 * res.t_gnn_graph_edges_s / res.t_gnn_graph_build_s : 0.0)
                  << "%)\n" << std::setprecision(4);
        std::cout << "  Inférence pure (forward)  : " << res.t_gnn_forward_s       << " s\n";
        std::cout << "  Calcul borne L1(λ) (ζ)    : " << res.t_gnn_bound_compute_s << " s\n";
        std::cout << "  Overhead IPC (pipe)       : " << res.t_gnn_ipc_overhead_s  << " s\n";
    }
    if (res.nodes > 0)
        std::cout << "  Temps moyen par nœud     : "
                  << std::setprecision(3)
                  << (res.solve_time_s - res.t_gnn_warmup_s) / res.nodes * 1000.0
                  << " ms  (hors pré-chauffage GNN)\n";
    std::cout << "═══════════════════════════════════════════════════\n";

    // ── Statistiques par profondeur ───────────────────────────────────────────
    // Objectif : repérer la depth au-delà de laquelle calculer la borne ne
    // provoque plus aucun fail (fails_from_bound == 0), pour évaluer si on
    // peut arrêter de l'appeler passé ce seuil. Vide en mode REF (aucune borne
    // calculée). depth = nombre d'items x déjà fixés (0 = racine, n = feuille).
    if (bound_mode != BoundMode::REF && !res.nodes_by_depth.empty()) {
        std::cout << "\n  --- utilité de la borne par profondeur ---\n";
        std::cout << "  depth   nodes  ub_tightened  fails_from_bound   %tightened   %fails   ms/nœud   %temps_borne\n";
        int last_fail_depth = -1;
        for (size_t d = 0; d < res.nodes_by_depth.size(); ++d) {
            if (res.nodes_by_depth[d] > 0 && res.fails_from_bound_by_depth[d] > 0)
                last_fail_depth = static_cast<int>(d);
        }
        double t_bound_by_depth_total = 0.0;
        for (double t : res.t_bound_by_depth) t_bound_by_depth_total += t;
        for (size_t d = 0; d < res.nodes_by_depth.size(); ++d) {
            int64_t nd = res.nodes_by_depth[d];
            if (nd == 0) continue;
            int64_t ut = res.ub_tightened_by_depth[d];
            int64_t fb = res.fails_from_bound_by_depth[d];
            double  tb = res.t_bound_by_depth[d];
            std::cout << "  " << std::setw(4) << d
                       << std::setw(9) << nd
                       << std::setw(15) << ut
                       << std::setw(19) << fb
                       << std::setw(12) << std::fixed << std::setprecision(1)
                       << (100.0 * ut / nd) << "%"
                       << std::setw(9) << std::fixed << std::setprecision(1)
                       << (100.0 * fb / nd) << "%"
                       << std::setw(10) << std::fixed << std::setprecision(3)
                       << (tb / nd * 1000.0)
                       << std::setw(13) << std::fixed << std::setprecision(1)
                       << (t_bound_by_depth_total > 0 ? 100.0 * tb / t_bound_by_depth_total : 0.0) << "%\n";
        }
        if (last_fail_depth >= 0) {
            // Coût cumulé de la borne pour toutes les depths STRICTEMENT
            // au-delà de last_fail_depth — ce qui serait économisé en coupant
            // le calcul de la borne à ce seuil (fails_from_bound == 0 partout
            // au-delà, donc aucun élagage immédiat perdu par cet arrêt — les
            // resserrements non-fail (ub_tightened) au-delà du seuil restent,
            // eux, une perte potentielle non quantifiée ici, cf. commentaire).
            double t_beyond = 0.0;
            for (size_t d = last_fail_depth + 1; d < res.t_bound_by_depth.size(); ++d)
                t_beyond += res.t_bound_by_depth[d];
            std::cout << "  → dernière depth avec fails_from_bound > 0 : "
                      << last_fail_depth << " / " << inst.n
                      << " (borne inutile pour élaguer au-delà)\n"
                      << "  → coût de la borne au-delà de cette depth : "
                      << std::fixed << std::setprecision(3) << t_beyond << " s"
                      << " (" << std::setprecision(1)
                      << (t_bound_by_depth_total > 0 ? 100.0 * t_beyond / t_bound_by_depth_total : 0.0)
                      << "% du temps de borne total — potentiellement économisable ;"
                      << " NB : ub_tightened sans fail immédiat au-delà de ce seuil"
                      << " peut quand même contribuer à élaguer plus profond, non capturé ici)\n";
        } else {
            std::cout << "  → aucun fail directement dû à la borne à aucune profondeur\n";
        }
        std::cout << "═══════════════════════════════════════════════════\n";
    }

    return 0;
}
