/*
 * gecode_bab.cpp — Implémentation de greedy_mkp et solve_gecode_bab_mkp.
 */

#include "gecode_bab.hpp"
#include "gnn_sld_client.hpp"
#include "gnn_sld_native.hpp"
#include "gnn_sd_edit_native.hpp"
#include "gnn_sd_edit_native_aoti.hpp"
#include "gnn_dld_sd_edit_native.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <tuple>

// ═════════════════════════════════════════════════════════════════════════════
// greedy_mkp
// ═════════════════════════════════════════════════════════════════════════════

std::pair<double, std::vector<int8_t>> greedy_mkp(
    const Vec1d& c, const Vec1d& p,
    const Vec3d& A, const Vec2d& b,
    double epsilon)
{
    const int n = static_cast<int>(c.size());
    const int N = static_cast<int>(p.size());
    const int m = static_cast<int>(b[0].size());

    // Tri des items par ratio c_k / poids_surrogate décroissant.
    // poids_surrogate = Σ_i Σ_j p_i * A_i[j][k] / b_i[j]  (contribution normalisée)
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int bk) {
        double wa = 0.0, wb = 0.0;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < m; ++j)
                if (b[i][j] > 1e-9) {
                    wa += p[i] * A[i][j][a]  / b[i][j];
                    wb += p[i] * A[i][j][bk] / b[i][j];
                }
        double ra = (wa > 1e-12) ? c[a]  / wa  : c[a]  * 1e12;
        double rb = (wb > 1e-12) ? c[bk] / wb  : c[bk] * 1e12;
        return ra > rb;
    });

    // Vérification de faisabilité : P(∃j : Σ A_i[j][k] x_k > b_i[j]) ≤ ε
    auto feasible = [&](const std::vector<int8_t>& x) {
        double pv = 0.0;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < m; ++j) {
                double load = 0.0;
                for (int k = 0; k < n; ++k)
                    if (x[k] == 1) load += A[i][j][k];
                if (load > b[i][j] + 1e-9) { pv += p[i]; break; }
            }
            if (pv > epsilon + 1e-9) return false;
        }
        return true;
    };

    std::vector<int8_t> x(n, 0);
    double val = 0.0;
    for (int k : order) {
        if (c[k] <= 0.0) continue;
        x[k] = 1;
        if (!feasible(x)) x[k] = 0;
        else              val  += c[k];
    }
    return {val, x};
}

GecodeSLDResult solve_gecode_bab_mkp(
    int n, int m, int N, double epsilon,
    const Vec1d& c, const Vec1d& p,
    const Vec3d& A, const Vec2d& b,
    double    time_lim,
    int       T_root,
    int       T_node,
    BoundMode bound_mode,
    bool      patience_mode,
    double    patience_tol,
    std::string gnn_checkpoint_path,
    std::string gnn_python_exe,
    std::string gnn_device,
    std::string gnn_server_script,
    std::string gnn_backend,
    std::string gnn_variant,
    double gnn_min_depth_frac,
    double gnn_min_gap_frac,
    double gnn_max_depth_frac,
    bool   use_surrogate_zeta)
{
    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();

    // ── 1. Solution gloutonne — incumbent initial ─────────────────────────────
    // On calcule une solution faisable avant de lancer Gecode.
    // Sa valeur est postée comme contrainte obj >= greedy_int dans le constructeur
    // de MKPSpaceSLD, ce qui élimine d'emblée les branches sous-optimales.
    auto [greedy_val, greedy_x] = greedy_mkp(c, p, A, b, epsilon);
    int greedy_int = static_cast<int>(std::floor(greedy_val * MKPSharedData::SCALE));

    // ── 2. Données partagées ──────────────────────────────────────────────────
    MKPSharedData shared(n, m, N, epsilon, c, p, A, b, T_node);
    shared.T_root         = T_root;
    shared.bound_mode     = bound_mode;
    shared.patience_mode  = patience_mode;
    shared.patience_tol   = patience_tol;
    shared.initial_lb_int = greedy_int;   // fournit l'incumbent à l'espace racine
    shared.gnn_min_depth_frac = gnn_min_depth_frac;
    shared.gnn_min_gap_frac   = gnn_min_gap_frac;
    shared.gnn_max_depth_frac = gnn_max_depth_frac;
    shared.use_surrogate_zeta = use_surrogate_zeta;
    // GNN_COMPARE_BOUND : opt-in (variable d'env, même pattern que
    // ROOT_SUBGRAD_TRACE) — cf. doc MKPSharedData::gnn_compare_bound.
    // Absent/non-parsable → 1e18 (désactivé, comportement historique).
    if (const char* env_bound = std::getenv("GNN_COMPARE_BOUND")) {
        try {
            shared.gnn_compare_bound = std::stod(env_bound);
        } catch (const std::exception&) {
            shared.gnn_compare_bound = 1e18;
        }
    }
    // Découple more() de BoundBrancher::commit() (cf. revue niveau 1 point A) :
    // la boucle DFS ci-dessous appelle explicitement more() APRÈS un premier
    // status() de propagation CP pure, pour ne payer le coût de la borne
    // (SLD/DLD/GNN) que sur les nœuds que le CP seul n'a pas déjà tués.
    shared.defer_bound_to_driver = true;

    // ── 2bis. Mode GNN : backend IPC (process Python persistant) ou natif
    // (LibTorch en-process) — cf. gnn_backend, doc sur solve_gecode_bab_mkp.
    // Possédé localement (unique_ptr) — shared.gnn_client n'est qu'un pointeur
    // non-propriétaire, valide tant que ce unique_ptr vit (toute la durée de
    // la résolution, cf. commentaire sur MKPSharedData::gnn_client).
    std::unique_ptr<GNNBoundBackend> gnn_client;
    double t_gnn_warmup_s = 0.0;
    if (bound_mode == BoundMode::GNN || bound_mode == BoundMode::GNN_ROOT ||
        bound_mode == BoundMode::GNN_ROOT_DLD) {
        if (gnn_checkpoint_path.empty())
            throw std::invalid_argument(
                "solve_gecode_bab_mkp : bound_mode=GNN/GNN_ROOT/GNN_ROOT_DLD requiert "
                "gnn_checkpoint_path");
        if (bound_mode == BoundMode::GNN_ROOT_DLD || gnn_backend == "native" || gnn_backend == "native_aoti") {
            // GNN_ROOT_DLD : adaptateur mono-nœud (query()) autour du
            // backend batché DLD (GNNDLDSdEditNativeBatched, query_batch,
            // taille de batch 1) — cf. gnn_dld_sd_edit_native.hpp. Une seule
            // topologie de graphe disponible côté DLD (sd_edit) : pas de
            // gnn_variant=homogene ni de backend ipc/native_aoti pour ce mode.
            if (bound_mode == BoundMode::GNN_ROOT_DLD) {
                if (gnn_backend != "native")
                    throw std::invalid_argument(
                        "solve_gecode_bab_mkp : bound_mode=GNN_ROOT_DLD ne supporte que "
                        "gnn_backend=native (pas d'adaptateur IPC/AOTI DLD pour l'instant)");
                gnn_client = std::make_unique<GNNDLDSdEditNative>(
                    gnn_checkpoint_path, gnn_device, n, m, N, epsilon, c, p, A, b,
                    use_surrogate_zeta);
            } else {
            // build_graph_cpp() est réimplémenté séparément pour chaque
            // topologie de graphe : GNNSLDNative (gnn_sld_native.cpp,
            // gnn_variant="homogene", clique scénario-scénario) et
            // GNNSdEditNative (gnn_sd_edit_native.cpp, gnn_variant="sd_edit",
            // étoile ancrée au scénario 0 — cf. mémoire "portage backend natif
            // sd_edit"). Échec explicite pour toute autre valeur plutôt qu'une
            // borne calculée silencieusement sur la mauvaise topologie de graphe.
            //
            // gnn_backend=native_aoti : backend AOTInductor (au lieu de
            // TorchScript), gain wall-clock ×1.77-1.79 mesuré sur le forward
            // seul (cf. mémoire "aoti-sdedit-wallclock-result") — sd_edit
            // uniquement pour l'instant (homogène non porté).
            if (gnn_backend == "native_aoti" && gnn_variant != "sd_edit")
                throw std::invalid_argument(
                    "solve_gecode_bab_mkp : gnn_backend=native_aoti ne supporte que "
                    "gnn_variant=sd_edit pour l'instant (homogène non porté)");

            if (gnn_backend == "native_aoti")
                gnn_client = std::make_unique<GNNSdEditNativeAOTI>(
                    gnn_checkpoint_path, gnn_device, n, m, N, epsilon, c, p, A, b);
            else if (gnn_variant == "homogene")
                gnn_client = std::make_unique<GNNSLDNative>(
                    gnn_checkpoint_path, gnn_device, n, m, N, epsilon, c, p, A, b);
            else if (gnn_variant == "sd_edit")
                gnn_client = std::make_unique<GNNSdEditNative>(
                    gnn_checkpoint_path, gnn_device, n, m, N, epsilon, c, p, A, b);
            else
                throw std::invalid_argument(
                    "solve_gecode_bab_mkp : gnn_backend=native ne supporte que "
                    "gnn_variant=homogene|sd_edit (variante \"" + gnn_variant +
                    "\" non portée en C++) — utiliser gnn_backend=ipc");
            }

            // Pré-chauffage TorchScript (identique pour les deux variantes —
            // même mécanisme JIT "profiling", indépendant de la topologie du
            // graphe) : l'executor JIT (mode "profiling")
            // recompile une fois PAR FORME distincte de tenseur d'entrée
            // (n_g = nb d'items encore libres, qui décroît avec la profondeur
            // du B&B), puis met le plan compilé en cache indéfiniment — mesuré
            // ~2026-07-11 (cf. mémoire projet
            // "gnn_inference_forward_cost_benchmark") : jusqu'à ~0.3s pour la
            // première rencontre de n_g=20, quasi gratuit ensuite. Sans ce
            // pré-chauffage, chaque profondeur inédite paie cette taxe une
            // fois pendant la recherche chronométrée — ici on la paie une
            // seule fois par valeur de n_g, avant de commencer, pour que le
            // reste de la recherche ne voie que le régime stable.
            {
                auto t_warm0 = Clock::now();
                std::vector<int8_t> warm_mask(n, 0);
                for (int n_g = 1; n_g <= n; ++n_g) {
                    for (int k = 0; k < n_g; ++k) warm_mask[k] = 2;
                    for (int k = n_g; k < n; ++k) warm_mask[k] = 0;
                    GNNCallTiming warm_timing;
                    // warm_mask ne fixe jamais rien à 1 (seulement 0/libre) :
                    // b_res == b (capacité pleine) est donc correct ici, pas
                    // besoin de le recalculer pour ce simple pré-chauffage de
                    // formes.
                    for (int rep = 0; rep < 2; ++rep)
                        gnn_client->query(warm_mask, b, /*curr_cost=*/0.0,
                                          /*depth=*/n - n_g, warm_timing);
                }
                t_gnn_warmup_s = std::chrono::duration<double>(Clock::now() - t_warm0).count();
            }
        } else if (gnn_backend == "ipc") {
            gnn_client = std::make_unique<GNNSLDClient>(
                gnn_python_exe, gnn_server_script, gnn_checkpoint_path, gnn_device,
                n, m, N, epsilon, c, p, A, b, gnn_variant);
        } else {
            throw std::invalid_argument(
                "solve_gecode_bab_mkp : gnn_backend inconnu (\"" + gnn_backend +
                "\"), attendu \"ipc\", \"native\" ou \"native_aoti\"");
        }
        shared.gnn_client = gnn_client.get();
    }

    // ── 2. DFS B&B avec Gecode Spaces ────────────────────────────────────────
    // On gère nous-mêmes la pile DFS (LIFO) en utilisant l'API publique de
    // Gecode : status() / choice() / commit() / clone().
    //
    // Séquence par nœud :
    //   1. pop(space) depuis la pile (dernier poussé = exploré en premier)
    //   2. space->status() : propage les contraintes, retourne SS_BRANCH/SS_SOLVED/SS_FAILED
    //   3. space->choice() : décision de branchement (variable RATIO, valeur 1 en premier)
    //   4. Pour a ∈ {0,1} :
    //      child = space->clone()
    //      child->commit(*c, a)   → BoundBrancher::commit() → more() → obj ≤ ub_int
    //      child->status()        → propage, y compris obj ≤ ub_int
    //      push_back(child)       si child_ub > incumbent
    //   5. delete c ; delete space

    auto time_ok = [&]() {
        if (time_lim <= 0.0) return true;
        return std::chrono::duration<double>(Clock::now() - t0).count() < time_lim;
    };

    using NodeTup = std::tuple<double, int, MKPSpaceSLD*>;  // (ub, depth, space)
    std::vector<NodeTup> open;  // pile DFS (LIFO)

    int64_t nodes          = 0;
    int64_t fails          = 0;
    int64_t max_depth      = 0;
    int64_t incumbent_impr = 0;
    double  best_val       = greedy_val;
    MKPSpaceSLD* best     = nullptr;

    // Initialiser la racine
    {
        MKPSpaceSLD* root = new MKPSpaceSLD(&shared);
        SpaceStatus rst = root->status();
        if (rst == SS_BRANCH) {
            double rub = root->obj.max() / (double)MKPSharedData::SCALE;
            open.push_back({rub, 0, root});
        } else if (rst == SS_SOLVED) {
            best_val = root->obj.val() / (double)MKPSharedData::SCALE;
            best = root; ++incumbent_impr;
        } else {
            delete root;  // infaisable dès la racine
        }
    }

    while (!open.empty() && time_ok()) {
        auto [ub, depth, space] = open.back(); open.pop_back();

        // Élagage par incumbent
        if (ub <= best_val + 1e-6) { ++fails; delete space; continue; }

        SpaceStatus st = space->status();

        if (st == SS_FAILED) { ++fails; delete space; continue; }

        if (st == SS_SOLVED) {
            ++nodes;
            double val = space->obj.val() / (double)MKPSharedData::SCALE;
            if (val > best_val + 1e-9) {
                best_val = val; delete best; best = space; space = nullptr;
                ++incumbent_impr;
            } else { delete space; }
            continue;
        }

        // SS_BRANCH : branchement
        ++nodes;
        if (depth > max_depth) max_depth = depth;

        const Gecode::Choice* c = space->choice();

        for (int a = 0; a < 2; ++a) {
            MKPSpaceSLD* child = static_cast<MKPSpaceSLD*>(space->clone());
            child->commit(*c, a);                // fixation seule (defer_bound_to_driver=true,
                                                   // cf. BoundBrancher::commit()) — more() n'est
                                                   // PAS appelé ici
            SpaceStatus cst = child->status();    // propagation CP pure : capacités réifiées,
                                                   // chance constraint — pas encore de borne

            if (cst == SS_FAILED) { ++fails; delete child; continue; }

            // Borne (SLD/DLD/GNN) calculée seulement APRÈS que la
            // propagation CP native ait eu sa chance de tuer le nœud
            // gratuitement — cf. revue niveau 1 point A : dans l'ancien
            // ordre (borne postée depuis commit(), avant tout status()), le
            // coût GNN complet était payé même sur les nœuds que fails_from_cp
            // montre déjà tués par le CP seul, indépendamment de la borne.
            // depth < n : encore en phase de branchement sur x (more() ne
            // fait rien lors du branchement sur z qui suit, cf. more()) —
            // more() retourne false si aucune contrainte n'a été postée
            // (REF, GNN_ROOT déjà réchauffé, gnn_min_depth_frac non atteint,
            // ou borne qui n'améliore pas obj.max()), auquel cas le second
            // status() serait pur gaspillage et est sauté.
            if (depth < n && child->more()) {
                cst = child->status();            // repropage la borne fraîchement postée
                if (cst == SS_FAILED) { ++fails; delete child; continue; }
            }

            double cub = child->obj.max() / (double)MKPSharedData::SCALE;
            if (cub <= best_val + 1e-6) { ++fails; delete child; continue; }

            if (cst == SS_SOLVED) {
                double val = child->obj.val() / (double)MKPSharedData::SCALE;
                if (val > best_val + 1e-9) {
                    best_val = val; delete best; best = child; child = nullptr;
                    ++incumbent_impr;
                } else { delete child; }
                continue;
            }

            open.push_back({cub, depth + 1, child});
        }

        delete c;
        delete space;
    }

    bool timed_out = !open.empty();   // queue non vide → time limit atteinte

    // Libérer les nœuds restants dans la file
    while (!open.empty()) {
        auto [ub, d, sp] = open.back(); open.pop_back();
        delete sp;
    }

    double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();

    // ── 7. Construction du résultat ───────────────────────────────────────────
    GecodeSLDResult res;
    res.greedy_value     = greedy_val;
    res.best_open_ub     = shared.best_open_ub;
    res.solve_time_s     = elapsed;
    res.nodes            = nodes;
    res.fails            = fails;
    res.fails_from_bound = static_cast<int64_t>(shared.fails_from_bound);
    res.fails_from_cp    = fails - res.fails_from_bound;
    res.depth            = max_depth;
    res.incumbent_improvements = incumbent_impr;
    res.more_calls       = static_cast<int64_t>(shared.more_calls);
    res.ub_tightened       = static_cast<int64_t>(shared.ub_tightened);
    res.root_bound_time_s  = shared.root_bound_time_s;
    res.root_bound_value   = shared.root_bound_value;
    res.t_bound_total_s    = shared.t_bound_total_s;
    res.t_gnn_warmup_s     = t_gnn_warmup_s;
    res.t_bab_pure_s       = res.solve_time_s - shared.t_bound_total_s - t_gnn_warmup_s;
    res.clone_count        = shared.clone_count;
    res.t_clone_state_s    = shared.t_clone_state_s;
    res.t_clone_total_s    = shared.t_clone_total_s;
    res.t_gnn_graph_build_s   = shared.t_gnn_graph_build_s;
    res.t_gnn_graph_features_s = shared.t_gnn_graph_features_s;
    res.t_gnn_graph_edges_s    = shared.t_gnn_graph_edges_s;
    res.t_gnn_forward_s       = shared.t_gnn_forward_s;
    res.t_gnn_bound_compute_s = shared.t_gnn_bound_compute_s;
    res.t_gnn_ipc_overhead_s  = shared.t_gnn_ipc_overhead_s;
    res.nodes_by_depth             = shared.nodes_by_depth;
    res.ub_tightened_by_depth      = shared.ub_tightened_by_depth;
    res.fails_from_bound_by_depth  = shared.fails_from_bound_by_depth;
    res.t_bound_by_depth           = shared.t_bound_by_depth;

    if (best) {
        // Dé-scaler l'objectif : best->obj.val() est en unités SCALE
        res.objective = static_cast<double>(best->obj.val()) / shared.SCALE;
        res.feasible  = true;
        res.optimal   = !timed_out;
        res.status    = timed_out ? "FEASIBLE" : "OPTIMAL";
        delete best;
    } else {
        // Aucune solution trouvée : soit infaisable, soit time-out avant la première solution
        res.objective = std::numeric_limits<double>::quiet_NaN();
        res.feasible  = false;
        res.optimal   = false;
        res.status    = timed_out ? "TIMEOUT" : "INFEASIBLE";
    }

    return res;
}
