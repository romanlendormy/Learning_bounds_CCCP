/*
 * gecode_bab_bucketed.cpp — Implémentation de solve_gecode_bab_mkp_bucketed.
 *
 * Ce qui change par rapport à la boucle DFS de gecode_bab.cpp : le calcul de
 * la borne (commit() → BoundBrancher::commit() → more() → obj ≤ ub_int) reste
 * STRICTEMENT identique et synchrone. Ce qui est différé, c'est la DÉCISION
 * de pruning/incumbent pour un enfant fraîchement créé : au lieu de comparer
 * `cub` à `best_val` tout de suite (gecode_bab.cpp:257-258), on stocke
 * l'enfant dans un buffer ("bucket") indexé par n_g = n - depth_items (le
 * nombre d'items encore libres, calculé en scannant x — même définition que
 * more() ligne 728 de gecode_bab.hpp), et on ne juge tout le bucket que
 * lorsqu'il atteint la taille K ou que la pile `open` se vide.
 *
 * Les échecs de propagation (SS_FAILED) restent détectés IMMÉDIATEMENT au
 * commit — ils ne dépendent pas d'un incumbent, rien à différer là.
 *
 * Protocole de flush choisi (un choix parmi d'autres, cf. plan) : snapshot de
 * best_val UNE FOIS au début du flush, tous les membres du bucket sont jugés
 * contre ce snapshot (donc "en aveugle" les uns des autres), la meilleure
 * feuille résolue du bucket est comparée à best_val UNE SEULE FOIS en fin de
 * flush.
 */

#include "gecode_bab_bucketed.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <vector>

// ── greedy_mkp — copie locale ────────────────────────────────────────────────
// Identique à gecode_bab.cpp::greedy_mkp. Dupliquée ici (même convention que
// load_song_local, dupliqué dans chaque main_*.cpp de ce dossier) pour ne PAS
// compiler gecode_bab.cpp dans ce binaire — celui-ci référence
// GNNV1LDNative/GNNSdEditNative/GNNV1LDClient (LibTorch), jamais utilisés par
// cette variante (bound_mode restreint à V1LD/V1CLD/REF).
std::pair<double, std::vector<int8_t>> greedy_mkp(
    const Vec1d& c, const Vec1d& p,
    const Vec3d& A, const Vec2d& b,
    double epsilon)
{
    const int n = static_cast<int>(c.size());
    const int N = static_cast<int>(p.size());
    const int m = static_cast<int>(b[0].size());

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

namespace {

// Nombre d'items encore libres → n_g = n - depth_items. Même définition que
// more() (gecode_bab.hpp:696-728), recalculée ici en scannant le membre
// public `x` — ne modifie rien dans MKPSpaceV1LD, juste une lecture.
int compute_items_depth(const MKPSpaceV1LD& space, int n) {
    int n_free = 0;
    for (int k = 0; k < n; ++k)
        if (!space.x[k].assigned()) ++n_free;
    return n - n_free;
}

struct PendingEntry {
    double        ub;           // obj.max()/SCALE (== obj.val()/SCALE si is_solved)
    int           outer_depth;  // profondeur générique (niveau d'arbre, x+z confondus)
    MKPSpaceV1LD* space;
    bool          is_solved;
};

} // namespace

BucketedBabResult solve_gecode_bab_mkp_bucketed(
    int n, int m, int N, double epsilon,
    const Vec1d& c, const Vec1d& p,
    const Vec3d& A, const Vec2d& b,
    double    time_lim,
    int       T_root,
    int       T_node,
    BoundMode bound_mode,
    bool      patience_mode,
    double    patience_tol,
    int       K)
{
    if (bound_mode == BoundMode::GNN || bound_mode == BoundMode::GNN_ROOT)
        throw std::invalid_argument(
            "solve_gecode_bab_mkp_bucketed : bound_mode=GNN/GNN_ROOT hors périmètre "
            "(cette variante teste le risque algorithmique du bucketing à borne "
            "inchangée V1LD/V1CLD/REF — le batching du forward GNN est une suite "
            "séparée, non implémentée ici)");
    if (K < 1)
        throw std::invalid_argument("solve_gecode_bab_mkp_bucketed : K doit être >= 1");

    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();

    // ── 1. Solution gloutonne — incumbent initial (identique à gecode_bab.cpp) ──
    auto [greedy_val, greedy_x] = greedy_mkp(c, p, A, b, epsilon);
    int greedy_int = static_cast<int>(std::floor(greedy_val * MKPSharedData::SCALE));

    // ── 2. Données partagées (identique) ─────────────────────────────────────
    MKPSharedData shared(n, m, N, epsilon, c, p, A, b, T_node);
    shared.T_root         = T_root;
    shared.bound_mode     = bound_mode;
    shared.patience_mode  = patience_mode;
    shared.patience_tol   = patience_tol;
    shared.initial_lb_int = greedy_int;

    auto time_ok = [&]() {
        if (time_lim <= 0.0) return true;
        return std::chrono::duration<double>(Clock::now() - t0).count() < time_lim;
    };

    using NodeTup = std::tuple<double, int, MKPSpaceV1LD*>;  // (ub, depth, space) — pour `open`
    std::vector<NodeTup> open;  // pile DFS (LIFO), identique à gecode_bab.cpp

    // Un bucket par n_g possible (0..n) — même convention de taille n+1 que
    // MKPSharedData::nodes_by_depth.
    std::vector<std::vector<PendingEntry>> pending(n + 1);

    int64_t nodes          = 0;
    int64_t fails          = 0;
    int64_t max_depth      = 0;
    int64_t incumbent_impr = 0;
    double  best_val       = greedy_val;
    MKPSpaceV1LD* best     = nullptr;

    int64_t n_bucket_flushes          = 0;
    int64_t sum_bucket_fill_at_flush  = 0;
    int64_t max_pending_live          = 0;

    auto count_pending_live = [&]() {
        int64_t total = 0;
        for (const auto& bucket : pending) total += static_cast<int64_t>(bucket.size());
        return total;
    };

    auto any_pending_nonempty = [&]() {
        for (const auto& bucket : pending) if (!bucket.empty()) return true;
        return false;
    };

    // Flush du bucket à l'indice n_g = d : juge tous ses membres contre UN
    // SEUL snapshot de best_val (pris au début), donc "en aveugle" les uns
    // des autres — c'est exactement l'effet qu'aurait un batching GNN qui
    // calculerait leurs bornes en un seul appel avant que l'incumbent ne
    // puisse être mis à jour par l'un d'eux.
    auto flush_bucket = [&](int d) {
        auto& bucket = pending[d];
        if (bucket.empty()) return;

        ++n_bucket_flushes;
        sum_bucket_fill_at_flush += static_cast<int64_t>(bucket.size());

        const double snapshot_best = best_val;
        double best_leaf_val = -std::numeric_limits<double>::infinity();
        MKPSpaceV1LD* best_leaf_space = nullptr;

        for (auto& e : bucket) {
            if (e.is_solved) {
                if (e.ub > best_leaf_val) {
                    if (best_leaf_space) delete best_leaf_space;
                    best_leaf_val = e.ub;
                    best_leaf_space = e.space;
                } else {
                    delete e.space;
                }
                continue;
            }
            if (e.ub <= snapshot_best + 1e-6) {
                ++fails;
                delete e.space;
                continue;
            }
            open.push_back({e.ub, e.outer_depth, e.space});
        }
        bucket.clear();

        if (best_leaf_space) {
            if (best_leaf_val > best_val + 1e-9) {
                best_val = best_leaf_val;
                delete best;
                best = best_leaf_space;
                ++incumbent_impr;
            } else {
                delete best_leaf_space;
            }
        }
    };

    // Initialiser la racine (identique à gecode_bab.cpp — le bucketing ne
    // s'applique qu'à la création des enfants, pas à la racine).
    {
        MKPSpaceV1LD* root = new MKPSpaceV1LD(&shared);
        SpaceStatus rst = root->status();
        if (rst == SS_BRANCH) {
            double rub = root->obj.max() / (double)MKPSharedData::SCALE;
            open.push_back({rub, 0, root});
        } else if (rst == SS_SOLVED) {
            best_val = root->obj.val() / (double)MKPSharedData::SCALE;
            best = root; ++incumbent_impr;
        } else {
            delete root;
        }
    }

    while ((!open.empty() || any_pending_nonempty()) && time_ok()) {
        if (open.empty()) {
            // Plus rien à dépiler : forcer la progression en vidant tous les
            // buckets non vides (ordre n_g croissant, arbitraire mais fixe).
            for (int d = 0; d <= n; ++d) flush_bucket(d);
            continue;
        }

        auto [ub, depth, space] = open.back(); open.pop_back();

        // Élagage par incumbent (identique à gecode_bab.cpp — ce test-ci,
        // sur la pile `open` elle-même, n'est PAS différé : chaque nœud de
        // `open` a déjà survécu à un flush avec le best_val d'alors, ce
        // deuxième test utilise le best_val le plus frais possible).
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

        const Gecode::Choice* c_choice = space->choice();

        for (int a = 0; a < 2; ++a) {
            MKPSpaceV1LD* child = static_cast<MKPSpaceV1LD*>(space->clone());
            child->commit(*c_choice, a);         // commit → more() → obj ≤ ub_int (INCHANGÉ)
            SpaceStatus cst = child->status();   // propage, y compris obj ≤ ub_int

            if (cst == SS_FAILED) { ++fails; delete child; continue; }

            double cub = child->obj.max() / (double)MKPSharedData::SCALE;
            int ng_depth = compute_items_depth(*child, n);

            if (ng_depth == n) {
                // x entièrement fixé : more() ne calcule plus aucune borne à
                // partir d'ici (gecode_bab.hpp:721, "free_vars.empty() →
                // return") — le commit de CE choice() vient forcément du
                // brancher standard sur z (satisfaction des scénarios),
                // jamais de BoundBrancher. Rien à différer/batcher : un futur
                // batching GNN ne concernerait QUE les nœuds où more()
                // calcule réellement une borne (ng_depth < n). Traiter ce
                // nœud dans un bucket serait mesurer un risque qui n'existe
                // pas dans la vraie cible du batching — retiré après un
                // premier passage où ça gonflait artificiellement le nombre
                // de nœuds (le sous-arbre z, potentiellement grand, se
                // retrouvait tout entier différé dans un seul bucket
                // pending[n]). Donc : traitement IMMÉDIAT, identique à
                // gecode_bab.cpp (élagage direct contre l'incumbent le plus
                // frais).
                if (cub <= best_val + 1e-6) { ++fails; delete child; continue; }
                if (cst == SS_SOLVED) {
                    if (cub > best_val + 1e-9) {
                        best_val = cub; delete best; best = child; child = nullptr;
                        ++incumbent_impr;
                    } else { delete child; }
                    continue;
                }
                open.push_back({cub, depth + 1, child});
                continue;
            }

            // ng_depth < n : more() calcule/poste réellement une borne ICI —
            // seule zone où un futur batching GNN s'appliquerait. On défère
            // la décision de pruning/incumbent dans un bucket par n_g.
            pending[ng_depth].push_back({cub, depth + 1, child, cst == SS_SOLVED});

            max_pending_live = std::max(max_pending_live, count_pending_live());

            if (static_cast<int>(pending[ng_depth].size()) == K)
                flush_bucket(ng_depth);
        }

        delete c_choice;
        delete space;
    }

    bool timed_out = !open.empty() || any_pending_nonempty();

    // Libérer les nœuds restants — `open` ET tous les buckets `pending`.
    while (!open.empty()) {
        auto [ub, d, sp] = open.back(); open.pop_back();
        delete sp;
    }
    for (auto& bucket : pending) {
        for (auto& e : bucket) delete e.space;
        bucket.clear();
    }

    double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();

    // ── Construction du résultat (mêmes champs que solve_gecode_bab_mkp,
    // shared.* étant peuplé identiquement puisque more() est inchangé) ──────
    BucketedBabResult res;
    res.K                = K;
    res.n_bucket_flushes  = n_bucket_flushes;
    res.mean_bucket_fill_at_flush =
        n_bucket_flushes > 0
            ? static_cast<double>(sum_bucket_fill_at_flush) / static_cast<double>(n_bucket_flushes)
            : 0.0;
    res.max_pending_live = max_pending_live;

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
    res.t_gnn_warmup_s     = 0.0;
    res.t_bab_pure_s       = res.solve_time_s - shared.t_bound_total_s;
    res.clone_count        = shared.clone_count;
    res.t_clone_state_s    = shared.t_clone_state_s;
    res.t_clone_total_s    = shared.t_clone_total_s;
    res.nodes_by_depth             = shared.nodes_by_depth;
    res.ub_tightened_by_depth      = shared.ub_tightened_by_depth;
    res.fails_from_bound_by_depth  = shared.fails_from_bound_by_depth;
    res.t_bound_by_depth           = shared.t_bound_by_depth;

    if (best) {
        res.objective = static_cast<double>(best->obj.val()) / shared.SCALE;
        res.feasible  = true;
        res.optimal   = !timed_out;
        res.status    = timed_out ? "FEASIBLE" : "OPTIMAL";
        delete best;
    } else {
        res.objective = std::numeric_limits<double>::quiet_NaN();
        res.feasible  = false;
        res.optimal   = false;
        res.status    = timed_out ? "TIMEOUT" : "INFEASIBLE";
    }

    return res;
}
