/*
 * gecode_bab.cpp — Implémentation de greedy_mkp et solve_gecode_bab_mkp.
 */

#include "gecode_bab.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>
#include <queue>
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

GecodeV1LDResult solve_gecode_bab_mkp(
    int n, int m, int N, double epsilon,
    const Vec1d& c, const Vec1d& p,
    const Vec3d& A, const Vec2d& b,
    double    time_lim,
    int       T_root,
    int       T_node,
    BoundMode bound_mode)
{
    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();

    // ── 1. Solution gloutonne — incumbent initial ─────────────────────────────
    // On calcule une solution faisable avant de lancer Gecode.
    // Sa valeur est postée comme contrainte obj >= greedy_int dans le constructeur
    // de MKPSpaceV1LD, ce qui élimine d'emblée les branches sous-optimales.
    auto [greedy_val, greedy_x] = greedy_mkp(c, p, A, b, epsilon);
    int greedy_int = static_cast<int>(std::floor(greedy_val * MKPSharedData::SCALE));

    // ── 2. Données partagées ──────────────────────────────────────────────────
    MKPSharedData shared(n, m, N, epsilon, c, p, A, b, T_node);
    shared.T_root         = T_root;
    shared.bound_mode     = bound_mode;
    shared.initial_lb_int = greedy_int;   // fournit l'incumbent à l'espace racine

    // ── 2. Best-first B&B avec Gecode Spaces ─────────────────────────────────
    // On implémente nous-mêmes la file de priorité (par UB décroissante)
    // en utilisant l'API publique de Gecode : status() / choice() / commit() / clone().
    // Cela donne la même stratégie de recherche que bab_mkp.cpp (best-first).
    //
    // Séquence par nœud :
    //   1. pop(space) depuis la priority_queue (UB max en tête)
    //   2. space->status() : propage les contraintes, retourne SS_BRANCH/SS_SOLVED/SS_FAILED
    //   3. space->choice() : décision de branchement (variable RATIO, valeur 1 en premier)
    //   4. Pour a ∈ {0,1} :
    //      child = space->clone()
    //      child->commit(*c, a)   → BoundBrancher::commit() → more() → obj ≤ ub_int
    //      child->status()        → propage, y compris obj ≤ ub_int
    //      push(child_ub, child)  si child_ub > incumbent
    //   5. delete c ; delete space

    auto time_ok = [&]() {
        if (time_lim <= 0.0) return true;
        return std::chrono::duration<double>(Clock::now() - t0).count() < time_lim;
    };

    using NodeTup = std::tuple<double, int, MKPSpaceV1LD*>;  // (ub, depth, space)
    auto node_cmp = [](const NodeTup& a, const NodeTup& b) {
        return std::get<0>(a) < std::get<0>(b);   // max-heap par UB
    };
    std::priority_queue<NodeTup, std::vector<NodeTup>, decltype(node_cmp)> open(node_cmp);

    int64_t nodes          = 0;
    int64_t fails          = 0;
    int64_t max_depth      = 0;
    int64_t incumbent_impr = 0;
    double  best_val       = greedy_val;
    MKPSpaceV1LD* best     = nullptr;

    // Initialiser la racine
    {
        MKPSpaceV1LD* root = new MKPSpaceV1LD(&shared);
        SpaceStatus rst = root->status();
        if (rst == SS_BRANCH) {
            double rub = root->obj.max() / (double)MKPSharedData::SCALE;
            open.push({rub, 0, root});
        } else if (rst == SS_SOLVED) {
            best_val = root->obj.val() / (double)MKPSharedData::SCALE;
            best = root; ++incumbent_impr;
        } else {
            delete root;  // infaisable dès la racine
        }
    }

    while (!open.empty() && time_ok()) {
        auto [ub, depth, space] = open.top(); open.pop();

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
            MKPSpaceV1LD* child = static_cast<MKPSpaceV1LD*>(space->clone());
            child->commit(*c, a);        // commit → more() → obj ≤ ub_int
            SpaceStatus cst = child->status();  // propage, y compris obj ≤ ub_int

            if (cst == SS_FAILED) { ++fails; delete child; continue; }

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

            open.push({cub, depth + 1, child});
        }

        delete c;
        delete space;
    }

    bool timed_out = !open.empty();   // queue non vide → time limit atteinte

    // Libérer les nœuds restants dans la file
    while (!open.empty()) {
        auto [ub, d, sp] = open.top(); open.pop();
        delete sp;
    }

    double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();

    // ── 7. Construction du résultat ───────────────────────────────────────────
    GecodeV1LDResult res;
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
    res.ub_tightened     = static_cast<int64_t>(shared.ub_tightened);

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
