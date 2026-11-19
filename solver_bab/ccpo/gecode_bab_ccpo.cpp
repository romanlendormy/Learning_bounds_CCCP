/*
 * gecode_bab_ccpo.cpp — Implémentation de greedy_ccpo et solve_gecode_bab_ccpo.
 *
 * Boucle DFS manuelle copiée du patron de solve_gecode_bab_mkp
 * (solver_bab/mkp/gecode_bab.cpp), réduite au strict REF : aucun more(),
 * aucune borne, aucune statistique par profondeur.
 */

#include "gecode_bab_ccpo.hpp"

#include <limits>
#include <tuple>

// ═════════════════════════════════════════════════════════════════════════
// greedy_ccpo — incumbent initial simple
// ═════════════════════════════════════════════════════════════════════════
//
// Sélectionne les k actifs de meilleur ratio rendement/risque, alloue Q en
// remplissant chacun à U (ou moins pour respecter les bornes des slots
// restants), puis vérifie la contrainte de chance. AUCUNE logique de
// réparation si le résultat est infaisable (choix assumé pour cette première
// version -- cf. gecode_bab_ccpo.hpp) : dans ce cas on retourne feasible=false
// et le B&B part sans incumbent, juste plus lentement.

GreedyCCPOResult greedy_ccpo(
    int n, int N, int k, int Q, int L, int U,
    const std::vector<int>& c_int,
    const std::vector<std::vector<SigmaTriplet>>& sigma,
    long long B, int min_feas)
{
    std::vector<double> avg_risk(n, 0.0);
    for (int j = 0; j < N; ++j)
        for (const auto& t : sigma[j])
            if (t[0] == t[1]) avg_risk[t[0]] += t[2];
    for (int i = 0; i < n; ++i) avg_risk[i] /= std::max(1, N);

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        double ra = (avg_risk[a] > 1e-9) ? c_int[a] / avg_risk[a] : c_int[a] * 1e12;
        double rb = (avg_risk[b] > 1e-9) ? c_int[b] / avg_risk[b] : c_int[b] * 1e12;
        return ra > rb;
    });

    const int support_size = std::min(k, Q / L);
    if (support_size < 1 || support_size * U < Q || support_size * L > Q)
        return {0.0, std::vector<int>(n, 0), false};  // grille infaisable structurellement

    std::vector<int> q(n, 0);
    int remaining = Q;
    for (int idx = 0; idx < support_size; ++idx) {
        int i = order[idx];
        int slots_left_after = support_size - idx - 1;
        int alloc = std::min(U, remaining - slots_left_after * L);
        q[i] = alloc;
        remaining -= alloc;
    }

    double value = 0.0;
    for (int i = 0; i < n; ++i) value += static_cast<double>(c_int[i]) * q[i];

    int satisfied = 0;
    for (int j = 0; j < N; ++j) {
        long long qtq = 0;
        for (const auto& t : sigma[j]) {
            const int i = t[0], kk = t[1], val = t[2];
            long long term = static_cast<long long>(val) * q[i] * q[kk];
            qtq += (i == kk) ? term : 2 * term;
        }
        if (qtq <= B) ++satisfied;
    }

    bool feasible = satisfied >= min_feas;
    return {feasible ? value : 0.0, q, feasible};
}

// ═════════════════════════════════════════════════════════════════════════
// solve_gecode_bab_ccpo
// ═════════════════════════════════════════════════════════════════════════

CCPOResult solve_gecode_bab_ccpo(
    int n, int N, int k, int Q, int L, int U, long long B, int min_feas,
    const std::vector<int>& c_int,
    const std::vector<std::vector<SigmaTriplet>>& sigma,
    double time_lim,
    BoundMode bound_mode,
    int T_root,
    int T_node,
    int omp_threads)
{
    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();

    CCPOSharedData shared(n, N, k, Q, L, U, B, min_feas, c_int, sigma);
    shared.bound_mode = bound_mode;
    shared.T_root = T_root;
    shared.T_node = T_node;
    shared.omp_threads = omp_threads;

    // Infaisabilité globale par presolve : si plus de scénarios sont
    // PROUVÉS infaisables (cf. CCPOSharedData::infeasible_scenario) que la
    // marge de tolérance ne le permet (N - min_feas), aucun q ne peut
    // jamais satisfaire la contrainte de chance -- inutile de construire le
    // moindre nœud B&B. Court-circuit avant même le glouton.
    if (shared.n_infeasible > N - min_feas) {
        CCPOResult res;
        res.objective = std::numeric_limits<double>::quiet_NaN();
        res.feasible = false;
        res.optimal = false;
        res.status = "INFEASIBLE";
        res.greedy_value = std::numeric_limits<double>::quiet_NaN();
        res.solve_time_s = std::chrono::duration<double>(Clock::now() - t0).count();
        res.nodes = 0; res.fails = 0; res.depth = 0; res.incumbent_improvements = 0;
        return res;
    }

    GreedyCCPOResult greedy = greedy_ccpo(n, N, k, Q, L, U, c_int, sigma, B, min_feas);
    if (greedy.feasible)
        shared.initial_lb_int = static_cast<int>(greedy.value);

    auto time_ok = [&]() {
        if (time_lim <= 0.0) return true;
        return std::chrono::duration<double>(Clock::now() - t0).count() < time_lim;
    };

    using NodeTup = std::tuple<double, int, CCPOSpaceSLD*>;
    std::vector<NodeTup> open;

    int64_t nodes = 0, fails = 0, max_depth = 0, incumbent_impr = 0;
    bool   has_incumbent = greedy.feasible;
    double best_val      = greedy.feasible ? greedy.value : -1.0;
    CCPOSpaceSLD* best = nullptr;

    {
        CCPOSpaceSLD* root = new CCPOSpaceSLD(&shared);
        SpaceStatus rst = root->status();
        if (rst == SS_BRANCH) {
            open.push_back({static_cast<double>(root->obj.max()), 0, root});
        } else if (rst == SS_SOLVED) {
            best_val = root->obj.val(); has_incumbent = true;
            best = root; ++incumbent_impr;
        } else {
            delete root;  // infaisable dès la racine
        }
    }

    while (!open.empty() && time_ok()) {
        auto [ub, depth, space] = open.back(); open.pop_back();

        if (has_incumbent && ub <= best_val + 1e-6) { ++fails; delete space; continue; }

        SpaceStatus st = space->status();
        if (st == SS_FAILED) { ++fails; delete space; continue; }

        if (st == SS_SOLVED) {
            ++nodes;
            double val = space->obj.val();
            if (!has_incumbent || val > best_val + 1e-9) {
                best_val = val; has_incumbent = true;
                delete best; best = space; space = nullptr;
                ++incumbent_impr;
            } else {
                delete space;
            }
            continue;
        }

        ++nodes;
        if (depth > max_depth) max_depth = depth;

        const Gecode::Choice* c = space->choice();

        for (int a = 0; a < 2; ++a) {
            CCPOSpaceSLD* child = static_cast<CCPOSpaceSLD*>(space->clone());
            child->commit(*c, a);
            SpaceStatus cst = child->status();  // propagation CP pure d'abord

            if (cst == SS_FAILED) { ++fails; delete child; continue; }

            // Borne (SLD) calculée seulement APRÈS que la propagation CP native
            // ait eu sa chance de tuer le nœud gratuitement -- more() est un
            // no-op rapide en mode REF ou sur une feuille (q entièrement fixé),
            // cf. CCPOSpaceSLD::more().
            if (child->more()) {
                cst = child->status();  // repropage la borne fraîchement postée
                if (cst == SS_FAILED) { ++fails; delete child; continue; }
            }

            double cub = child->obj.max();
            if (has_incumbent && cub <= best_val + 1e-6) { ++fails; delete child; continue; }

            if (cst == SS_SOLVED) {
                double val = child->obj.val();
                if (!has_incumbent || val > best_val + 1e-9) {
                    best_val = val; has_incumbent = true;
                    delete best; best = child; child = nullptr;
                    ++incumbent_impr;
                } else {
                    delete child;
                }
                continue;
            }

            open.push_back({cub, depth + 1, child});
        }

        delete c;
        delete space;
    }

    bool timed_out = !open.empty();
    while (!open.empty()) {
        auto [ub, d, sp] = open.back(); open.pop_back();
        delete sp;
    }

    double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();

    CCPOResult res;
    res.greedy_value = greedy.feasible ? greedy.value : std::numeric_limits<double>::quiet_NaN();
    res.solve_time_s = elapsed;
    res.nodes  = nodes;
    res.fails  = fails;
    res.depth  = max_depth;
    res.incumbent_improvements = incumbent_impr;
    res.fails_from_bound  = shared.fails_from_bound;
    res.fails_from_cp     = fails - shared.fails_from_bound;
    res.more_calls        = shared.more_calls;
    res.ub_tightened      = shared.ub_tightened;
    res.root_bound_time_s = shared.root_bound_time_s;
    res.root_bound_value  = shared.root_bound_value;
    res.t_bound_total_s   = shared.t_bound_total_s;

    if (best) {
        res.objective = best->obj.val();
        res.feasible  = true;
        res.optimal   = !timed_out;
        res.status    = timed_out ? "FEASIBLE" : "OPTIMAL";
        delete best;
    } else if (greedy.feasible) {
        // L'incumbent glouton (faisable) sert à élaguer mais n'est pas stocké dans `best` : jamais amélioré => il EST la solution finale.
        res.objective = greedy.value;
        res.feasible  = true;
        res.optimal   = !timed_out;
        res.status    = timed_out ? "FEASIBLE" : "OPTIMAL";
    } else {
        res.objective = std::numeric_limits<double>::quiet_NaN();
        res.feasible  = false;
        res.optimal   = false;
        res.status    = timed_out ? "TIMEOUT" : "INFEASIBLE";
    }

    return res;
}
