/*
 * gecode_solver.cpp — Solveur CP exact (Gecode 6) pour CCMKP et CCSSP.
 *
 * CCMKP : Formulation CP directe avec réification (Reify + RM_IMP)
 *         z_i = 1 → A_i x ≤ b_i   (sans Big-M)
 *
 * CCSSP : Automate produit + extensional + dom-réification
 *
 * SCALE = 1000 pour les coefficients réels → entiers int32.
 */

#include "gecode_solver.hpp"

#include <gecode/int.hh>
#include <gecode/search.hh>
#include <gecode/minimodel.hh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>

using namespace Gecode;

// ── Timer ─────────────────────────────────────────────────────────────────────

using Clock = std::chrono::steady_clock;
static double elapsed_s(std::chrono::time_point<Clock> t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

static constexpr int SCALE = 1000;
static int to_int(double x)       { return static_cast<int>(std::llround(x * SCALE)); }
static int to_int_floor(double x) { return static_cast<int>(std::floor(x * SCALE)); }

// ════════════════════════════════════════════════════════════════════════════════
// CCMKP — CP direct avec réification
// ════════════════════════════════════════════════════════════════════════════════
//
//   max  Σ c_k x_k
//   s.t. z_i = 1  →  Σ_k A_ijk x_k ≤ b_ij   ∀ i, j   (Reify RM_IMP)
//        Σ z_i ≥ ⌈(1-ε)N⌉
//        x ∈ {0,1}^n,  z ∈ {0,1}^N

class MKPSpace : public IntMaximizeSpace {
public:
    BoolVarArray x, z;
    IntVar       obj;

    MKPSpace(int n, int m, int N, int min_feas,
             const std::vector<int>& c_int,
             const std::vector<std::vector<std::vector<int>>>& A_int,
             const std::vector<std::vector<int>>& b_int,
             int obj_max)
        : x(*this, n, 0, 1),
          z(*this, N, 0, 1),
          obj(*this, 0, obj_max)
    {
        // Contraintes réifiées : z_i = 1 → A_i[j] · x ≤ b_i[j]
        IntVarArgs x_int(n);
        for (int k = 0; k < n; ++k) x_int[k] = channel(*this, x[k]);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < m; ++j) {
                IntArgs coefs(n);
                for (int k = 0; k < n; ++k) coefs[k] = A_int[i][j][k];
                linear(*this, coefs, x_int, IRT_LQ, b_int[i][j],
                       Reify(z[i], RM_IMP));
            }
        }

        // Contrainte de chance
        linear(*this, z, IRT_GQ, min_feas);

        // Objectif
        IntArgs c_args(n);
        for (int k = 0; k < n; ++k) c_args[k] = c_int[k];
        linear(*this, c_args, x_int, IRT_EQ, obj);

        branch(*this, x, BOOL_VAR_NONE(), BOOL_VAL_MAX());
        branch(*this, z, BOOL_VAR_NONE(), BOOL_VAL_MAX());
    }

    MKPSpace(MKPSpace& s) : IntMaximizeSpace(s) {
        x.update(*this, s.x);
        z.update(*this, s.z);
        obj.update(*this, s.obj);
    }
    Space*  copy()  override { return new MKPSpace(*this); }
    IntVar  cost()  const override { return obj; }
};

GecodeResult solve_gecode_mkp(
    int n, int m, int N, double epsilon,
    const Vec1d& c, const Vec1d& /*p*/,
    const Vec3d& A, const Vec2d& b,
    double time_lim)
{
    auto t0 = Clock::now();

    int min_feas = static_cast<int>(std::ceil((1.0 - epsilon) * N));

    std::vector<int> c_int(n);
    for (int k = 0; k < n; ++k) c_int[k] = to_int(c[k]);

    std::vector<std::vector<std::vector<int>>> A_int(
        N, std::vector<std::vector<int>>(m, std::vector<int>(n)));
    std::vector<std::vector<int>> b_int(N, std::vector<int>(m));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < n; ++k) A_int[i][j][k] = to_int(A[i][j][k]);
            b_int[i][j] = to_int_floor(b[i][j]);
        }

    int obj_max = 0;
    for (int k = 0; k < n; ++k) obj_max += std::max(0, c_int[k]);

    MKPSpace* model = new MKPSpace(n, m, N, min_feas, c_int, A_int, b_int, obj_max);

    Search::Options opts;
    Search::TimeStop* stop = nullptr;
    if (time_lim > 0.0) {
        stop      = new Search::TimeStop(static_cast<unsigned long long>(time_lim * 1000));
        opts.stop = stop;
    }

    BAB<MKPSpace> engine(model, opts);
    delete model;
    double cold_start_s = elapsed_s(t0);

    MKPSpace* best = nullptr;
    int64_t inc_impr = 0;
    while (MKPSpace* sol = engine.next()) { delete best; best = sol; ++inc_impr; }

    bool timed_out = stop && engine.stopped();
    auto stats = engine.statistics();
    delete stop;

    GecodeResult res;
    res.solve_time_s      = elapsed_s(t0);
    res.cold_start_time_s = cold_start_s;
    res.nodes        = static_cast<int64_t>(stats.node);
    res.fails        = static_cast<int64_t>(stats.fail);
    res.depth        = static_cast<int64_t>(stats.depth);
    res.incumbent_improvements = inc_impr;
    if (best) {
        res.objective = static_cast<double>(best->obj.val()) / SCALE;
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

// ════════════════════════════════════════════════════════════════════════════════
// CCSSP — couverture multi-employés, chance-constraint joint
// ════════════════════════════════════════════════════════════════════════════════
//
// min  Σ_t salary[t] · y[t]
// s.t.
// (a)  extensional(x[e][·], DFA)          ∀e     DFA partagé, GAC dur, non réifié
// (b)  works[e][t] ⟺ x[e][t] ∈ work_set  ∀e,t   canal dom-réifié
// (b') y[t] = Σ_e works[e][t]             ∀t     linear BoolVarArgs
// (c)  zti[i][t] ⟺ y[t] ≥ demand[i][t]   ∀i,t   rel réifié 1er niveau
//      z[i]  ⟺ AND_t zti[i][t]            ∀i     BOT_AND 2e niveau
//      Σ_i z[i] ≥ min_feas                        quantile
// (d)  Σ_t sal[t] · y[t] = total_cost             objectif (scalé)
// (e)  x[e][·] ≤_lex x[e+1][·]           ∀e     brisure symétrie (opt.)

class SSPSpace : public IntMinimizeSpace {
public:
    IntVarArray  x_flat;      // E×T  activité ∈ {0..m-1}
    IntVarArray  y;           // T    dotation y[t] = Σ_e works[e][t]
    BoolVarArray z;           // N    z[i]=1 ↔ scénario i entièrement couvert
    IntVar       total_cost;  // objectif scalé

    SSPSpace(int E, int T, int m, int N, int min_feas,
             const DFA& dfa,
             const IntSet& work_set,
             const std::vector<int>& sal_sc,  // taille T, entiers scalés
             int sal_ub,                       // borne sup. domaine total_cost
             const std::vector<std::vector<int>>& demand,  // [N][T]
             bool break_symmetry)
        : x_flat    (*this, E * T, 0, m - 1),
          y         (*this, T, 0, E),
          z         (*this, N, 0, 1),
          total_cost(*this, 0, sal_ub)
    {
        // ── (a) automate par employé : hard GAC, non réifié ─────────────
        for (int e = 0; e < E; ++e) {
            IntVarArgs row(T);
            for (int t = 0; t < T; ++t) row[t] = x_flat[e * T + t];
            extensional(*this, row, dfa);
        }

        // ── (b) indicateurs de travail : works[e][t] ⟺ x[e][t] ∈ work_set
        BoolVarArray works(*this, E * T, 0, 1);
        if (work_set.ranges() == 0) {
            // aucun symbole de travail : works toujours 0
            for (int i = 0; i < E * T; ++i)
                rel(*this, works[i], IRT_EQ, 0);
        } else {
            for (int e = 0; e < E; ++e)
                for (int t = 0; t < T; ++t)
                    dom(*this, x_flat[e * T + t], work_set, works[e * T + t]);
        }

        // ── (b') dotation y[t] = Σ_e works[e][t] ───────────────────────
        for (int t = 0; t < T; ++t) {
            BoolVarArgs col(E);
            for (int e = 0; e < E; ++e) col[e] = works[e * T + t];
            linear(*this, col, IRT_EQ, y[t]);
        }

        // ── (c) réification 2 niveaux + quantile ────────────────────────
        BoolVarArray zti(*this, N * T, 0, 1);
        for (int i = 0; i < N; ++i) {
            for (int t = 0; t < T; ++t) {
                int d = demand[i][t];
                if (d <= 0)
                    rel(*this, zti[i * T + t], IRT_EQ, 1);   // toujours couvert
                else if (d > E)
                    rel(*this, zti[i * T + t], IRT_EQ, 0);   // impossible
                else
                    rel(*this, y[t], IRT_GQ, d, Reify(zti[i * T + t]));
            }
            BoolVarArgs row_i(T);
            for (int t = 0; t < T; ++t) row_i[t] = zti[i * T + t];
            rel(*this, BOT_AND, row_i, z[i]);
        }
        linear(*this, z, IRT_GQ, min_feas);

        // ── (d) objectif : Σ_t sal[t] · y[t] = total_cost ──────────────
        IntArgs sal(T);
        for (int t = 0; t < T; ++t) sal[t] = sal_sc[t];
        linear(*this, sal, y, IRT_EQ, total_cost);

        // ── (e) brisure de symétrie lex (employés interchangeables) ─────
        if (break_symmetry) {
            for (int e = 0; e < E - 1; ++e) {
                IntVarArgs row_e(T), row_e1(T);
                for (int t = 0; t < T; ++t) {
                    row_e [t] = x_flat[ e      * T + t];
                    row_e1[t] = x_flat[(e + 1) * T + t];
                }
                rel(*this, row_e, IRT_LQ, row_e1);
            }
        }

        // ── Branchement (patron MKP) ─────────────────────────────────────
        branch(*this, x_flat, INT_VAR_SIZE_MIN(), INT_VAL_MIN());
        branch(*this, z,      BOOL_VAR_NONE(),    BOOL_VAL_MAX());
    }

    SSPSpace(SSPSpace& s) : IntMinimizeSpace(s) {
        x_flat    .update(*this, s.x_flat);
        y         .update(*this, s.y);
        z         .update(*this, s.z);
        total_cost.update(*this, s.total_cost);
    }
    Space*  copy()  override { return new SSPSpace(*this); }
    IntVar  cost()  const override { return total_cost; }
};

// ── Constructeurs d'objets Gecode partagés ────────────────────────────────────

static DFA build_dfa(int Q, int m, int q0,
                     const std::vector<std::vector<int>>& delta,
                     const std::vector<int>& accepting)
{
    std::vector<DFA::Transition> trans;
    trans.reserve(Q * m + 1);
    for (int q = 0; q < Q; ++q)
        for (int a = 0; a < m; ++a)
            if (delta[q][a] >= 0)
                trans.push_back({q, a, delta[q][a]});
    trans.push_back({-1, 0, 0});  // sentinelle obligatoire

    std::vector<int> finals = accepting;
    finals.push_back(-1);         // sentinelle obligatoire

    return DFA(q0, trans.data(), finals.data(), /*minimize=*/false);
}

static IntSet build_work_set(int m, const std::vector<int>& work_indicator)
{
    std::vector<int> ws;
    for (int a = 0; a < m; ++a)
        if (work_indicator[a]) ws.push_back(a);
    if (ws.empty()) return IntSet(IntSet::empty);
    return IntSet(ws.data(), static_cast<int>(ws.size()));
}

// ── solve_gecode_ssp ──────────────────────────────────────────────────────────

GecodeResult solve_gecode_ssp(
    int E, int T, int m, int Q, int N, double epsilon,
    const std::vector<int>& work_indicator,
    const Vec1d&            salary,
    const std::vector<std::vector<int>>& delta,
    int  q0,
    const std::vector<int>& accepting,
    const std::vector<std::vector<int>>& demand,
    double time_lim,
    bool   break_symmetry)
{
    auto t0 = Clock::now();

    int min_feas = static_cast<int>(std::ceil((1.0 - epsilon) * N));

    std::vector<int> sal_sc(T);
    int sal_ub = 0;
    for (int t = 0; t < T; ++t) {
        sal_sc[t]  = to_int(salary[t]);
        sal_ub    += E * std::max(0, sal_sc[t]);
    }

    DFA    dfa      = build_dfa(Q, m, q0, delta, accepting);
    IntSet work_set = build_work_set(m, work_indicator);

    SSPSpace* model = new SSPSpace(E, T, m, N, min_feas, dfa, work_set,
                                    sal_sc, sal_ub, demand, break_symmetry);

    Search::Options opts;
    Search::TimeStop* stop = nullptr;
    if (time_lim > 0.0) {
        stop      = new Search::TimeStop(static_cast<unsigned long long>(time_lim * 1000));
        opts.stop = stop;
    }

    BAB<SSPSpace> engine(model, opts);
    delete model;
    double cold_start_s = elapsed_s(t0);

    SSPSpace* best = nullptr;
    int64_t inc_impr = 0;
    while (SSPSpace* sol = engine.next()) { delete best; best = sol; ++inc_impr; }

    bool timed_out = stop && engine.stopped();
    auto stats = engine.statistics();
    delete stop;

    GecodeResult res;
    res.solve_time_s      = elapsed_s(t0);
    res.cold_start_time_s = cold_start_s;
    res.nodes        = static_cast<int64_t>(stats.node);
    res.fails        = static_cast<int64_t>(stats.fail);
    res.depth        = static_cast<int64_t>(stats.depth);
    res.incumbent_improvements = inc_impr;
    if (best) {
        res.objective = static_cast<double>(best->total_cost.val()) / SCALE;
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

// ════════════════════════════════════════════════════════════════════════════════
// Tests de régression SSP
// ════════════════════════════════════════════════════════════════════════════════

// Helper interne : résout et retourne le meilleur SSPSpace* (caller doit delete).
// Remplit *r_out si non-null (nodes, fails, feasible).
static SSPSpace* ssp_solve_raw(
    int E, int T, int m, int N, int min_feas,
    const DFA& dfa, const IntSet& work_set,
    const std::vector<int>& sal_sc, int sal_ub,
    const std::vector<std::vector<int>>& demand,
    bool break_symmetry,
    GecodeResult* r_out)
{
    SSPSpace* model = new SSPSpace(E, T, m, N, min_feas, dfa, work_set,
                                    sal_sc, sal_ub, demand, break_symmetry);
    BAB<SSPSpace> engine(model);
    delete model;

    SSPSpace* best = nullptr;
    while (SSPSpace* sol = engine.next()) { delete best; best = sol; }

    if (r_out) {
        auto st = engine.statistics();
        r_out->nodes    = static_cast<int64_t>(st.node);
        r_out->fails    = static_cast<int64_t>(st.fail);
        r_out->feasible = (best != nullptr);
        r_out->objective = best
            ? static_cast<double>(best->total_cost.val()) / SCALE
            : std::numeric_limits<double>::quiet_NaN();
    }
    return best;
}

bool run_ssp_tests()
{
    // ── Instance jouet ────────────────────────────────────────────────────────
    // E=3 employés, T=8 périodes, m=4 symboles, Q=3 états, N=20, ε=0.10
    // Alphabet : 0=W(travail), 1=P(préparation), 2=R(repos), 3=O(hors-service)
    // DFA : bloc de travail W⁺, suivi de repos R⁺, puis hors-service O*.
    //        État 0 = pré-service, 1 = en service, 2 = post-service.
    const int E = 3, T = 8, m = 4, Q = 3, N = 20;
    const double epsilon = 0.10;
    const int q0         = 0;
    const int min_feas   = static_cast<int>(std::ceil((1.0 - epsilon) * N));  // 18

    std::vector<std::vector<int>> delta(Q, std::vector<int>(m, -1));
    delta[0][0] = 1;  // état 0, W → en service
    delta[0][1] = 0;  // état 0, P → reste pré
    delta[0][3] = 0;  // état 0, O → reste pré
    delta[1][0] = 1;  // état 1, W → reste en service
    delta[1][2] = 2;  // état 1, R → repos
    delta[2][2] = 2;  // état 2, R → reste repos
    delta[2][3] = 2;  // état 2, O → reste post

    const std::vector<int>  accepting        = {0, 1, 2};
    const std::vector<int>  work_indicator   = {1, 0, 0, 0};  // seulement W=0
    const Vec1d             salary           = {3., 4., 5., 5., 4., 3., 2., 1.};

    std::vector<int> sal_sc(T);
    int sal_ub = 0;
    for (int t = 0; t < T; ++t) {
        sal_sc[t]  = to_int(salary[t]);
        sal_ub    += E * std::max(0, sal_sc[t]);
    }

    // Demande aléatoire déterministe ∈ {0,1}
    std::vector<std::vector<int>> demand(N, std::vector<int>(T, 0));
    {
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, 1);
        for (int i = 0; i < N; ++i)
            for (int t = 0; t < T; ++t)
                demand[i][t] = dist(rng);
    }

    DFA    dfa      = build_dfa(Q, m, q0, delta, accepting);
    IntSet work_set = build_work_set(m, work_indicator);

    bool all_pass = true;
    std::cout << "\n=== run_ssp_tests (E=" << E << " T=" << T
              << " N=" << N << " ε=" << epsilon << ") ===\n";

    // ── T1/T2/T3 : légalité DFA, couverture/quantile, dérivation y ───────────
    {
        GecodeResult r;
        SSPSpace* best = ssp_solve_raw(E, T, m, N, min_feas, dfa, work_set,
                                        sal_sc, sal_ub, demand, true, &r);
        bool t1 = false, t2 = false, t3 = false;

        if (!r.feasible) {
            std::cout << "[FAIL] T1/T2/T3 : aucune solution (instance infaisable ?)\n";
        } else {
            // T1 : rejouer le DFA sur chaque ligne x[e]
            t1 = true;
            for (int e = 0; e < E && t1; ++e) {
                int q = q0;
                for (int t = 0; t < T && t1; ++t) {
                    int a      = best->x_flat[e * T + t].val();
                    int q_next = (q >= 0 && q < Q) ? delta[q][a] : -1;
                    if (q_next < 0) { t1 = false; break; }
                    q = q_next;
                }
                if (t1 && std::find(accepting.begin(), accepting.end(), q) == accepting.end())
                    t1 = false;
            }

            // T2 : quantile Σ z[i] ≥ min_feas
            int covered = 0;
            for (int i = 0; i < N; ++i) covered += best->z[i].val();
            t2 = (covered >= min_feas);

            // T3 : y[t] == nombre d'employés actifs à t
            t3 = true;
            for (int t = 0; t < T && t3; ++t) {
                int cnt = 0;
                for (int e = 0; e < E; ++e)
                    if (work_indicator[best->x_flat[e * T + t].val()]) cnt++;
                if (best->y[t].val() != cnt) t3 = false;
            }

            delete best;
        }

        std::cout << "[" << (t1 ? "PASS" : "FAIL") << "] T1 Légalité DFA\n";
        std::cout << "[" << (t2 ? "PASS" : "FAIL") << "] T2 Couverture/quantile"
                  << "  (couverts=" << (t2 ? ">=" : "<") << min_feas << "/" << N << ")\n";
        std::cout << "[" << (t3 ? "PASS" : "FAIL") << "] T3 Dérivation y\n";
        all_pass = all_pass && t1 && t2 && t3;
    }

    // ── T4 : symétrie lex réduit (ou maintient) le nombre de nœuds ───────────
    {
        GecodeResult r_sym, r_nosym;
        SSPSpace* s1 = ssp_solve_raw(E, T, m, N, min_feas, dfa, work_set,
                                      sal_sc, sal_ub, demand, true,  &r_sym);
        SSPSpace* s2 = ssp_solve_raw(E, T, m, N, min_feas, dfa, work_set,
                                      sal_sc, sal_ub, demand, false, &r_nosym);
        delete s1; delete s2;

        // WARN plutôt que FAIL : peut ne pas être monotone sur toute petite instance
        bool t4 = r_sym.nodes <= r_nosym.nodes;
        std::cout << "[" << (t4 ? "PASS" : "WARN") << "] T4 Symétrie"
                  << "  nodes_sym=" << r_sym.nodes
                  << "  nodes_nosym=" << r_nosym.nodes << "\n";
        // Ne bloque pas all_pass (WARN = informatif)
    }

    // ── T5 : minimisation — demande plus élevée → coût optimal ≥ ─────────────
    {
        // Demande nulle → coût optimal = 0 (on peut ne faire travailler personne)
        std::vector<std::vector<int>> demand_zero(N, std::vector<int>(T, 0));

        // Demande élevée : 1 travailleur requis sur toutes les périodes centrales
        // pour tous les scénarios.  Le coût doit être > 0.
        std::vector<std::vector<int>> demand_high(N, std::vector<int>(T, 0));
        for (int i = 0; i < N; ++i)
            for (int t = 2; t <= 5; ++t)
                demand_high[i][t] = 1;

        GecodeResult r_low, r_high;
        SSPSpace* s_low  = ssp_solve_raw(E, T, m, N, min_feas, dfa, work_set,
                                          sal_sc, sal_ub, demand_zero, true, &r_low);
        SSPSpace* s_high = ssp_solve_raw(E, T, m, N, min_feas, dfa, work_set,
                                          sal_sc, sal_ub, demand_high, true, &r_high);
        delete s_low; delete s_high;

        bool t5 = r_low.feasible && r_high.feasible
                  && (r_high.objective >= r_low.objective - 1e-6);
        std::cout << "[" << (t5 ? "PASS" : "FAIL") << "] T5 Minimisation"
                  << "  obj_zero=" << r_low.objective
                  << "  obj_high=" << r_high.objective << "\n";
        all_pass = all_pass && t5;
    }

    std::cout << "=== " << (all_pass ? "TOUS OK" : "ECHEC") << " ===\n";
    return all_pass;
}
