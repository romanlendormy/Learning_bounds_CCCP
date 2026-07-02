#pragma once
/*
 * gecode_bab_ssp.hpp — B&B Gecode pour le CCSSP avec borne v1LD/v1CLD à chaque nœud.
 *
 * Même patron que solver_bab/mkp/gecode_bab.hpp (MKPSpaceV1LD), adapté au SSP —
 * y compris pour le moteur de recherche : comme pour MKP, la résolution est
 * pilotée par Gecode::BAB<> natif (pas de DFS manuelle — une première version
 * avait introduit une DFS maison façon "pile status()/choice()/commit()", mais
 * elle élaguait plus agressivement que Gecode natif sur une borne stockée sans
 * rappeler status(), ce qui la rendait non fiable à plus grande échelle ; on
 * s'en tient au moteur Gecode standard, plus robuste).
 *
 * Différence structurelle clé avec MKP (voir plan / discussion) : pour MKP, le
 * sous-problème résiduel d'un nœud se construit en RÉTRÉCISSANT les tableaux
 * (items fixés retirés). Pour SSP ça ne marche pas — l'automate d'un employé
 * porte sur la séquence complète des T périodes. On MASQUE plutôt :
 * SSPProblemBnB / SSPCLDProblemBnB (solver_generic/ssp) acceptent un masque
 * (E×T, -1 = libre, sinon symbole imposé) qui restreint exactement le domaine
 * de maximisation de θ/ζ au sous-polytope du nœud courant — la borne duale
 * reste valide sans terme correctif séparé (cf. commentaires dans ssp_problem.hpp).
 *
 * Modèle CP (x_flat, extensional par employé, chance-constraint AND-based) repris
 * tel quel de gecode_base/gecode_solver.cpp::SSPSpace — seule différence : un
 * BoundBrancher remplace le branchement générique sur x_flat, avec notre ordre
 * ratio demande/salaire (SSPSharedData::branch_order), pour appeler more()
 * après chaque fixation.
 *
 * Convention de signe : SSPProblemBnB/SSPCLDProblemBnB maximisent cost^T x avec
 * cost[e,t,W] = −salary[t] (comme SSPShiftProblem/SSPCLDProblem). Le dual retourné
 * (v1LD_star / v1CLD_star) est donc en convention max(−salaire) ; la borne inférieure
 * sur le coût salarial réel (à poster sur total_cost, qui MINIMISE) est son opposé.
 */

#include "../../solver_generic/v1ld_base.hpp"
#include "../../solver_generic/v1ld_solver.hpp"
#include "../../solver_generic/v1cld_base.hpp"
#include "../../solver_generic/v1cld_solver.hpp"
#include "../../solver_generic/ssp/ssp_problem_bnb.hpp"
#include "../../solver_generic/ssp/ssp_cld_problem_bnb.hpp"

#include <gecode/int.hh>
#include <gecode/minimodel.hh>
#include <gecode/search.hh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

using namespace Gecode;

// Alphabet Σ = {O=0, P=1, R=2, W=3} — identique à SSPShiftProblem/SSPCLDProblem.
static constexpr int SSP_BAB_SIGMA_SIZE = 4;
static constexpr int SSP_BAB_SIGMA_W    = 3;

// ── Mode de calcul de la borne aux nœuds (même sémantique que pour MKP) ──────
// V1LD  : ζ_i exact via CP-SAT à chaque more() → borne serrée, TRÈS lent (NP-difficile).
//         Praticable uniquement sur de petites instances (E, T, N faibles).
// V1CLD : ζ_{i,j} glouton O(E·T·Q) → borne plus lâche, rapide.
enum class SSPBoundMode { V1LD, V1CLD };

// ── SSPSharedData — données partagées, construites une seule fois à la racine ─

struct SSPSharedData {
    static constexpr int SCALE = 1000;

    int    E, T, N;
    double epsilon;
    std::vector<std::tuple<int,int,int>> transitions;
    std::vector<int> accepting;
    int    initial;
    Vec1d  salary;
    Vec2i  demand;
    Vec1d  p;
    int    Q;

    DFA    dfa;
    IntSet work_set;

    std::vector<int> sal_sc;   // salary × SCALE
    int    sal_ub  = 0;
    int    min_feas = 0;

    // Ordre de branchement (calculé une fois à la construction) — analogue au
    // ratio profit/poids de MKP (branch_order dans mkp/gecode_bab.hpp) : les
    // employés étant interchangeables (seule la période t distingue le coût et
    // la demande), le score ne dépend que de t, pas de e.
    //   score[t] = demande_attendue[t] / salaire[t]
    // Périodes à fort score = beaucoup de demande pour un coût faible : les
    // décider en premier (et essayer W en premier, cf. BoundBrancher::commit)
    // mime le raisonnement glouton "satisfaire la couverture la moins chère
    // d'abord" qui sous-tend déjà le tri par delta de solve_zeta_masked.
    // Employé-major (toutes les périodes de e décidées avant e+1) pour
    // s'accorder avec le brise-symétrie lex (x[e] ≤lex x[e+1]).
    std::vector<int> branch_order;   // indices e*T+t, taille E*T

    SSPBoundMode bound_mode  = SSPBoundMode::V1LD;
    int    T_root     = 200;
    int    T_node     = 30;
    double rel_tol    = 1e-4;
    int    tol_window = 30;

    // Ablation : faut-il calculer/poster la borne Lagrangienne dans more()
    // (sinon on garde le même arbre/branchement — toujours l'ordre ratio —
    // mais sans jamais appeler solve_v1ld/solve_v1cld). Isole l'effet de la
    // borne, à heuristique et moteur de recherche fixés.
    bool use_lagrangian_bound = true;

    // Budget global (utilisé par more() pour ne pas dépasser le temps imparti
    // au B&B avec un seul appel solve_v1ld/solve_v1cld trop long — cf. note
    // dans more() : le contrôle interne du sous-gradient ne vérifie le temps
    // que tous les 50 itérations, donc pour T_root/T_node < 50 ce garde-fou
    // reste imparfait, mais il borne au moins les gros T_root sur grande N).
    double time_lim = -1.0;   // -1 = illimité
    std::chrono::steady_clock::time_point t_start;

    mutable int64_t more_calls        = 0;
    mutable double  root_bound_time_s = 0.0;
    mutable double  t_bound_total_s   = 0.0;

    static int compute_Q(int initial_,
                          const std::vector<std::tuple<int,int,int>>& trans,
                          const std::vector<int>& acc)
    {
        int Q = initial_ + 1;
        for (auto& [s, sym, d] : trans) Q = std::max(Q, std::max(s, d) + 1);
        for (int q : acc) Q = std::max(Q, q + 1);
        return Q;
    }

    static DFA build_dfa(int Q, int q0,
                          const std::vector<std::tuple<int,int,int>>& trans,
                          const std::vector<int>& acc)
    {
        std::vector<DFA::Transition> tr;
        tr.reserve(trans.size() + 1);
        for (auto& [s, sym, d] : trans) tr.push_back({s, sym, d});
        tr.push_back({-1, 0, 0});  // sentinelle obligatoire

        std::vector<int> finals = acc;
        finals.push_back(-1);      // sentinelle obligatoire
        (void)Q;
        return DFA(q0, tr.data(), finals.data(), /*minimize=*/false);
    }

    static IntSet build_work_set()
    {
        static const int ws[] = { SSP_BAB_SIGMA_W };
        return IntSet(ws, 1);
    }

    SSPSharedData(int E_, int T_, int N_, double eps,
                  std::vector<std::tuple<int,int,int>> trans,
                  std::vector<int> acc, int init,
                  Vec1d sal, Vec2i dem, Vec1d p_)
        : E(E_), T(T_), N(N_), epsilon(eps),
          transitions(std::move(trans)), accepting(std::move(acc)), initial(init),
          salary(std::move(sal)), demand(std::move(dem)), p(std::move(p_)),
          Q(compute_Q(initial, transitions, accepting)),
          dfa(build_dfa(Q, initial, transitions, accepting)),
          work_set(build_work_set())
    {
        sal_sc.resize(T);
        for (int t = 0; t < T; ++t) {
            sal_sc[t] = static_cast<int>(std::llround(salary[t] * SCALE));
            sal_ub   += E * std::max(0, sal_sc[t]);
        }
        min_feas = static_cast<int>(std::ceil((1.0 - epsilon) * N));

        // ── Score par période et ordre de branchement ─────────────────────────
        Vec1d expected_demand(T, 0.0);
        for (int i = 0; i < N; ++i)
            for (int t = 0; t < T; ++t)
                expected_demand[t] += p[i] * demand[i][t];

        std::vector<int> t_order(T);
        std::iota(t_order.begin(), t_order.end(), 0);
        std::sort(t_order.begin(), t_order.end(), [&](int a, int b) {
            double score_a = expected_demand[a] / std::max(1e-9, salary[a]);
            double score_b = expected_demand[b] / std::max(1e-9, salary[b]);
            return score_a > score_b;
        });

        branch_order.resize(E * T);
        int k = 0;
        for (int e = 0; e < E; ++e)
            for (int t : t_order)
                branch_order[k++] = e * T + t;
    }
};

// ── SSPSpaceBnB — espace de recherche Gecode pour le CCSSP + borne v1LD/v1CLD ─

class SSPSpaceBnB : public IntMinimizeSpace {
public:
    IntVarArray  x_flat;      // E×T, symbole ∈ {0..3}
    IntVarArray  y;           // T, dotation y[t] = Σ_e works[e][t]
    BoolVarArray z;           // N, z[i]=1 ↔ scénario i entièrement couvert
    IntVar       total_cost;  // objectif scalé (à MINIMISER)

    const SSPSharedData* data;

    // Multiplicateurs Lagrangiens — taille fixe (ne rétrécit jamais, contrairement
    // à MKP) puisqu'on masque plutôt que de rétrécir l'instance résiduelle.
    Vec2d lambda_v1ld;    // (N, E*T*4)   — mode V1LD
    Vec3d lambda_v1cld;   // (N, T, E*T*4) — mode V1CLD  (m = T contraintes)
    bool  lambda_warmed = false;

    // ── BoundBrancher — brancher custom sur x_flat, déclenche more() ─────────
    class BoundBrancher : public Brancher {
    protected:
        ViewArray<Int::IntView> vars;

        struct PosVal : public Choice {
            int pos;
            PosVal(const BoundBrancher& b, int p)
                : Choice(b, SSP_BAB_SIGMA_SIZE), pos(p) {}
            void archive(Archive& e) const override { Choice::archive(e); e << pos; }
        };

    public:
        BoundBrancher(Home home, ViewArray<Int::IntView>& v)
            : Brancher(home), vars(v) {}

        static void post(Home home, ViewArray<Int::IntView>& v) {
            new (home) BoundBrancher(home, v);
        }

        size_t dispose(Space& home) override {
            Brancher::dispose(home); return sizeof(*this);
        }

        BoundBrancher(Space& home, BoundBrancher& b)
            : Brancher(home, b) { vars.update(home, b.vars); }

        Brancher* copy(Space& home) override {
            return new (home) BoundBrancher(home, *this);
        }

        bool status(const Space&) const override {
            for (int i = 0; i < vars.size(); ++i)
                if (!vars[i].assigned()) return true;
            return false;
        }

        // Ordre de branchement : data->branch_order (score demande/salaire par
        // période, employé-major — cf. SSPSharedData).
        Choice* choice(Space& home) override {
            const auto& order = static_cast<SSPSpaceBnB&>(home).data->branch_order;
            for (int idx : order)
                if (!vars[idx].assigned()) return new PosVal(*this, idx);
            GECODE_NEVER;
            return nullptr;
        }

        Choice* choice(const Space&, Archive& e) override {
            int pos; e >> pos;
            return new PosVal(*this, pos);
        }

        // Essaie toujours W en premier (mime le tri glouton par delta de
        // solve_zeta_masked). use_lagrangian_bound=false : on fixe quand même
        // la variable (même arbre), mais on n'appelle jamais more() — isole
        // l'effet de la borne à heuristique et moteur de recherche fixés.
        ExecStatus commit(Space& home, const Choice& c, unsigned int a) override {
            const PosVal& pv = static_cast<const PosVal&>(c);
            auto& space = static_cast<SSPSpaceBnB&>(home);
            static const int VALUE_ORDER[SSP_BAB_SIGMA_SIZE] = {
                SSP_BAB_SIGMA_W, 0, 1, 2
            };
            ExecStatus es = me_failed(vars[pv.pos].eq(home, VALUE_ORDER[a]))
                            ? ES_FAILED : ES_OK;
            if (es == ES_OK && space.data->use_lagrangian_bound)
                space.more();
            return es;
        }

        void print(const Space&, const Choice& c, unsigned int a,
                   std::ostream& os) const override {
            const PosVal& pv = static_cast<const PosVal&>(c);
            static const int VALUE_ORDER[SSP_BAB_SIGMA_SIZE] = {
                SSP_BAB_SIGMA_W, 0, 1, 2
            };
            os << "x_flat[" << pv.pos << "] = " << VALUE_ORDER[a];
        }
    };

    // ── Constructeur racine ───────────────────────────────────────────────────
    // Modèle CP repris de gecode_base/gecode_solver.cpp::SSPSpace.
    explicit SSPSpaceBnB(const SSPSharedData* d)
        : x_flat    (*this, d->E * d->T, 0, SSP_BAB_SIGMA_SIZE - 1),
          y         (*this, d->T, 0, d->E),
          z         (*this, d->N, 0, 1),
          total_cost(*this, 0, d->sal_ub),
          data(d),
          lambda_v1ld (d->bound_mode == SSPBoundMode::V1LD
                       ? Vec2d(d->N, Vec1d(d->E * d->T * SSP_BAB_SIGMA_SIZE, 0.0)) : Vec2d{}),
          lambda_v1cld(d->bound_mode == SSPBoundMode::V1CLD
                       ? Vec3d(d->N, Vec2d(d->T, Vec1d(d->E * d->T * SSP_BAB_SIGMA_SIZE, 0.0)))
                       : Vec3d{})
    {
        const int E = d->E, T = d->T, N = d->N;

        for (int e = 0; e < E; ++e) {
            IntVarArgs row(T);
            for (int t = 0; t < T; ++t) row[t] = x_flat[e * T + t];
            extensional(*this, row, d->dfa);
        }

        BoolVarArray works(*this, E * T, 0, 1);
        for (int e = 0; e < E; ++e)
            for (int t = 0; t < T; ++t)
                dom(*this, x_flat[e * T + t], d->work_set, works[e * T + t]);

        for (int t = 0; t < T; ++t) {
            BoolVarArgs col(E);
            for (int e = 0; e < E; ++e) col[e] = works[e * T + t];
            linear(*this, col, IRT_EQ, y[t]);
        }

        BoolVarArray zti(*this, N * T, 0, 1);
        for (int i = 0; i < N; ++i) {
            for (int t = 0; t < T; ++t) {
                int dmd = d->demand[i][t];
                if (dmd <= 0)      rel(*this, zti[i * T + t], IRT_EQ, 1);
                else if (dmd > E)  rel(*this, zti[i * T + t], IRT_EQ, 0);
                else               rel(*this, y[t], IRT_GQ, dmd, Reify(zti[i * T + t]));
            }
            BoolVarArgs row_i(T);
            for (int t = 0; t < T; ++t) row_i[t] = zti[i * T + t];
            rel(*this, BOT_AND, row_i, z[i]);
        }
        linear(*this, z, IRT_GQ, d->min_feas);

        IntArgs sal(T);
        for (int t = 0; t < T; ++t) sal[t] = d->sal_sc[t];
        linear(*this, sal, y, IRT_EQ, total_cost);

        // Brise-symétrie lex (employés interchangeables) — repris de
        // gecode_base/gecode_solver.cpp::SSPSpace. Sans ça, le B&B explore des
        // sous-arbres équivalents par permutation d'employés — c'était la
        // principale cause du nombre de nœuds plus élevé que Gecode base observé
        // avant cet ajout (Gecode base l'a via break_symmetry=true).
        for (int e = 0; e < E - 1; ++e) {
            IntVarArgs row_e(T), row_e1(T);
            for (int t = 0; t < T; ++t) {
                row_e [t] = x_flat[ e      * T + t];
                row_e1[t] = x_flat[(e + 1) * T + t];
            }
            rel(*this, row_e, IRT_LQ, row_e1);
        }

        IntVarArgs x_args(E * T);
        for (int k = 0; k < E * T; ++k) x_args[k] = x_flat[k];
        ViewArray<Int::IntView> xv(*this, x_args);
        BoundBrancher::post(*this, xv);

        branch(*this, z, BOOL_VAR_NONE(), BOOL_VAL_MAX());
    }

    // ── Constructeur de clonage ────────────────────────────────────────────────
    SSPSpaceBnB(SSPSpaceBnB& s)
        : IntMinimizeSpace(s), data(s.data),
          lambda_v1ld(s.lambda_v1ld),
          lambda_v1cld(s.lambda_v1cld),
          lambda_warmed(s.lambda_warmed)
    {
        x_flat    .update(*this, s.x_flat);
        y         .update(*this, s.y);
        z         .update(*this, s.z);
        total_cost.update(*this, s.total_cost);
    }

    Space*  copy() override { return new SSPSpaceBnB(*this); }
    IntVar  cost() const override { return total_cost; }

    // ── more() — borne v1LD/v1CLD sur le sous-problème résiduel (masqué) ──────
    void more() {
        const int E = data->E, T = data->T;

        Vec2i mask(E, Vec1i(T, -1));
        bool any_free = false;
        for (int e = 0; e < E; ++e)
            for (int t = 0; t < T; ++t) {
                const IntVar& v = x_flat[e * T + t];
                if (v.assigned()) mask[e][t] = v.val();
                else              any_free   = true;
            }

        data->more_calls++;
        if (!any_free) return;  // feuille : total_cost déjà exact

        const bool is_first = !lambda_warmed;
        const int  T_iters  = is_first ? data->T_root : data->T_node;

        using Clock = std::chrono::steady_clock;
        auto bound_t0 = Clock::now();

        // Budget restant à passer à solve_v1ld/solve_v1cld : borne le pire cas
        // d'un more() individuel au temps global restant. Garde-fou imparfait
        // (solve_v1ld/v1cld ne revérifient le temps que tous les 50 itérations,
        // donc pour T_root/T_node < 50 le contrôle n'agit qu'entre nœuds, pas
        // en cours d'appel) — d'où l'intérêt de garder T_root/T_node petits en
        // mode V1LD (ζ = CP-SAT à chaque itération, déjà coûteux en soi).
        double remaining = -1.0;
        if (data->time_lim > 0.0) {
            double elapsed_global =
                std::chrono::duration<double>(bound_t0 - data->t_start).count();
            remaining = std::max(0.01, data->time_lim - elapsed_global);
        }

        double dual_val;  // convention max(−salaire)

        if (data->bound_mode == SSPBoundMode::V1LD) {
            SSPProblemBnB prob(data->transitions, data->accepting, data->initial,
                                data->salary, data->demand, data->p, data->epsilon,
                                E, mask);
            V1LDResult res = solve_v1ld(prob, T_iters, 0, false, remaining,
                                        data->rel_tol, data->tol_window, -1e18,
                                        lambda_v1ld);
            lambda_v1ld = std::move(res.lambda_star);
            dual_val    = res.v1LD_star;
        } else {
            SSPCLDProblemBnB prob(data->transitions, data->accepting, data->initial,
                                   data->salary, data->demand, data->p, data->epsilon,
                                   E, mask);
            V1CLDResult res = solve_v1cld(prob, T_iters, 0, false, remaining,
                                          data->rel_tol, data->tol_window, -1e18,
                                          lambda_v1cld);
            lambda_v1cld = std::move(res.lambda_star);
            dual_val     = res.v1CLD_star;
        }

        double elapsed = std::chrono::duration<double>(Clock::now() - bound_t0).count();
        data->t_bound_total_s += elapsed;
        if (is_first) data->root_bound_time_s = elapsed;
        lambda_warmed = true;

        // dual_val est en convention max(−salaire) → borne inf. sur le coût salarial réel.
        // floor (pas ceil) : une borne inférieure valide ne doit jamais être surestimée.
        const double lb_salary = -dual_val;
        const int    lb_int    = static_cast<int>(std::floor(lb_salary * SSPSharedData::SCALE));

        if (lb_int > total_cost.min())
            rel(*this, total_cost >= lb_int);
    }
};

// ── Résultat ──────────────────────────────────────────────────────────────────

struct GecodeSSPBabResult {
    double      objective;
    bool        optimal;
    bool        feasible;
    double      solve_time_s;
    double      root_bound_time_s;
    double      t_bound_total_s;
    double      t_bab_pure_s;      // = solve_time_s - t_bound_total_s
    int64_t     nodes;
    int64_t     fails;
    int64_t     depth;
    int64_t     incumbent_improvements;
    int64_t     more_calls;
    std::string status;
};

// ── Point d'entrée public ──────────────────────────────────────────────────────
//
// Piloté par Gecode::BAB<> natif (comme solve_gecode_mkp / solve_gecode_bab_mkp
// pour MKP). Deux usages typiques :
//   - use_lagrangian_bound=false : le modèle CP + notre heuristique ratio,
//     SANS borne Lagrangienne — sert de référence "base" (comme
//     bab_mkp_gecode_base pour MKP, mais avec notre heuristique plutôt que
//     celle générique de Gecode, pour comparer à heuristique fixée).
//   - use_lagrangian_bound=true, bound_mode=V1LD (défaut, comme pour MKP) :
//     le vrai B&B avec borne Lagrangienne à chaque nœud.
GecodeSSPBabResult solve_gecode_bab_ssp(
    int E, int T, int N, double epsilon,
    const std::vector<std::tuple<int,int,int>>& transitions,
    const std::vector<int>& accepting, int initial,
    const Vec1d& salary, const Vec2i& demand, Vec1d p,
    double       time_lim             = 300.0,
    int          T_root               = 200,
    int          T_node               = 30,
    SSPBoundMode bound_mode           = SSPBoundMode::V1LD,
    double       rel_tol              = 1e-4,
    int          tol_window           = 30,
    bool         use_lagrangian_bound = true);
