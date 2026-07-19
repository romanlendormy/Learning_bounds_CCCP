#pragma once
/*
 * kp_gecode_bab.hpp — B&B Gecode pour le CC-KP (m=1) avec borne V1LD à chaque nœud.
 *
 * Adaptation de solver_bab/mkp/gecode_bab.hpp au cas particulier m=1 (une seule
 * contrainte de capacité par scénario, pas m). Même architecture (modèle CP +
 * bornage Lagrangien dans more()), mêmes heuristique de branchement (RATIO) et
 * moteur DFS maison, MAIS :
 *   - solve_zeta est calculé via KPProblem (B&B 1D exact, réel, sans arrondi —
 *     cf. solver_generic/kp/kp_problem.hpp) au lieu de MKPProblemBnB (B&B
 *     multi-dim généraliste) : ~1300x plus rapide que le chemin CP-SAT
 *     générique à m=1 (cf. micro-benchmark de session), et strictement
 *     équivalent au cas m=1 de MKPProblemBnB/MKPCLDProblem (même algorithme,
 *     sans le surcoût des boucles génériques sur m).
 *   - Seuls BoundMode::V1LD et BoundMode::REF sont supportés : V1CLD est
 *     mathématiquement identique à V1LD à m=1 (aucune contrainte à décomposer,
 *     cf. docstring collect/collect_kp.py) — redondant. GNN/GNN_ROOT sont un
 *     fil de recherche séparé (borne apprise, dépendance LibTorch) — hors
 *     scope ici.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * FORMULATION CP DU CC-KP
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 *   max  Σ_k c_k x_k
 *   s.t. z_i = 1  →  Σ_k A_i[k] x_k ≤ b_i   ∀ i ∈ {1..N}
 *        Σ_i z_i ≥ ⌈(1−ε) N⌉
 *        x ∈ {0,1}^n,  z ∈ {0,1}^N
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * BORNE V1LD AUX NŒUDS DE L'ARBRE (mode V1LD ; REF = pas de borne, cf. more())
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Après chaque fixation de variable x_k = v par le brancher, more() est appelé :
 *   1. Profit fixé       : fp = Σ_{k fixé à 1} c_k
 *   2. Capacité réduite  : b̃_i = b_i − Σ_{k fixé à 1} A_i[k]
 *   3. Sous-problème résiduel sur les variables encore libres L
 *   4. v1LD* = solve_v1ld(KPProblem(sous-problème), T_iters)
 *   5. UB = fp + v1LD*  →  on poste  obj ≤ ⌈UB × SCALE⌉
 *
 * Cf. solver_bab/mkp/gecode_bab.hpp pour la discussion complète du mécanisme
 * de clonage Gecode et l'instrumentation (comptage more_calls/ub_tightened/
 * fails_from_bound, temps de borne vs temps B&B pur, coût du clonage).
 */

#include "../../solver_generic/v1ld_base.hpp"
#include "../../solver_generic/v1ld_solver.hpp"
#include "../../solver_generic/kp/kp_problem.hpp"

#include <gecode/int.hh>
#include <gecode/minimodel.hh>
#include <gecode/search.hh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

using namespace Gecode;

// ── Mode de calcul de la borne aux nœuds ─────────────────────────────────────
// V1LD : ζᵢ résolu exactement (B&B 1D, KPProblem)  → borne serrée
// REF  : squelette V1LD nu — même heuristique de branchement, même incumbent
//        glouton, même moteur DFS — mais more() ne calcule/poste aucune borne.
//        Sert de référence pour isoler l'effet de la borne du reste (branchement
//        + incumbent + moteur), cf. gecode_bab.hpp pour la discussion complète.
enum class KPBoundMode { V1LD, REF };

// ═════════════════════════════════════════════════════════════════════════════
// KPSharedData — données partagées entre tous les espaces Gecode
// ═════════════════════════════════════════════════════════════════════════════

struct KPSharedData {
    int    n;         // nombre de variables de décision (items)
    int    N;         // nombre de scénarios
    double epsilon;   // niveau de risque ε

    Vec1d  c;   // profits déterministes (n,)
    Vec1d  p;   // probabilités des scénarios (N,)
    Vec2d  A;   // coefficients de contrainte (N, n) — float
    Vec1d  b;   // capacités (N,) — float

    // ── Représentation entière pour Gecode (linear() n'accepte que des int) ──
    static constexpr int SCALE = 1000;
    std::vector<int>              c_int;   // c × SCALE
    std::vector<std::vector<int>> A_int;   // A × SCALE (arrondi)
    std::vector<int>              b_int;   // b × SCALE (floor, conservatif)
    int obj_max;

    KPBoundMode bound_mode = KPBoundMode::V1LD;
    int    T_root     = 200;
    int    T_node     = 30;
    double rel_tol    = 1e-4;
    int    tol_window = 30;

    bool   patience_mode      = false;
    double patience_tol       = 0.1;
    int    patience_max_iters = 500;

    int initial_lb_int = 0;

    mutable int    more_calls       = 0;
    mutable int    ub_tightened     = 0;
    mutable int    fails_from_bound = 0;
    mutable double best_open_ub     = 0.0;
    mutable double root_bound_time_s = 0.0;
    mutable double root_bound_value  = 0.0;
    mutable double t_bound_total_s   = 0.0;

    mutable int64_t clone_count     = 0;
    mutable double  t_clone_state_s = 0.0;
    mutable double  t_clone_total_s = 0.0;

    // Ordre de branchement RATIO : c_k / poids_surrogate décroissant
    std::vector<int> branch_order;

    KPSharedData(int n_, int N_, double eps, Vec1d c_, Vec1d p_, Vec2d A_, Vec1d b_,
                 int T_node_ = 30)
        : n(n_), N(N_), epsilon(eps),
          c(std::move(c_)), p(std::move(p_)), A(std::move(A_)), b(std::move(b_)),
          T_node(T_node_)
    {
        c_int.resize(n);
        for (int k = 0; k < n; ++k)
            c_int[k] = static_cast<int>(std::llround(c[k] * SCALE));

        A_int.assign(N, std::vector<int>(n));
        b_int.assign(N, 0);
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < n; ++k)
                A_int[i][k] = static_cast<int>(std::llround(A[i][k] * SCALE));
            b_int[i] = static_cast<int>(std::floor(b[i] * SCALE));
        }

        obj_max = 0;
        for (int k = 0; k < n; ++k) obj_max += std::max(0, c_int[k]);

        branch_order.resize(n);
        std::iota(branch_order.begin(), branch_order.end(), 0);
        std::sort(branch_order.begin(), branch_order.end(), [&](int a, int bk) {
            double wa = 0.0, wb = 0.0;
            for (int i = 0; i < N; ++i)
                if (b[i] > 1e-9) {
                    wa += p[i] * A[i][a]  / b[i];
                    wb += p[i] * A[i][bk] / b[i];
                }
            double ra = (wa > 1e-12) ? c[a]  / wa  : c[a]  * 1e12;
            double rb = (wb > 1e-12) ? c[bk] / wb  : c[bk] * 1e12;
            return ra > rb;
        });
    }
};

// ═════════════════════════════════════════════════════════════════════════════
// GecodeKPResult — statistiques de résolution retournées à l'appelant
// ═════════════════════════════════════════════════════════════════════════════

struct GecodeKPResult {
    double      objective;
    bool        optimal;
    bool        feasible;
    double      greedy_value;
    double      best_open_ub;
    double      solve_time_s;
    int64_t     nodes;
    int64_t     fails;
    int64_t     fails_from_bound;
    int64_t     fails_from_cp;
    int64_t     incumbent_improvements;
    int64_t     more_calls;
    int64_t     ub_tightened;
    int64_t     depth;
    double      root_bound_time_s;
    double      root_bound_value = 0.0;
    double      t_bound_total_s;
    double      t_bab_pure_s;
    int64_t     clone_count     = 0;
    double      t_clone_state_s = 0.0;
    double      t_clone_total_s = 0.0;
    std::string status;
};

// ═════════════════════════════════════════════════════════════════════════════
// KPSpaceV1LD — espace de recherche Gecode pour le CC-KP + borne V1LD
// ═════════════════════════════════════════════════════════════════════════════

class KPSpaceV1LD : public IntMaximizeSpace {
public:
    BoolVarArray x;    // x[k] ∈ {0,1} : item k sélectionné ou non
    BoolVarArray z;    // z[i] ∈ {0,1} : scénario i satisfait ou non
    IntVar       obj;  // objectif scalé = Σ_k c_int[k] * x[k]

    const KPSharedData* data;

    Vec2d lambda_v1ld;    // (N, n) — multiplicateurs Lagrangiens
    bool  lambda_warmed = false;

    Vec1d                b_res;       // (N,) capacité résiduelle
    std::vector<int8_t>  fixed_mask;
    double               fp_running = 0.0;

    // ── BoundBrancher — déclenche more() après chaque commit de x[k] ─────────
    class BoundBrancher : public Brancher {
    protected:
        ViewArray<Int::BoolView> vars;

        struct PosVal : public Choice {
            int pos, val;
            PosVal(const BoundBrancher& b, int p, int v)
                : Choice(b, 2), pos(p), val(v) {}
            void archive(Archive& e) const override {
                Choice::archive(e); e << pos << val;
            }
        };

    public:
        BoundBrancher(Home home, ViewArray<Int::BoolView>& v)
            : Brancher(home), vars(v) {}

        static void post(Home home, ViewArray<Int::BoolView>& v) {
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

        bool status(const Space& home) const override {
            for (int i = 0; i < vars.size(); ++i)
                if (!vars[i].assigned()) return true;
            return false;
        }

        Choice* choice(Space& home) override {
            const auto& order = static_cast<KPSpaceV1LD&>(home).data->branch_order;
            for (int k : order)
                if (!vars[k].assigned())
                    return new PosVal(*this, k, 1);
            GECODE_NEVER;
            return nullptr;
        }

        Choice* choice(const Space&, Archive& e) override {
            int pos, val; e >> pos >> val;
            return new PosVal(*this, pos, val);
        }

        ExecStatus commit(Space& home, const Choice& c, unsigned int a) override {
            const PosVal& pv = static_cast<const PosVal&>(c);
            int pos = pv.pos;
            int val = (a == 0) ? 1 : 0;

            ExecStatus es = me_failed(vars[pos].eq(home, val))
                            ? ES_FAILED : ES_OK;

            if (es == ES_OK)
                static_cast<KPSpaceV1LD&>(home).more();

            return es;
        }

        void print(const Space&, const Choice& c, unsigned int a,
                   std::ostream& os) const override {
            const PosVal& pv = static_cast<const PosVal&>(c);
            os << "x[" << pv.pos << "] = " << (a == 0 ? 1 : 0);
        }
    };

    // ── Constructeur racine ────────────────────────────────────────────────
    explicit KPSpaceV1LD(const KPSharedData* d)
        : x(*this, d->n, 0, 1),
          z(*this, d->N, 0, 1),
          obj(*this, 0, d->obj_max),
          data(d),
          lambda_v1ld(d->bound_mode == KPBoundMode::V1LD
                      ? Vec2d(d->N, Vec1d(d->n, 0.0)) : Vec2d{}),
          b_res(d->b),
          fixed_mask(d->n, 0)
    {
        int n = d->n, N = d->N;
        int min_feas = static_cast<int>(std::ceil((1.0 - d->epsilon) * N));

        IntVarArgs x_int(n);
        for (int k = 0; k < n; ++k) x_int[k] = channel(*this, x[k]);

        // z[i] = 1 → Σ_k A_i[k] x[k] ≤ b_i  (une seule contrainte par scénario)
        for (int i = 0; i < N; ++i) {
            IntArgs coefs(n);
            for (int k = 0; k < n; ++k) coefs[k] = d->A_int[i][k];
            linear(*this, coefs, x_int, IRT_LQ, d->b_int[i], Reify(z[i], RM_IMP));
        }

        linear(*this, z, IRT_GQ, min_feas);

        IntArgs c_args(n);
        for (int k = 0; k < n; ++k) c_args[k] = d->c_int[k];
        linear(*this, c_args, x_int, IRT_EQ, obj);

        if (d->initial_lb_int > 0)
            rel(*this, obj >= d->initial_lb_int);

        BoolVarArgs x_args(n);
        for (int k = 0; k < n; ++k) x_args[k] = x[k];
        ViewArray<Int::BoolView> xv(*this, x_args);
        BoundBrancher::post(*this, xv);

        branch(*this, z, BOOL_VAR_NONE(), BOOL_VAL_MAX());
    }

    // ── Constructeur de clonage ────────────────────────────────────────────
    KPSpaceV1LD(KPSpaceV1LD& s)
        : IntMaximizeSpace(s), data(s.data), lambda_warmed(s.lambda_warmed)
    {
        x.update(*this, s.x);
        z.update(*this, s.z);
        obj.update(*this, s.obj);

        using Clock = std::chrono::steady_clock;
        auto t_clone_start = Clock::now();
        lambda_v1ld = s.lambda_v1ld;
        b_res       = s.b_res;
        fixed_mask  = s.fixed_mask;
        fp_running  = s.fp_running;
        data->t_clone_state_s += std::chrono::duration<double>(
            Clock::now() - t_clone_start).count();
        data->clone_count++;
    }

    Space* copy() override {
        using Clock = std::chrono::steady_clock;
        auto t0 = Clock::now();
        Space* s = new KPSpaceV1LD(*this);
        data->t_clone_total_s += std::chrono::duration<double>(Clock::now() - t0).count();
        return s;
    }

    IntVar cost() const override { return obj; }

    // ── more() — calcul de la borne V1LD et propagation dans l'espace ───────
    void more() {
        if (data->bound_mode == KPBoundMode::REF) {
            data->more_calls++;
            return;
        }

        const int n = data->n;
        const int N = data->N;

        std::vector<int> free_vars;
        free_vars.reserve(n);

        for (int k = 0; k < n; ++k) {
            if (!x[k].assigned()) { free_vars.push_back(k); continue; }
            int val = x[k].val();

            if (fixed_mask[k] != 0) continue;  // déjà comptabilisé
            fixed_mask[k] = 1;
            if (val == 1) {
                for (int i = 0; i < N; ++i)
                    b_res[i] -= data->A[i][k];
                fp_running += data->c[k];
            }
        }

        data->more_calls++;
        if (free_vars.empty()) return;   // feuille : objectif exact, rien à borner

        const int n_free = static_cast<int>(free_vars.size());
        double fp = fp_running;

        using Clock = std::chrono::steady_clock;
        auto t_bound_start = Clock::now();

        Vec1d c_free(n_free);
        for (int kf = 0; kf < n_free; ++kf)
            c_free[kf] = data->c[free_vars[kf]];

        Vec2d A_free(N, Vec1d(n_free, 0.0));
        for (int i = 0; i < N; ++i)
            for (int kf = 0; kf < n_free; ++kf)
                A_free[i][kf] = data->A[i][free_vars[kf]];

        const Vec1d& b_adj = b_res;

        bool is_first = !lambda_warmed;
        int    T_iters;
        double node_tol;
        if (is_first) {
            T_iters  = data->T_root;
            node_tol = data->rel_tol;
        } else if (data->patience_mode) {
            T_iters  = data->patience_max_iters;
            node_tol = data->patience_tol;
        } else {
            T_iters  = data->T_node;
            node_tol = data->rel_tol;
        }

        Vec2d lambda_init(N, Vec1d(n_free, 0.0));
        if (!is_first)
            for (int i = 0; i < N; ++i)
                for (int kf = 0; kf < n_free; ++kf)
                    lambda_init[i][kf] = lambda_v1ld[i][free_vars[kf]];

        KPProblem prob(c_free, data->p, A_free, b_adj, data->epsilon);
        V1LDResult res = solve_v1ld(prob, T_iters, 0, false, -1.0,
                                    node_tol, data->tol_window, -1e18,
                                    lambda_init);

        for (int i = 0; i < N; ++i)
            for (int kf = 0; kf < n_free; ++kf)
                lambda_v1ld[i][free_vars[kf]] = res.lambda_star[i][kf];
        double ub = fp + res.v1LD_star;

        double elapsed_bound = std::chrono::duration<double>(
            Clock::now() - t_bound_start).count();
        data->t_bound_total_s += elapsed_bound;
        if (is_first) {
            data->root_bound_time_s = elapsed_bound;
            data->root_bound_value  = ub;
        }

        lambda_warmed = true;
        if (ub > data->best_open_ub) data->best_open_ub = ub;
        int ub_int = static_cast<int>(std::ceil(ub * data->SCALE));

        if (ub_int < obj.max()) {
            data->ub_tightened++;
            if (ub_int < obj.min()) data->fails_from_bound++;
            rel(*this, obj <= ub_int);
        }
    }
};

// ═════════════════════════════════════════════════════════════════════════════
// greedy_kp — solution gloutonne faisable pour le CC-KP
// ═════════════════════════════════════════════════════════════════════════════

std::pair<double, std::vector<int8_t>> greedy_kp(
    const Vec1d& c, const Vec1d& p, const Vec2d& A, const Vec1d& b, double epsilon);

// ═════════════════════════════════════════════════════════════════════════════
// solve_gecode_bab_kp — point d'entrée public
// ═════════════════════════════════════════════════════════════════════════════

GecodeKPResult solve_gecode_bab_kp(
    int n, int N, double epsilon,
    const Vec1d& c, const Vec1d& p,
    const Vec2d& A, const Vec1d& b,
    double      time_lim      = 300.0,
    int         T_root        = 200,
    int         T_node        = 30,
    KPBoundMode bound_mode    = KPBoundMode::V1LD,
    bool        patience_mode = false,
    double      patience_tol  = 0.1
);
