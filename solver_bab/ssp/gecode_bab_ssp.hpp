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
#include "ssp_gnn_bound_backend.hpp"

#include <gecode/int.hh>
#include <gecode/minimodel.hh>
#include <gecode/search.hh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
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
// GNN   : un seul forward pass du GNN v1CLD entraîné (gnn_v1cld_native.hpp),
//         PAS d'itération/warm-start contrairement à V1CLD (borne one-shot,
//         même convention que le mode GNN de MKP, cf. gecode_bab.hpp mkp) —
//         partout dans l'arbre, y compris après le branchement complet de x
//         (zone jamais vue à l'entraînement, cf. discussion 2026-08-18 :
//         borne reste valide dans tous les cas — la preuve tient pour tout λ
//         — mais peut y être lâche).
// REF   : squelette nu (même mirroir que BoundMode::REF côté MKP, cf.
//         gecode_bab.hpp mkp) — même classe, même heuristique de branchement,
//         même moteur, même amorçage d'incumbent, mais more() ne reconstruit
//         AUCUN masque et ne calcule/poste AUCUNE borne. Sert de référence
//         pour isoler l'effet de la borne seule (écart REF vs GNN/V1CLD) de
//         celui du branchement/moteur (écart Gecode de base vs REF).
enum class SSPBoundMode { V1LD, V1CLD, GNN, REF };

// ── Heuristique de branchement du BoundBrancher ──────────────────────────────
// RATIO    : ordre data->branch_order (score demande/salaire), valeur W en
//            premier — pensé pour épouser le raisonnement glouton de la borne.
// SIZE_MIN : imite le brancher générique de Gecode base (INT_VAR_SIZE_MIN /
//            INT_VAL_MIN) — domaine le plus petit d'abord (ties : index
//            croissant e*T+t), valeur croissante en premier (0..3). Sert à
//            isoler l'effet de l'heuristique de celui de la borne : même
//            more()/bound_mode, mais recherche pilotée comme Gecode base.
enum class SSPBranchHeuristic { RATIO, SIZE_MIN };

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
    // salary_rounded[t] = sal_sc[t] / SCALE — EXPÉRIMENTAL (2026-07-09) : le
    // modèle CP (total_cost) est construit à partir de sal_sc (arrondi au
    // millième), mais more() passait jusqu'ici 'salary' brut (double, non
    // arrondi) aux oracles θ/ζ — un écart systématique (~2e-4/période dans les
    // instances observées, TOUJOURS du même signe ici) entre les deux peut
    // suffire à ce qu'une borne duale valide (au sens du coût réel non arrondi)
    // dépasse total_cost.max() (construit sur le coût arrondi) — test de
    // l'hypothèse "troncature façon MKP" en passant la version arrondie ici.
    Vec1d  salary_rounded;
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

    SSPBoundMode      bound_mode      = SSPBoundMode::V1LD;
    SSPBranchHeuristic branch_heuristic = SSPBranchHeuristic::SIZE_MIN;

    // Mode GNN uniquement (bound_mode == SSPBoundMode::GNN) — backend chargé
    // et possédé par l'appelant (cf. main_gecode.cpp côté MKP, même
    // convention : SSPSharedData ne construit/ne détruit jamais le modèle).
    // Requis (non nullptr) si bound_mode == GNN, ignoré sinon.
    SSPGNNBoundBackend* gnn_client = nullptr;
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

    mutable int64_t more_calls         = 0;
    mutable double  root_bound_time_s  = 0.0;
    // Valeur de la borne inférieure (coût salarial, même échelle que objective)
    // calculée au tout premier appel more() (racine, avant tout branchement).
    // NaN tant qu'aucun more() n'a été exécuté (ex: use_lagrangian_bound=false).
    mutable double  root_bound_value   = std::numeric_limits<double>::quiet_NaN();
    mutable double  t_bound_total_s    = 0.0;
    // Répartition du coût pour bound_mode==V1CLD uniquement (2026-08-20,
    // comparaison à la répartition GNN graph_build/forward/zeta_theta) :
    // t_v1cld_problem_build_s = construction de SSPCLDProblemBnB par nœud
    // (analogue de graph_build) ; t_v1cld_zeta_theta_s = temps cumulé dans
    // l'étape B (calcul θ/ζ) de solve_v1cld, sur tous les nœuds. Le "reste"
    // (mise à jour de λ -- analogue du forward GNN) se déduit par soustraction
    // à t_bound_total_s dans l'appelant, pas stocké séparément.
    mutable double  t_v1cld_problem_build_s = 0.0;
    mutable double  t_v1cld_zeta_theta_s    = 0.0;
    // Diagnostic 2026-08-20 : nombre total d'itérations (A)-(H) réellement
    // exécutées cumulé sur tous les more(), pour vérifier a posteriori si le
    // temps zeta_theta quasi nul mesuré est plausible (peu d'itérations au
    // total) ou suspect (beaucoup d'itérations pour un temps mesuré ~0).
    mutable int64_t v1cld_iters_total       = 0;
    // Nombre de fois où la borne duale a réellement resserré total_cost
    // (lb_int > total_cost.min() dans more()) — permet de distinguer "borne
    // correcte mais inutile sur cette instance" de "borne qui ne se déclenche
    // jamais" (signe possible d'un bug de calibration/signe/scale).
    mutable int64_t bound_tightenings  = 0;
    // Sous-ensemble de bound_tightenings où la borne est en plus strictement
    // supérieure au MAXIMUM courant de total_cost (lb_int > total_cost.max()) :
    // contradiction immédiate, directement imputable à la borne (pas à la
    // propagation CP native qui tournera plus tard). Équivalent SSP de
    // fails_from_bound dans mkp/gecode_bab.hpp (ub_int < obj.min() côté max).
    mutable int64_t fails_from_bound   = 0;

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
        salary_rounded.resize(T);
        for (int t = 0; t < T; ++t) {
            sal_sc[t] = static_cast<int>(std::llround(salary[t] * SCALE));
            sal_ub   += E * std::max(0, sal_sc[t]);
            salary_rounded[t] = sal_sc[t] / static_cast<double>(SCALE);
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

        // Ordre de branchement : RATIO suit data->branch_order (score demande/
        // salaire, employé-major) ; SIZE_MIN imite INT_VAR_SIZE_MIN de Gecode
        // base (domaine le plus petit d'abord, ties par index croissant).
        Choice* choice(Space& home) override {
            const auto* data = static_cast<SSPSpaceBnB&>(home).data;
            if (data->branch_heuristic == SSPBranchHeuristic::RATIO) {
                for (int idx : data->branch_order)
                    if (!vars[idx].assigned()) return new PosVal(*this, idx);
                GECODE_NEVER;
                return nullptr;
            }
            int best = -1, best_size = std::numeric_limits<int>::max();
            for (int i = 0; i < vars.size(); ++i) {
                if (vars[i].assigned()) continue;
                int sz = vars[i].size();
                if (sz < best_size) { best_size = sz; best = i; }
            }
            if (best < 0) GECODE_NEVER;
            return new PosVal(*this, best);
        }

        Choice* choice(const Space&, Archive& e) override {
            int pos; e >> pos;
            return new PosVal(*this, pos);
        }

        // RATIO : essaie toujours W en premier (mime le tri glouton par delta
        // de solve_zeta_masked). SIZE_MIN : valeur croissante en premier,
        // comme INT_VAL_MIN de Gecode base. use_lagrangian_bound=false : on
        // fixe quand même la variable (même arbre), mais on n'appelle jamais
        // more() — isole l'effet de la borne à heuristique et moteur de
        // recherche fixés.
        ExecStatus commit(Space& home, const Choice& c, unsigned int a) override {
            const PosVal& pv = static_cast<const PosVal&>(c);
            auto& space = static_cast<SSPSpaceBnB&>(home);
            static const int VALUE_ORDER_RATIO[SSP_BAB_SIGMA_SIZE] = {
                SSP_BAB_SIGMA_W, 0, 1, 2
            };
            static const int VALUE_ORDER_SIZE_MIN[SSP_BAB_SIGMA_SIZE] = {
                0, 1, 2, SSP_BAB_SIGMA_W
            };
            const int* value_order = (space.data->branch_heuristic == SSPBranchHeuristic::RATIO)
                                      ? VALUE_ORDER_RATIO : VALUE_ORDER_SIZE_MIN;
            ExecStatus es = me_failed(vars[pv.pos].eq(home, value_order[a]))
                            ? ES_FAILED : ES_OK;
            if (es == ES_OK && space.data->use_lagrangian_bound)
                space.more();
            return es;
        }

        void print(const Space& home, const Choice& c, unsigned int a,
                   std::ostream& os) const override {
            const PosVal& pv = static_cast<const PosVal&>(c);
            static const int VALUE_ORDER_RATIO[SSP_BAB_SIGMA_SIZE] = {
                SSP_BAB_SIGMA_W, 0, 1, 2
            };
            static const int VALUE_ORDER_SIZE_MIN[SSP_BAB_SIGMA_SIZE] = {
                0, 1, 2, SSP_BAB_SIGMA_W
            };
            const auto& data = static_cast<const SSPSpaceBnB&>(home).data;
            const int* value_order = (data->branch_heuristic == SSPBranchHeuristic::RATIO)
                                      ? VALUE_ORDER_RATIO : VALUE_ORDER_SIZE_MIN;
            os << "x_flat[" << pv.pos << "] = " << value_order[a];
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
        // Mode REF : squelette nu, cf. docstring SSPBoundMode::REF ci-dessus —
        // court-circuit AVANT le scan des variables (Étape 1), qui n'a
        // d'utilité que pour alimenter un calcul de borne (mirroir exact de
        // BoundMode::REF côté MKP, gecode_bab.hpp).
        if (data->bound_mode == SSPBoundMode::REF) {
            data->more_calls++;
            return;
        }

        const int E = data->E, T = data->T;

        Vec2i mask(E, Vec1i(T, -1));
        bool any_free = false;
        // curr_cost/free_count : sous-produits gratuits du même scan O(E·T) —
        // nécessaires seulement en mode GNN (build_graph_cpp attend curr_cost,
        // depth ; cf. discussion 2026-08-18 -- aucun coût supplémentaire par
        // rapport au scan déjà fait pour V1LD/V1CLD).
        double curr_cost   = 0.0;
        int    free_count  = 0;
        for (int e = 0; e < E; ++e)
            for (int t = 0; t < T; ++t) {
                const IntVar& v = x_flat[e * T + t];
                if (v.assigned()) {
                    mask[e][t] = v.val();
                    if (v.val() == SSP_BAB_SIGMA_W) curr_cost += data->salary[t];
                } else {
                    any_free = true;
                    ++free_count;
                }
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
                                data->salary_rounded, data->demand, data->p, data->epsilon,
                                E, mask);
            V1LDResult res = solve_v1ld(prob, T_iters, 0, false, remaining,
                                        data->rel_tol, data->tol_window, -1e18,
                                        lambda_v1ld);
            lambda_v1ld = std::move(res.lambda_star);
            dual_val    = res.v1LD_star;
        } else if (data->bound_mode == SSPBoundMode::V1CLD) {
            auto t_pb0 = Clock::now();
            SSPCLDProblemBnB prob(data->transitions, data->accepting, data->initial,
                                   data->salary_rounded, data->demand, data->p, data->epsilon,
                                   E, mask);
            data->t_v1cld_problem_build_s +=
                std::chrono::duration<double>(Clock::now() - t_pb0).count();
            V1CLDResult res = solve_v1cld(prob, T_iters, 0, false, remaining,
                                          data->rel_tol, data->tol_window, -1e18,
                                          lambda_v1cld);
            data->t_v1cld_zeta_theta_s += res.t_zeta_theta_s;
            data->v1cld_iters_total    += res.iterations_run;
            lambda_v1cld = std::move(res.lambda_star);
            dual_val     = res.v1CLD_star;
        } else {
            // Mode GNN : un seul forward pass, pas d'itération/warm-start (cf.
            // docstring SSPBoundMode::GNN). x_mask/z_mask reconstruits en
            // convention disque (4/2 = libre) attendue par
            // SSPGNNBoundBackend::query -- mask/z sont en convention -1 = libre.
            std::vector<int8_t> x_mask(E * T);
            for (int e = 0; e < E; ++e)
                for (int t = 0; t < T; ++t)
                    x_mask[e * T + t] = (mask[e][t] == -1)
                        ? static_cast<int8_t>(SSP_BAB_SIGMA_SIZE)
                        : static_cast<int8_t>(mask[e][t]);

            const int N = data->N;
            std::vector<int8_t> z_mask(N);
            int free_z_count = 0;
            for (int i = 0; i < N; ++i) {
                if (z[i].assigned()) {
                    z_mask[i] = static_cast<int8_t>(z[i].val());
                } else {
                    z_mask[i] = 2;
                    ++free_z_count;
                }
            }
            // N_z == 0 possible même si des x_flat restent libres : la
            // propagation CP (zti/BOT_AND) peut résoudre tous les z[i] avant
            // tout branchement explicite sur z -- l'implémentation concrète lève
            // dans ce cas (jamais entraîné sur N_z=0). Pas de borne postée
            // pour ce nœud, comme si use_lagrangian_bound=false ici.
            if (free_z_count == 0) return;

            const int depth = E * T - free_count;
            double cost_lower = data->gnn_client->query(x_mask, z_mask, curr_cost, depth);
            dual_val = -cost_lower;   // reconvertit vers la convention max(−salaire) partagée ci-dessous
        }

        double elapsed = std::chrono::duration<double>(Clock::now() - bound_t0).count();
        data->t_bound_total_s += elapsed;
        if (is_first) {
            data->root_bound_time_s = elapsed;
            data->root_bound_value  = -dual_val;  // borne au nœud racine, convention coût réel
        }
        lambda_warmed = true;

        // dual_val est en convention max(−salaire) → borne inf. sur le coût salarial réel.
        // floor (pas ceil) : une borne inférieure valide ne doit jamais être surestimée.
        const double lb_salary = -dual_val;
        const int    lb_int    = static_cast<int>(std::floor(lb_salary * SSPSharedData::SCALE));

        // DEBUG TEMPORAIRE — lb_int ne peut JAMAIS dépasser sal_ub (coût si tous
        // les employés travaillent toutes les périodes), quel que soit le nœud/masque :
        // c'est un plafond universel sur toute solution faisable, donc aussi sur
        // toute borne inférieure valide. Un dépassement ici est un certificat
        // direct d'invalidité (pas besoin de connaître la solution optimale).
        // Élargi (2026-07-09) : déclenche aussi sur lb_int > total_cost.max()
        // (= fails_from_bound, contradiction avec le domaine COURANT, rétréci
        // par l'incumbent Gecode) — pas seulement le plafond universel sal_ub.
        // Nécessaire pour capturer des runs (ex: v1LD, T_root/T_node petits)
        // où le sous-gradient diverge moins fort et ne dépasse jamais sal_ub,
        // mais où la borne est quand même invalide localement.
        const bool exceeds_sal_ub  = lb_int > data->sal_ub;
        const bool exceeds_cur_max = lb_int > total_cost.max();
        if (getenv("SSP_DEBUG_BOUND") && (exceeds_sal_ub || exceeds_cur_max)) {
            std::cerr << "*** BORNE INVALIDE ["
                      << (data->bound_mode == SSPBoundMode::V1LD ? "V1LD" : "V1CLD")
                      << (exceeds_sal_ub ? "/sal_ub" : "/cur_max")
                      << "] : lb_int=" << lb_int
                      << " > sal_ub=" << data->sal_ub
                      << " (total_cost.max()=" << total_cost.max() << ")"
                      << "  (more_calls=" << data->more_calls
                      << ", lb_salary=" << lb_salary << ") ***\n";
            const char* dump_path = getenv("SSP_DEBUG_DUMP");
            if (dump_path && !std::ifstream(dump_path).good()) {
                std::ofstream f(dump_path);
                f << "mask\n";
                for (int e = 0; e < E; ++e) {
                    for (int t = 0; t < T; ++t) f << mask[e][t] << " ";
                    f << "\n";
                }
                f << "lambda_mode " << (data->bound_mode == SSPBoundMode::V1LD ? "ld" : "cld") << "\n";
                if (data->bound_mode == SSPBoundMode::V1LD) {
                    f << "lambda_v1ld " << lambda_v1ld.size() << " " << (lambda_v1ld.empty() ? 0 : lambda_v1ld[0].size()) << "\n";
                    for (auto& row : lambda_v1ld) { for (double v : row) f << v << " "; f << "\n"; }
                } else {
                    f << "lambda_v1cld " << lambda_v1cld.size() << " "
                      << (lambda_v1cld.empty() ? 0 : lambda_v1cld[0].size()) << " "
                      << (lambda_v1cld.empty() || lambda_v1cld[0].empty() ? 0 : lambda_v1cld[0][0].size()) << "\n";
                    for (auto& mat : lambda_v1cld)
                        for (auto& row : mat) { for (double v : row) f << v << " "; f << "\n"; }
                }
            }
        }

        if (lb_int > total_cost.min()) {
            data->bound_tightenings++;
            if (lb_int > total_cost.max())
                data->fails_from_bound++;
            rel(*this, total_cost >= lb_int);
        }
    }
};

// ── Résultat ──────────────────────────────────────────────────────────────────

struct GecodeSSPBabResult {
    double      objective;
    bool        optimal;
    bool        feasible;
    double      solve_time_s;
    double      root_bound_time_s;
    double      root_bound_value;   // NaN si use_lagrangian_bound=false (pas de borne)
    double      t_bound_total_s;
    double      t_bab_pure_s;      // = solve_time_s - t_bound_total_s
    // Répartition du coût, mode V1CLD uniquement (0.0 sinon) -- cf.
    // SSPSharedData::t_v1cld_problem_build_s / t_v1cld_zeta_theta_s.
    double      t_v1cld_problem_build_s;
    double      t_v1cld_zeta_theta_s;
    int64_t     v1cld_iters_total;
    int64_t     nodes;
    int64_t     fails;
    int64_t     depth;
    int64_t     incumbent_improvements;
    int64_t     more_calls;
    int64_t     bound_tightenings;   // cf. SSPSharedData::bound_tightenings
    int64_t     fails_from_bound;    // cf. SSPSharedData::fails_from_bound
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
    bool         use_lagrangian_bound = true,
    SSPBranchHeuristic branch_heuristic = SSPBranchHeuristic::SIZE_MIN,
    // Requis (non nullptr) si bound_mode == SSPBoundMode::GNN — possédé par
    // l'appelant (cf. SSPSharedData::gnn_client), ignoré sinon.
    SSPGNNBoundBackend* gnn_client = nullptr);
