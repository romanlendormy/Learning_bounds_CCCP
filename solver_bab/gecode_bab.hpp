#pragma once
/*
 * gecode_bab_v1ld.hpp — B&B Gecode pour le CCMKP avec borne V1LD à chaque nœud.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * ARCHITECTURE GÉNÉRALE
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Ce fichier combine deux mécanismes :
 *
 *  1. Un modèle CP Gecode (MKPSpaceV1LD, hérite de IntMaximizeSpace) :
 *     - encode le CCMKP comme un programme en variables entières/booléennes
 *     - Gecode pilote l'arbre B&B, gère la propagation et le clonage des espaces
 *
 *  2. Un bornage Lagrangien (méthode more()) :
 *     - appelé après chaque décision de branchement
 *     - calcule v1LD sur le sous-problème résiduel (variables encore libres)
 *     - poste la contrainte  obj ≤ ⌈UB⌉  dans l'espace courant
 *     - Gecode utilise cette borne pour élaguer les branches non-prometteuses
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * FORMULATION CP DU CCMKP
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 *   max  Σ_k c_k x_k
 *   s.t. z_i = 1  →  Σ_k A_i[j][k] x_k ≤ b_i[j]   ∀ i ∈ {1..N}, j ∈ {1..m}
 *        Σ_i z_i ≥ ⌈(1−ε) N⌉
 *        x ∈ {0,1}^n,  z ∈ {0,1}^N
 *
 * La contrainte de chance est reformulée via N variables booléennes z_i :
 *   z_i = 1  ↔  le scénario i est "satisfait" (toutes ses contraintes sont tenues)
 *   Σ z_i ≥ ⌈(1−ε)N⌉  impose qu'au moins (1−ε) fraction des scénarios sont satisfaits.
 *
 * La réification  z_i = 1 → A_i x ≤ b_i  est encodée avec RM_IMP (implication) :
 * si z_i = 1 alors les contraintes capacitaires du scénario i doivent être tenues.
 * Ceci évite le recours à une constante Big-M et donne une propagation CP exacte.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * BORNE V1LD AUX NŒUDS DE L'ARBRE
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Après chaque fixation de variable x_k = v par le brancher, more() est appelé :
 *
 *   Soit F₁ = {k : x_k fixé à 1},  F₀ = {k : x_k fixé à 0},  L = {k : x_k libre}.
 *
 *   1. Profit fixé       : fp = Σ_{k ∈ F₁} c_k
 *   2. Capacité réduite  : b̃_i[j] = b_i[j] − Σ_{k ∈ F₁} A_i[j][k]
 *   3. Sous-problème résiduel sur L (avec b̃_i, A restreinte aux colonnes L)
 *   4. v1LD* = solve_v1ld(sous-problème, T_node itérations)
 *   5. UB = fp + v1LD*
 *   6. On poste  obj ≤ ⌈UB × SCALE⌉  (SCALE=1000 : conversion float→int Gecode)
 *
 * Si UB ≤ meilleure solution connue (incumbent), Gecode détecte la contradiction
 * sur obj et élaguer automatiquement la branche.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * MÉCANISME DE CLONAGE GECODE
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Gecode clone l'espace courant à chaque branchement. Pour éviter de dupliquer
 * les données du problème (A, b, c, p — potentiellement volumineuses), celles-ci
 * sont stockées dans MKPSharedData, accessible via un pointeur brut non-géré.
 * Ce pointeur est simplement copié lors du clonage : toutes les copies partagent
 * le même MKPSharedData en lecture seule, qui vit dans la fonction appelante.
 */

#include "../solver_generic/v1ld_base.hpp"
#include "../solver_generic/v1ld_solver.hpp"
#include "../solver_generic/v1cld_base.hpp"
#include "../solver_generic/v1cld_solver.hpp"
#include "../solver_generic/mkp/mkp_problem_bnb.hpp"
#include "../solver_generic/mkp/mkp_cld_problem.hpp"

#include <gecode/int.hh>
#include <gecode/minimodel.hh>
#include <gecode/search.hh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <vector>

using namespace Gecode;

// ── Mode de calcul de la borne aux nœuds ─────────────────────────────────────
// V1LD  : ζᵢ résolu exactement (B&B multi-dim)   → borne serrée, lent
// V1CLD : ζᵢⱼ LP-relax 1D (min sur j)             → borne lâche, rapide
enum class BoundMode { V1LD, V1CLD };

// ═════════════════════════════════════════════════════════════════════════════
// MKPSharedData — données partagées entre tous les espaces Gecode
// ═════════════════════════════════════════════════════════════════════════════
//
// Contient les données du problème sous deux formes :
//   - flottantes (c, p, A, b) : utilisées par V1LD (more())
//   - entières  (c_int, A_int, b_int, SCALE=1000) : utilisées par la formulation
//     CP Gecode, qui n'accepte que des entiers dans les contraintes linéaires.
//
// Durée de vie : doit rester valide pendant toute la recherche B&B.
// Ce struct est construit dans solve_gecode_v1ld_mkp et son adresse est transmise
// à la racine MKPSpaceV1LD, puis copiée (comme pointeur) à chaque clone.

struct MKPSharedData {
    int    n;         // nombre de variables de décision (items)
    int    m;         // nombre de contraintes par scénario
    int    N;         // nombre de scénarios
    double epsilon;   // niveau de risque ε

    Vec1d  c;   // profits déterministes (n,)
    Vec1d  p;   // probabilités des scénarios (N,)
    Vec3d  A;   // matrices de contraintes (N, m, n) — float
    Vec2d  b;   // capacités (N, m) — float

    // ── Représentation entière pour Gecode ────────────────────────────────────
    // Gecode n'accepte que des int dans linear() et dans les domaines de IntVar.
    // On multiplie tous les coefficients par SCALE=1000 pour conserver 3 décimales.
    // Attention : obj est aussi scalé, il faudra diviser par SCALE en fin de résolution.
    static constexpr int SCALE = 1000;
    std::vector<int>                           c_int;   // c × SCALE
    std::vector<std::vector<std::vector<int>>> A_int;   // A × SCALE (arrondi)
    std::vector<std::vector<int>>              b_int;   // b × SCALE (floor, conservatif)
    int obj_max;   // borne supérieure sur obj (= Σ max(0, c_int[k])), pour le domaine de obj

    // ── Options V1LD pour les appels dans more() ──────────────────────────────
    BoundMode bound_mode = BoundMode::V1LD;  // V1LD (exact) ou V1CLD (LP-relax)
    int    T_root     = 200;   // itérations sous-gradient au premier appel (λ=0, cold start)
    int    T_node     = 30;    // itérations sous-gradient aux nœuds suivants (warm-start)
    double rel_tol    = 1e-4;  // tolérance d'arrêt anticipé du sous-gradient
    int    tol_window = 30;    // fenêtre pour vérifier la tolérance

    // ── Mode patience ─────────────────────────────────────────────────────────
    // Si true, chaque appel more() aux nœuds (hors root) tourne jusqu'à ce que
    // la borne converge à patience_tol près (arrêt anticipé par rel_tol),
    // avec un plafond de patience_max_iters itérations.
    bool   patience_mode      = false;
    double patience_tol       = 0.1;    // tolérance de convergence (10^-1 par défaut)
    int    patience_max_iters = 500;    // plafond d'itérations en mode patience

    // ── Compteurs de statistiques (mutable : modifiés dans more() via const*) ─
    //
    // ub_tightened : nombre de fois où la borne V1LD est strictement inférieure
    //   au domaine courant de obj (ub_int < obj.max()).  Mesure l'utilité de
    //   calculer la borne aux nœuds internes vs seulement à la racine.
    //
    // fails_from_bound : sous-ensemble de ub_tightened où la borne est en plus
    //   strictement inférieure au minimum courant de obj (ub_int < obj.min()).
    //   Dans ce cas, la contradiction est immédiate et directement due à notre
    //   borne — pas à la propagation CP native de Gecode.
    //   Les autres fails (engine.statistics().fail − fails_from_bound) sont
    //   imputables à la propagation CP (réification, chance constraint, etc.).
    // ── Incumbent initial (optionnel) ─────────────────────────────────────────
    // Si > 0, on poste obj >= initial_lb_int dans le constructeur de MKPSpaceV1LD.
    // Cela évite d'explorer des branches dont l'objectif ne peut dépasser une
    // solution déjà connue (typiquement : la solution gloutonne).
    int initial_lb_int = 0;

    mutable int more_calls          = 0;    // nombre total d'appels à more()
    mutable int ub_tightened        = 0;    // appels où la borne réduit obj.max()
    mutable int fails_from_bound    = 0;    // appels où la borne crée une contradiction directe
    mutable double best_open_ub     = 0.0;  // meilleure UB vue sur un nœud non-élagué
    mutable double root_bound_time_s = 0.0; // durée du cold start (premier appel more(), T_root iters)
    mutable double t_bound_total_s   = 0.0; // temps total cumulé dans tous les appels more()

    // ── Ordre de branchement (RATIO, calculé une fois à la construction) ──────
    // Items triés par c_k / (Σ_i Σ_j p_i * A_i[j][k] / b_i[j]) décroissant.
    // Identique au critère de bab_mkp.cpp pour rendre les deux B&B comparables.
    std::vector<int> branch_order;

    MKPSharedData(int n_, int m_, int N_, double eps,
                  Vec1d c_, Vec1d p_, Vec3d A_, Vec2d b_,
                  int T_node_ = 30)
        : n(n_), m(m_), N(N_), epsilon(eps),
          c(std::move(c_)), p(std::move(p_)),
          A(std::move(A_)), b(std::move(b_)),
          T_node(T_node_)
    {
        // Conversion float → int pour la formulation CP
        c_int.resize(n);
        for (int k = 0; k < n; ++k)
            c_int[k] = static_cast<int>(std::llround(c[k] * SCALE));

        A_int.assign(N, std::vector<std::vector<int>>(m, std::vector<int>(n)));
        b_int.assign(N, std::vector<int>(m));
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < m; ++j) {
                for (int k = 0; k < n; ++k)
                    A_int[i][j][k] = static_cast<int>(std::llround(A[i][j][k] * SCALE));
                // floor sur b : légèrement conservatif, garantit que les solutions
                // entières de Gecode respectent les vraies capacités flottantes.
                b_int[i][j] = static_cast<int>(std::floor(b[i][j] * SCALE));
            }

        // Domaine supérieur de obj : somme des profits positifs (scalés)
        obj_max = 0;
        for (int k = 0; k < n; ++k)
            obj_max += std::max(0, c_int[k]);

        // Ordre de branchement RATIO : c_k / poids_surrogate décroissant
        branch_order.resize(n);
        std::iota(branch_order.begin(), branch_order.end(), 0);
        std::sort(branch_order.begin(), branch_order.end(), [&](int a, int bk) {
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
    }
};

// ═════════════════════════════════════════════════════════════════════════════
// GecodeV1LDResult — statistiques de résolution retournées à l'appelant
// ═════════════════════════════════════════════════════════════════════════════

struct GecodeV1LDResult {
    double      objective;     // valeur objective de la meilleure solution (non scalée)
    bool        optimal;       // true si optimalité prouvée (arbre épuisé)
    bool        feasible;      // true si au moins une solution faisable trouvée
    double      greedy_value;  // valeur de la solution gloutonne utilisée comme incumbent initial
    double      best_open_ub;  // meilleure UB V1LD vue sur un nœud non-élagué (≥ objective si optimal)
    double      solve_time_s;  // temps total de résolution (construction + B&B)
    int64_t     nodes;              // nœuds totaux Gecode (engine.statistics().node)
    int64_t     fails;              // nœuds échoués (propagation + borne, au total)
    int64_t     fails_from_bound;   // sous-ensemble : fails causés directement par notre borne V1LD
    int64_t     fails_from_cp;      // estimé = fails − fails_from_bound (propagation CP native)
    int64_t     incumbent_improvements; // nombre de fois où une meilleure solution a été trouvée
    int64_t     more_calls;           // appels totaux à more() (= branches explorées avec commit OK)
    int64_t     ub_tightened;         // sous-ensemble : borne V1LD a réellement réduit obj.max()
    int64_t     depth;                // profondeur maximale atteinte dans l'arbre
    double      root_bound_time_s;    // temps du premier appel à more() (cold start, T_root iters)
    double      t_bound_total_s;      // temps cumulé dans tous les appels more()
    double      t_bab_pure_s;         // temps B&B pur = t_total - t_bound_total
    std::string status;               // "OPTIMAL" | "FEASIBLE" | "INFEASIBLE" | "TIMEOUT"
};

// ═════════════════════════════════════════════════════════════════════════════
// MKPSpaceV1LD — espace de recherche Gecode pour le CCMKP + borne V1LD
// ═════════════════════════════════════════════════════════════════════════════
//
// Hérite de IntMaximizeSpace : Gecode pilote automatiquement la maximisation
// (il maintient une borne inférieure = meilleure solution, et élaguer les
// branches dont l'objectif ne peut dépasser cette borne).

class MKPSpaceV1LD : public IntMaximizeSpace {
public:
    BoolVarArray x;    // x[k] ∈ {0,1} : item k sélectionné ou non
    BoolVarArray z;    // z[i] ∈ {0,1} : scénario i satisfait ou non
    IntVar       obj;  // objectif scalé = Σ_k c_int[k] * x[k]  (÷ SCALE pour la vraie valeur)

    // Pointeur vers les données partagées (read-only, non cloné).
    const MKPSharedData* data;

    // Multiplicateurs Lagrangiens — un seul est utilisé selon bound_mode.
    // V1LD  : lambda_v1ld  (N × n)      — hérité/mis à jour par more() en mode V1LD
    // V1CLD : lambda_v1cld (N × m × n)  — hérité/mis à jour par more() en mode V1CLD
    // L'autre champ reste vide (Vec2d{} ou Vec3d{}) → aucun surcoût mémoire.
    Vec2d lambda_v1ld;    // (N, n)      — mode V1LD
    Vec3d lambda_v1cld;   // (N, m, n)   — mode V1CLD

    // false → premier appel : cold start (T_root iters) ; true → warm-start (T_node iters)
    bool lambda_warmed = false;

    // ─────────────────────────────────────────────────────────────────────────
    // BoundBrancher — brancher custom qui déclenche le bornage V1LD
    // ─────────────────────────────────────────────────────────────────────────
    //
    // Gecode utilise des "branchers" pour choisir quelle variable instancier et
    // avec quelle valeur. Ce brancher gère uniquement les variables x[k].
    //
    // Rôle principal : après chaque commit (fixation de x[k] = v), appeler
    // more() sur l'espace pour calculer la borne V1LD et la poster comme
    // contrainte obj ≤ ⌈UB⌉.
    //
    // Si la contrainte est contradictoire (borne < meilleure solution connue),
    // Gecode détecte l'échec et élaguer la branche sans explorer les fils.
    class BoundBrancher : public Brancher {
    protected:
        ViewArray<Int::BoolView> vars;  // vues internes Gecode sur les BoolVar x[k]

        // PosVal : décision de branchement = (indice de variable, valeur choisie).
        // Hérite de Choice (structure Gecode pour sérialiser les décisions).
        struct PosVal : public Choice {
            int pos;  // indice de la variable choisie
            int val;  // valeur : 1 (branche gauche) ou 0 (branche droite)
            PosVal(const BoundBrancher& b, int p, int v)
                : Choice(b, 2), pos(p), val(v) {}
            // archive() : sérialise la décision pour le moteur de recherche
            void archive(Archive& e) const override {
                Choice::archive(e); e << pos << val;
            }
        };

    public:
        // Constructeur d'initialisation : associe le brancher aux vues x[k]
        BoundBrancher(Home home, ViewArray<Int::BoolView>& v)
            : Brancher(home), vars(v) {}

        // post() : méthode statique pour attacher ce brancher à l'espace home.
        // Gecode alloue le brancher dans l'espace (gestion mémoire custom).
        static void post(Home home, ViewArray<Int::BoolView>& v) {
            new (home) BoundBrancher(home, v);
        }

        // dispose() : libère les ressources du brancher quand l'espace est détruit.
        size_t dispose(Space& home) override {
            Brancher::dispose(home); return sizeof(*this);
        }

        // Constructeur de clonage : appelé par Gecode lors du clonage de l'espace.
        // Copie le brancher dans le nouvel espace home et met à jour les vues.
        BoundBrancher(Space& home, BoundBrancher& b)
            : Brancher(home, b) { vars.update(home, b.vars); }

        // copy() : crée une copie de ce brancher dans l'espace home (requis par Gecode).
        Brancher* copy(Space& home) override {
            return new (home) BoundBrancher(home, *this);
        }

        // status() : retourne true si ce brancher a encore du travail.
        // Ici : au moins une variable x[k] n'est pas encore fixée.
        // Quand toutes les variables sont fixées, Gecode passe au brancher suivant
        // (celui sur z) ou conclut que la solution est complète.
        bool status(const Space& home) const override {
            for (int i = 0; i < vars.size(); ++i)
                if (!vars[i].assigned()) return true;
            return false;
        }

        // choice() : choisit la prochaine variable à brancher (heuristique RATIO).
        // On parcourt les items dans l'ordre de branch_order (ratio c/poids décroissant)
        // et on prend le premier non encore fixé — identique à bab_mkp.cpp.
        // Valeur 1 essayée en premier (favorise la sélection des items rentables).
        Choice* choice(Space& home) override {
            const auto& order = static_cast<MKPSpaceV1LD&>(home).data->branch_order;
            for (int k : order)
                if (!vars[k].assigned())
                    return new PosVal(*this, k, 1);
            GECODE_NEVER;
            return nullptr;
        }

        // choice(Archive) : reconstruit une décision depuis une archive (recherche distribuée).
        Choice* choice(const Space&, Archive& e) override {
            int pos, val; e >> pos >> val;
            return new PosVal(*this, pos, val);
        }

        // commit() : applique la décision (x[pos] = val) à l'espace home.
        //
        // a == 0 : branche principale  → x[pos] = 1
        // a == 1 : branche alternative → x[pos] = 0
        //
        // Après la fixation, si la propagation n'a pas échoué, on appelle more()
        // pour calculer la borne V1LD sur le sous-problème résiduel et la poster
        // comme contrainte sur obj. Si cette contrainte est infaisable (borne < LB),
        // Gecode détecte ES_FAILED lors de la propagation suivante.
        ExecStatus commit(Space& home, const Choice& c, unsigned int a) override {
            const PosVal& pv = static_cast<const PosVal&>(c);
            int pos = pv.pos;
            int val = (a == 0) ? 1 : 0;

            // Tente de fixer vars[pos] = val ; retourne ES_FAILED si contradiction
            ExecStatus es = me_failed(vars[pos].eq(home, val))
                            ? ES_FAILED : ES_OK;

            // Ne calcule la borne que si la fixation est cohérente avec les contraintes
            if (es == ES_OK)
                static_cast<MKPSpaceV1LD&>(home).more();

            return es;
        }

        // print() : affiche la décision pour le débogage (optionnel).
        void print(const Space&, const Choice& c, unsigned int a,
                   std::ostream& os) const override {
            const PosVal& pv = static_cast<const PosVal&>(c);
            os << "x[" << pv.pos << "] = " << (a == 0 ? 1 : 0);
        }
    };

    // ─────────────────────────────────────────────────────────────────────────
    // Constructeur racine
    // ─────────────────────────────────────────────────────────────────────────
    //
    // Construit l'espace initial (racine de l'arbre B&B) :
    //   - crée les variables CP x, z, obj
    //   - pose toutes les contraintes (réification, chance constraint, objectif)
    //   - attache le BoundBrancher sur x et un brancher standard sur z
    //
    // Les coefficients entiers (c_int, A_int, b_int) viennent de MKPSharedData.

    explicit MKPSpaceV1LD(const MKPSharedData* d)
        : x(*this, d->n, 0, 1),
          z(*this, d->N, 0, 1),
          obj(*this, 0, d->obj_max),
          data(d),
          lambda_v1ld (d->bound_mode == BoundMode::V1LD
                       ? Vec2d(d->N, Vec1d(d->n, 0.0)) : Vec2d{}),
          lambda_v1cld(d->bound_mode == BoundMode::V1CLD
                       ? Vec3d(d->N, Vec2d(d->m, Vec1d(d->n, 0.0))) : Vec3d{})
    {
        int n = d->n, m = d->m, N = d->N;

        // Nombre minimum de scénarios à satisfaire pour respecter la chance constraint :
        //   Σ_i z_i ≥ ⌈(1−ε) N⌉
        int min_feas = static_cast<int>(std::ceil((1.0 - d->epsilon) * N));

        // channel() : crée des IntVar liées aux BoolVar x[k], nécessaires pour
        // passer des BoolVar dans les contraintes linear() qui attendent des IntVar.
        IntVarArgs x_int(n);
        for (int k = 0; k < n; ++k) x_int[k] = channel(*this, x[k]);

        // Contraintes réifiées : z[i] = 1 → Σ_k A_i[j][k] x[k] ≤ b_i[j]
        // RM_IMP = implication (si z[i]=1 alors la contrainte doit être satisfaite).
        // Si z[i]=0, la contrainte est ignorée.
        // Propagation : si Gecode prouve que la contrainte ne peut être satisfaite,
        // il force z[i]=0 automatiquement (propagation inverse de la réification).
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < m; ++j) {
                IntArgs coefs(n);
                for (int k = 0; k < n; ++k) coefs[k] = d->A_int[i][j][k];
                linear(*this, coefs, x_int, IRT_LQ, d->b_int[i][j],
                       Reify(z[i], RM_IMP));
            }

        // Contrainte de chance : au moins min_feas scénarios doivent être satisfaits
        linear(*this, z, IRT_GQ, min_feas);

        // Définition de l'objectif : obj = Σ_k c_int[k] * x[k]
        // Gecode maximise la variable retournée par cost() (= obj).
        IntArgs c_args(n);
        for (int k = 0; k < n; ++k) c_args[k] = d->c_int[k];
        linear(*this, c_args, x_int, IRT_EQ, obj);

        // Incumbent initial : si une borne inférieure est fournie (solution gloutonne),
        // on exige obj >= initial_lb_int. Gecode n'explorera alors que les branches
        // pouvant dépasser cette valeur — équivalent à préchauffer l'incumbent.
        if (d->initial_lb_int > 0)
            rel(*this, obj >= d->initial_lb_int);

        // Attache le BoundBrancher sur x.
        // Note : BoolVarArray n'est pas directement convertible en ViewArray,
        // on passe par BoolVarArgs (VarArgArray) qui, lui, est accepté.
        BoolVarArgs x_args(n);
        for (int k = 0; k < n; ++k) x_args[k] = x[k];
        ViewArray<Int::BoolView> xv(*this, x_args);
        BoundBrancher::post(*this, xv);

        // Brancher standard sur z (après que x soit complètement fixé).
        // BOOL_VAL_MAX() : essaie z[i]=1 en premier (favorise les scénarios satisfaits).
        branch(*this, z, BOOL_VAR_NONE(), BOOL_VAL_MAX());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Constructeur de clonage (requis par Gecode)
    // ─────────────────────────────────────────────────────────────────────────
    //
    // Gecode clone l'espace courant à chaque branchement pour explorer les deux
    // branches indépendamment. Ce constructeur copie les variables CP dans le
    // nouvel espace home (via update()) et recopie le pointeur data (pas les données).

    MKPSpaceV1LD(MKPSpaceV1LD& s)
        : IntMaximizeSpace(s), data(s.data),
          lambda_v1ld(s.lambda_v1ld),
          lambda_v1cld(s.lambda_v1cld),
          lambda_warmed(s.lambda_warmed)
    {
        x.update(*this, s.x);
        z.update(*this, s.z);
        obj.update(*this, s.obj);
    }

    // copy() : appelé par Gecode pour cloner cet espace. Doit retourner un nouvel
    // espace construit avec le constructeur de clonage.
    Space*  copy()  override { return new MKPSpaceV1LD(*this); }

    // cost() : retourne la variable que Gecode doit maximiser.
    // IntMaximizeSpace utilise cette variable pour maintenir la borne inférieure
    // (incumbent) et élaguer les branches dont obj ne peut la dépasser.
    IntVar  cost()  const override { return obj; }

    // ─────────────────────────────────────────────────────────────────────────
    // more() — calcul de la borne V1LD et propagation dans l'espace courant
    // ─────────────────────────────────────────────────────────────────────────
    //
    // Appelé par BoundBrancher::commit() après chaque fixation d'une variable x[k].
    //
    // Algorithme :
    //   1. Parcourir x[0..n-1] pour identifier les variables libres (L) et fixées à 1 (F₁).
    //   2. Calculer le profit fixé fp = Σ_{k ∈ F₁} c[k].
    //   3. Construire le sous-problème réduit :
    //        c_free   = c restreint à L
    //        A_free   = A restreint aux colonnes L
    //        b_adj[i][j] = b[i][j] - Σ_{k ∈ F₁} A[i][j][k]   (capacité résiduelle)
    //   4. Appeler solve_v1ld() sur ce sous-problème pour obtenir v1LD*.
    //   5. UB = fp + v1LD*
    //   6. Poster  obj ≤ ⌈UB × SCALE⌉  dans l'espace courant.
    //      Si cette contrainte est contradictoire avec le domaine actuel de obj,
    //      la propagation Gecode détectera un échec → branche élagée.
    //
    // Note : pas de warm-start ici (lambda_init = {}). Chaque appel repart de λ=0.
    // Conséquence : la borne est moins précise que dans bab_mkp.cpp (qui warm-start),
    // mais l'implémentation est plus simple.

    void more() {
        const int n = data->n;
        const int m = data->m;
        const int N = data->N;

        // Étape 1 : identifier l'état courant des variables x
        std::vector<int> free_vars;   // indices des x[k] non encore fixés
        std::vector<int> fixed_one;   // indices des x[k] fixés à 1
        free_vars.reserve(n);
        fixed_one.reserve(n);

        for (int k = 0; k < n; ++k) {
            if (!x[k].assigned())
                free_vars.push_back(k);
            else if (x[k].val() == 1)
                fixed_one.push_back(k);
        }

        // Comptabilise cet appel (même les feuilles où on sort tôt)
        data->more_calls++;

        // À une feuille (tout fixé), pas de borne à calculer : l'objectif est exact.
        if (free_vars.empty()) return;

        const int n_free = static_cast<int>(free_vars.size());

        // Étape 2 : profit des items déjà sélectionnés
        double fp = 0.0;
        for (int k : fixed_one) fp += data->c[k];

        // Étape 3 : construire le sous-problème réduit (flottant, pour V1LD)
        Vec1d c_free(n_free);
        for (int kf = 0; kf < n_free; ++kf)
            c_free[kf] = data->c[free_vars[kf]];

        Vec3d A_free(N, Vec2d(m, Vec1d(n_free, 0.0)));
        Vec2d b_adj(N, Vec1d(m, 0.0));

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < m; ++j) {
                // Capacité consommée par les items déjà pris
                double consumed = 0.0;
                for (int k : fixed_one) consumed += data->A[i][j][k];
                b_adj[i][j] = data->b[i][j] - consumed;

                // Sous-matrice A restreinte aux variables libres
                for (int kf = 0; kf < n_free; ++kf)
                    A_free[i][j][kf] = data->A[i][j][free_vars[kf]];
            }
        }

        // Étape 4 : warm-start — projeter λ du nœud courant sur les variables libres.
        // lambda_warmed == false → λ hérité est nul (cold start) → T_root itérations.
        // lambda_warmed == true  → λ hérité est non-nul (warm start) → T_node itérations.
        // Chaque espace hérite lambda_warmed de son parent via le clone : les petits-fils
        // voient lambda_warmed=true dès que le parent a déjà appelé more().
        bool is_first = !lambda_warmed;

        // Nombre d'itérations et tolérance selon le mode
        int    T_iters;
        double node_tol;
        if (is_first) {
            T_iters   = data->T_root;
            node_tol  = data->rel_tol;
        } else if (data->patience_mode) {
            T_iters   = data->patience_max_iters;   // plafond
            node_tol  = data->patience_tol;         // arrêt anticipé à 10^-1
        } else {
            T_iters   = data->T_node;
            node_tol  = data->rel_tol;
        }

        using Clock = std::chrono::steady_clock;
        auto t_bound_start = Clock::now();

        double ub;  // borne supérieure retournée par le solveur Lagrangien choisi

        if (data->bound_mode == BoundMode::V1LD) {
            // ── Mode V1LD : ζᵢ exact (B&B multi-dim), λ ∈ R^{N×n} ─────────────
            Vec2d lambda_init(N, Vec1d(n_free, 0.0));
            if (!is_first)
                for (int i = 0; i < N; ++i)
                    for (int kf = 0; kf < n_free; ++kf)
                        lambda_init[i][kf] = lambda_v1ld[i][free_vars[kf]];

            MKPProblemBnB prob(c_free, data->p, A_free, b_adj, data->epsilon);
            V1LDResult res = solve_v1ld(prob, T_iters, 0, false, -1.0,
                                        node_tol, data->tol_window, -1e18,
                                        lambda_init);

            for (int i = 0; i < N; ++i)
                for (int kf = 0; kf < n_free; ++kf)
                    lambda_v1ld[i][free_vars[kf]] = res.lambda_star[i][kf];
            ub = fp + res.v1LD_star;

        } else {
            // ── Mode V1CLD : ζᵢⱼ LP-relax 1D, λ ∈ R^{N×m×n} ───────────────────
            Vec3d lambda_init(N, Vec2d(m, Vec1d(n_free, 0.0)));
            if (!is_first)
                for (int i = 0; i < N; ++i)
                    for (int j = 0; j < m; ++j)
                        for (int kf = 0; kf < n_free; ++kf)
                            lambda_init[i][j][kf] = lambda_v1cld[i][j][free_vars[kf]];

            MKPCLDProblem prob(c_free, data->p, A_free, b_adj, data->epsilon);
            V1CLDResult res = solve_v1cld(prob, T_iters, 0, false, -1.0,
                                          node_tol, data->tol_window, -1e18,
                                          lambda_init);

            for (int i = 0; i < N; ++i)
                for (int j = 0; j < m; ++j)
                    for (int kf = 0; kf < n_free; ++kf)
                        lambda_v1cld[i][j][free_vars[kf]] = res.lambda_star[i][j][kf];
            ub = fp + res.v1CLD_star;
        }

        double elapsed_bound = std::chrono::duration<double>(
            Clock::now() - t_bound_start).count();
        data->t_bound_total_s += elapsed_bound;
        if (is_first)
            data->root_bound_time_s = elapsed_bound;

        lambda_warmed = true;
        if (ub > data->best_open_ub) data->best_open_ub = ub;  // mise à jour de la meilleure UB vue
        int    ub_int = static_cast<int>(std::ceil(ub * data->SCALE));

        // La borne est utile uniquement si elle réduit le domaine courant de obj.
        // obj.max() est la valeur maximale encore atteignable selon Gecode.
        if (ub_int < obj.max()) {
            data->ub_tightened++;

            // Si ub_int est déjà en dessous du minimum de obj, la contradiction est
            // immédiate : ce fail vient directement de notre borne, pas de la propagation CP.
            if (ub_int < obj.min())
                data->fails_from_bound++;

            rel(*this, obj <= ub_int);
        }
        // Si ub_int >= obj.max(), la borne ne change rien : on ne la poste pas.
    }
};

// ═════════════════════════════════════════════════════════════════════════════
// greedy_mkp — solution gloutonne faisable pour le CCMKP
// ═════════════════════════════════════════════════════════════════════════════
//
// Trie les items par ratio c_k / (poids surrogate) décroissant, les ajoute
// un à un tant que la chance constraint est satisfaite.
//
// Retourne (valeur gloutonne, affectation binaire x ∈ {0,1}^n).
// Utilisée pour initialiser l'incumbent de Gecode avant la recherche B&B.

std::pair<double, std::vector<int8_t>> greedy_mkp(
    const Vec1d& c, const Vec1d& p,
    const Vec3d& A, const Vec2d& b,
    double epsilon);

// ═════════════════════════════════════════════════════════════════════════════
// solve_gecode_v1ld_mkp — point d'entrée public
// ═════════════════════════════════════════════════════════════════════════════
//
// Construit MKPSharedData, crée l'espace racine, lance le moteur BAB<> de Gecode
// et retourne les statistiques de résolution.
//
// Paramètres :
//   n, m, N   : dimensions du problème
//   epsilon   : niveau de risque ε ∈ (0,1)
//   c, p, A, b : données du problème (float)
//   time_lim  : limite de temps en secondes (-1 = illimité)
//   T_node    : itérations de sous-gradient V1LD par nœud (défaut 30)

GecodeV1LDResult solve_gecode_bab_mkp(
    int n, int m, int N, double epsilon,
    const Vec1d& c, const Vec1d& p,
    const Vec3d& A, const Vec2d& b,
    double    time_lim      = 300.0,
    int       T_root        = 200,
    int       T_node        = 30,
    BoundMode bound_mode    = BoundMode::V1LD,
    bool      patience_mode = false,
    double    patience_tol  = 0.1);
