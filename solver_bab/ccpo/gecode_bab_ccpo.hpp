#pragma once
/*
 * gecode_bab_ccpo.hpp — B&B Gecode pour le CCPO, modes REF et SLD.
 *
 * Cloné de solver_bab/mkp/gecode_bab.hpp -- même principe général
 * (SharedData immuable partagé entre clones + Space + BoundBrancher + moteur
 * DFS manuel dans gecode_bab_ccpo.cpp), noms alignés sur l'original autant
 * que la différence de problème le permet : CCPOSharedData ~ MKPSharedData,
 * CCPOSpaceSLD ~ MKPSpaceSLD, BoundBrancher ~ BoundBrancher, lambda_sld/
 * lambda_warmed/fp_running/fixed_mask ~ identiques.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * DIFFÉRENCES DE MODÈLE AVEC LE CCMKP (résumé -- détail au fil du fichier)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * 1. Décision : x∈{0,1}^n (sélection) chez MKP  vs  q∈{0}∪{L,...,U}^n
 *    (allocation entière) chez CCPO. Conséquences en cascade :
 *      - domaine de q : IntSet à trous, pas un simple BoolVar ;
 *      - contrainte structurelle Σq=Q + cardinalité(count) EN PLUS de la
 *        contrainte de risque -- MKP n'a que "z_i=1 → A_i x≤b_i" ;
 *      - obj_max = Σmax(0,c_i)·U (pas Σmax(0,c_i) -- bug trouvé et corrigé
 *        en cours de route : q_i peut valoir U, pas seulement 1).
 *
 * 2. Contrainte par scénario : LINÉAIRE multi-dimensionnelle chez MKP
 *    (Σ_k A_i[j][k]x_k≤b_i[j], m contraintes/scénario) vs QUADRATIQUE
 *    UNIDIMENSIONNELLE chez CCPO (q'Σ_j q≤B, une seule/scénario). D'où :
 *      - l'encodage CP : termes produits partagés p_ik=q_i·q_k (cf. docstring
 *        de CCPOSpaceSLD ci-dessous), inexistant chez MKP ;
 *      - le sous-problème ζ_j : chez MKP, sac-à-dos 0/1 LINÉAIRE -> B&B avec
 *        borne LP par dimension (Dantzig, MKPProblemBnB), rapide. Chez CCPO,
 *        max linéaire sous contrainte QUADRATIQUE -> pas d'équivalent direct ;
 *        résolu exactement par un B&B dédié (ccpo_sld.hpp::zeta_solve) avec
 *        élagage par relaxation continue (KKT + Cholesky sur la sous-matrice
 *        DENSE, PAS de reconstruction rang-1 -- cf. docstring de
 *        ccpo_sld.hpp pour la discussion complète de ce choix, tranché après
 *        qu'une première approche par relaxation continue RETOURNÉE TELLE
 *        QUELLE se soit révélée invalide : rien ne garantissait alors
 *        ζ_j≤θ_j, condition nécessaire à la reformulation sac-à-dos).
 *      - SANS budget de temps/nœuds interne pour ζ_j (décision actée en
 *        conversation : exactitude non négociable, quel qu'en soit le coût).
 *
 * 3. Presolve spécifique à CCPO (ccpo_sld.hpp::presolve_infeasible_scenarios) :
 *    la faisabilité de ζ_j (∃q∈D : q'Σ_jq≤B) ne dépend PAS de λ -- calculée
 *    UNE FOIS à la construction de CCPOSharedData, indépendamment du mode
 *    (sert à REF ET SLD). Pas d'équivalent chez MKP (sa contrainte linéaire
 *    ne pose pas ce risque de coût catastrophique sur les cas infaisables --
 *    prouver l'infaisabilité d'une contrainte quadratique par B&B peut
 *    coûter des millions de nœuds, mesuré empiriquement : 26s/17M nœuds sur
 *    un seul scénario avant l'ajout du presolve).
 *
 * 4. Pas de DLD/GNN/GNN_ROOT côté CCPO (pour l'instant) -- seulement
 *    BoundMode::{REF, SLD}. Le sac-à-dos min-coût sur z et la boucle de
 *    sous-gradient (ccpo_sld.hpp::solve_ccpo_sld) sont en revanche
 *    PROBLÈME-AGNOSTIQUES, repris quasi tels quels du patron MKP
 *    (sld_solver.cpp) : reparamétrage de centrage cost_j=c+λ_j/p_j-Σλ,
 *    δ_j=max(0,θ_j-ζ_j), pas diminuant 1/√(t+1), meilleur itéré retenu.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * FORMULATION CP DU CCPO
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   max  Σ_i c_i q_i
 *   s.t. Σ_i q_i = Q
 *        #{i : q_i != 0} ≤ k
 *        z_j = 1  →  q' Σ_j q ≤ B                    ∀ j ∈ {1..N}
 *        Σ_j z_j ≥ ⌈(1-ε) N⌉
 *        q_i ∈ {0} ∪ {L,...,U},  z_j ∈ {0,1}
 *
 * ── Encodage de la contrainte quadratique ────────────────────────────────
 * Un terme produit p_ik = q_i·q_k NE DÉPEND PAS du scénario — seul le
 * coefficient Σ_j[i][k] qui le multiplie en dépend. On construit donc les
 * n(n+1)/2 termes produits UNE SEULE FOIS à la racine (partagés par tous les
 * scénarios, via sqr()/mult()), puis pour chaque scénario on ne pose qu'une
 * combinaison LINÉAIRE de ces termes partagés, réifiée par z_j (RM_IMP, même
 * patron que la réification CCMKP) :
 *
 *   q' Σ_j q = Σ_i Σ_j[i][i]·q_i²  +  2·Σ_{i<k} Σ_j[i][k]·(q_i q_k)
 *
 * Coût : O(n²) variables auxiliaires (partagées), PAS O(n²·N).
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include "ccpo_gnn_backend_batch.hpp"
#include "ccpo_sld.hpp"

#include <gecode/int.hh>
#include <gecode/minimodel.hh>
#include <gecode/search.hh>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

using namespace Gecode;

// (i, k, valeur) — triangle supérieur de Sigma_int[j], i <= k, base 0.
using SigmaTriplet = std::array<int, 3>;

// FEAS_ONLY : variante d'isolation méthodologique -- ajoute UNIQUEMENT la détection de faisabilité
// résiduelle exacte (zeta_solve, sans aucune optimisation de λ) par-dessus
// REF. Sépare deux effets aujourd'hui mélangés dans la comparaison REF vs
// SLD : (1) CP seul ne détecte pas l'infaisabilité résiduelle (propagation
// par intervalles trop lâche sur la contrainte quadratique réifiée, cf.
// discussion) -- FEAS_ONLY corrige CE point seul ; (2) SLD/GNN resserrent
// EN PLUS la borne objectif via un bon λ. L'écart FEAS_ONLY -> SLD/GNN
// mesure alors la valeur ajoutée du choix de λ isolément, pas mélangée à
// l'effet (1).
// GNN_BATCHED : utilisé UNIQUEMENT par le driver batché
// (gecode_bab_ccpo_gnn_batched.cpp), qui appelle build_residual() lui-même --
// sert ici seulement à allouer d_full (garde `!= REF`) ; more() y est un no-op.
enum class BoundMode { REF, SLD, FEAS_ONLY, GNN_BATCHED };

// ═════════════════════════════════════════════════════════════════════════
// CCPOSharedData — données immuables, partagées (pointeur brut) entre tous
// les clones d'espace, comme MKPSharedData vis-à-vis de MKPSpaceSLD.
// ═════════════════════════════════════════════════════════════════════════

struct CCPOSharedData {
    int n, N, k, Q, L, U;
    long long B;                                    // budget de risque (déjà entier/scalé)
    int min_feas;                                    // ⌈(1-ε)N⌉

    std::vector<int> c_int;                          // (n,)
    std::vector<std::vector<SigmaTriplet>> sigma;     // (N,) de (i,k,val), i<=k
    std::vector<std::vector<std::vector<double>>> sigma_dense;  // (N,n,n) -- construit une fois,
                                                                  // réutilisé par more() à chaque nœud
                                                                  // (extraction de sous-matrices) et
                                                                  // par solve_ccpo_sld/zeta_solve.

    // ── Presolve : scénarios prouvés infaisables (z_j fixé à 0) ───────────
    // cf. ccpo_sld.hpp::presolve_infeasible_scenarios -- indépendant de λ,
    // calculé une fois à la construction. Sert à REF (contrainte réifiée
    // jamais postée) ET à SLD (zeta_solve jamais appelé sur ces scénarios).
    std::vector<bool> infeasible_scenario;
    int n_infeasible = 0;

    int initial_lb_int = 0;                           // incumbent glouton (0 = aucun)

    // ── Mode de calcul de la borne (cf. docstring du fichier, point 4) ────
    BoundMode bound_mode = BoundMode::REF;
    int    T_root     = 100;   // itérations de sous-gradient au premier appel more() (cold start)
    int    T_node     = 20;    // itérations aux nœuds suivants (warm-start du lambda)
    int    omp_threads = 1;    // >1 : boucle scenario OpenMP dans solve_ccpo_sld (mode SLD)
    double rel_tol    = 1e-3;
    int    tol_window = 15;

    // ── Ordre de branchement sur q ────────────────────────────────────────
    // Ratio rendement / risque moyen -- analogue du ratio c_k/poids_surrogate
    // du CCMKP (cf. MKPSharedData::branch_order), "poids" = contribution
    // moyenne à la variance (diagonale de Sigma_j) au lieu de capacité
    // consommée. Calculé une fois à la construction.
    std::vector<int> branch_order;

    // ── Statistiques (mutable : modifiées dans more() via const*, même
    // convention que MKPSharedData) ───────────────────────────────────────
    mutable int64_t more_calls          = 0;
    mutable int64_t ub_tightened        = 0;   // more() a réduit obj.max()
    mutable int64_t fails_from_bound    = 0;   // borne -> contradiction immédiate
    mutable double  root_bound_time_s   = 0.0;
    mutable double  root_bound_value    = 0.0;
    mutable double  t_bound_total_s     = 0.0;

    CCPOSharedData(int n_, int N_, int k_, int Q_, int L_, int U_, long long B_,
                   int min_feas_,
                   std::vector<int> c_,
                   std::vector<std::vector<SigmaTriplet>> sigma_)
        : n(n_), N(N_), k(k_), Q(Q_), L(L_), U(U_), B(B_), min_feas(min_feas_),
          c_int(std::move(c_)), sigma(std::move(sigma_))
    {
        std::vector<double> avg_risk(n, 0.0);
        for (int j = 0; j < N; ++j)
            for (const auto& t : sigma[j])
                if (t[0] == t[1]) avg_risk[t[0]] += t[2];
        for (int i = 0; i < n; ++i) avg_risk[i] /= std::max(1, N);

        branch_order.resize(n);
        std::iota(branch_order.begin(), branch_order.end(), 0);
        std::sort(branch_order.begin(), branch_order.end(), [&](int a, int b) {
            double ra = (avg_risk[a] > 1e-9) ? c_int[a] / avg_risk[a] : c_int[a] * 1e12;
            double rb = (avg_risk[b] > 1e-9) ? c_int[b] / avg_risk[b] : c_int[b] * 1e12;
            return ra > rb;
        });

        sigma_dense.assign(N, std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0)));
        for (int j = 0; j < N; ++j)
            for (const auto& t : sigma[j]) {
                sigma_dense[j][t[0]][t[1]] = t[2];
                sigma_dense[j][t[1]][t[0]] = t[2];
            }

        infeasible_scenario = presolve_infeasible_scenarios(
            n, Q, U, static_cast<double>(B), sigma);
        n_infeasible = static_cast<int>(
            std::count(infeasible_scenario.begin(), infeasible_scenario.end(), true));
    }
};

// ═════════════════════════════════════════════════════════════════════════
// CCPOResult — statistiques de résolution retournées à l'appelant
// ═════════════════════════════════════════════════════════════════════════

struct CCPOResult {
    // Tous les champs ont une valeur par défaut : un chemin de retour anticipé
    // (ex. court-circuit d'infaisabilité globale) qui oublierait d'en fixer un
    // laisse une valeur sûre (0/false), jamais une donnée non initialisée --
    // bug rencontré et corrigé (fails_from_bound/fails_from_cp contenaient des
    // valeurs indéterminées avant ce correctif, cf. discussion).
    double      objective = std::numeric_limits<double>::quiet_NaN();
    bool        optimal = false;       // true si optimalité prouvée (arbre épuisé)
    bool        feasible = false;      // true si au moins une solution faisable trouvée
    double      greedy_value = std::numeric_limits<double>::quiet_NaN();
    double      solve_time_s = 0.0;    // temps total de résolution
    int64_t     nodes = 0;             // nœuds totaux explorés
    int64_t     fails = 0;             // nœuds échoués au total
    int64_t     fails_from_bound = 0;  // sous-ensemble : échec immédiat dû à la borne (0 en REF)
    int64_t     fails_from_cp = 0;     // = fails - fails_from_bound
    int64_t     depth = 0;             // profondeur maximale atteinte
    int64_t     incumbent_improvements = 0;
    int64_t     more_calls        = 0;
    int64_t     ub_tightened      = 0;
    double      root_bound_time_s = 0.0;
    double      root_bound_value  = 0.0;
    double      t_bound_total_s   = 0.0;
    std::string status;        // "OPTIMAL" | "FEASIBLE" | "INFEASIBLE" | "TIMEOUT"
};

// ═════════════════════════════════════════════════════════════════════════
// CCPOSpaceSLD — espace de recherche Gecode pour le CCPO (modes REF/SLD)
// ═════════════════════════════════════════════════════════════════════════

class CCPOSpaceSLD : public IntMaximizeSpace {
public:
    IntVarArray  q;    // q[i] ∈ {0} ∪ {L,...,U} : lots alloués à l'actif i
    BoolVarArray z;    // z[j] ∈ {0,1} : scénario j satisfait ou non
    IntVar       obj;  // objectif = Σ_i c_int[i] * q[i]

    const CCPOSharedData* data;

    // Multiplicateurs lagrangiens -- (N,n), vide (taille 0) en mode REF pour
    // ne payer aucun coût mémoire/clonage (même garde que MKPSpaceSLD::lambda_sld).
    std::vector<std::vector<double>> lambda_sld;
    bool lambda_warmed = false;

    // ── État résiduel incrémental (mêmes noms/rôles que MKPSpaceSLD) ──────
    // b_res[j]     : budget de risque résiduel du scénario j (B - risque déjà
    //                engagé par les q_i fixés actifs), maintenu incrémentalement.
    // fixed_mask[i]: 0 = pas encore comptabilisé, 1 = déjà comptabilisé.
    // fp_running   : profit fixé (Σ c_int[i]*q_i pour i déjà fixé), incrémental.
    // fixed_sum/fixed_count_active : Σq_i et #{i fixé actif} déjà comptabilisés
    //                (nécessaires pour Q_res/k_res du sous-problème résiduel --
    //                pas d'équivalent chez MKP, dont Q/k n'existent pas).
    std::vector<double> b_res;
    std::vector<int8_t> fixed_mask;
    double fp_running = 0.0;
    int    fixed_sum_running = 0;
    int    fixed_count_active_running = 0;

    // d_full[j][i] : champ linéaire induit par TOUTES les variables déjà
    // fixées ACTIVES (i'!=i) sur l'indice i (fixé ou libre), scénario j --
    // Σ_{i' fixé actif, i'!=i} v_i'·Sigma_j[i'][i]. Maintenu incrémentalement
    // en parallèle de b_res (CORRECTIF, cf. docstring
    // ccpo_sld.hpp "terme croisé fixé×libre") ; restreint à free_vars au
    // moment de l'appel à solve_ccpo_sld -> R_ext. (N,n), vide en mode REF
    // (même garde que lambda_sld).
    std::vector<std::vector<double>> d_full;

    // ─────────────────────────────────────────────────────────────────────
    // BoundBrancher — brancher custom sur q (nom aligné sur MKPSpaceSLD --
    // ce brancher déclenche more() après chaque fixation, comme chez MKP).
    // ─────────────────────────────────────────────────────────────────────
    // Ordre de variable : branch_order (ratio rendement/risque décroissant).
    // Ordre de valeur : plus grande valeur du domaine courant en premier,
    // branchement à deux voies (=v / ≠v) -- schéma standard Gecode.
    //
    // Contrairement à MKPSpaceSLD, PAS de defer_bound_to_driver ici : commit()
    // ne calcule jamais la borne lui-même, c'est TOUJOURS le driver DFS
    // (solve_gecode_bab_ccpo) qui appelle more() explicitement après un
    // premier status() de propagation CP pure -- cf. discussion sur le coût
    // potentiellement élevé de zeta_solve : on ne veut jamais le payer sur un
    // nœud que le CP seul aurait de toute façon tué gratuitement.
    class BoundBrancher : public Brancher {
    protected:
        ViewArray<Int::IntView> vars;

        struct PosVal : public Choice {
            int pos;
            int val;
            PosVal(const BoundBrancher& b, int p, int v) : Choice(b, 2), pos(p), val(v) {}
            void archive(Archive& e) const override {
                Choice::archive(e); e << pos << val;
            }
        };

    public:
        BoundBrancher(Home home, ViewArray<Int::IntView>& v) : Brancher(home), vars(v) {}

        static void post(Home home, ViewArray<Int::IntView>& v) {
            new (home) BoundBrancher(home, v);
        }

        size_t dispose(Space& home) override {
            Brancher::dispose(home); return sizeof(*this);
        }

        BoundBrancher(Space& home, BoundBrancher& b) : Brancher(home, b) {
            vars.update(home, b.vars);
        }

        Brancher* copy(Space& home) override {
            return new (home) BoundBrancher(home, *this);
        }

        bool status(const Space&) const override {
            for (int i = 0; i < vars.size(); ++i)
                if (!vars[i].assigned()) return true;
            return false;
        }

        Choice* choice(Space& home) override {
            const auto& order = static_cast<CCPOSpaceSLD&>(home).data->branch_order;
            for (int i : order)
                if (!vars[i].assigned())
                    return new PosVal(*this, i, vars[i].max());
            GECODE_NEVER;
            return nullptr;
        }

        Choice* choice(const Space&, Archive& e) override {
            int pos, val; e >> pos >> val;
            return new PosVal(*this, pos, val);
        }

        // a == 0 : q[pos] = val   (branche principale, valeur haute d'abord)
        // a == 1 : q[pos] != val  (branche alternative)
        ExecStatus commit(Space& home, const Choice& c, unsigned int a) override {
            const PosVal& pv = static_cast<const PosVal&>(c);
            ExecStatus es = (a == 0)
                ? (me_failed(vars[pv.pos].eq(home, pv.val)) ? ES_FAILED : ES_OK)
                : (me_failed(vars[pv.pos].nq(home, pv.val)) ? ES_FAILED : ES_OK);
            return es;
        }

        void print(const Space&, const Choice& c, unsigned int a,
                   std::ostream& os) const override {
            const PosVal& pv = static_cast<const PosVal&>(c);
            os << "q[" << pv.pos << "] " << (a == 0 ? "= " : "!= ") << pv.val;
        }
    };

    // ─────────────────────────────────────────────────────────────────────
    // Constructeur racine
    // ─────────────────────────────────────────────────────────────────────
    explicit CCPOSpaceSLD(const CCPOSharedData* d)
        : q(*this, d->n,
            IntSet({std::pair<int, int>{0, 0}, std::pair<int, int>{d->L, d->U}})),
          z(*this, d->N, 0, 1),
          data(d),
          lambda_sld(d->bound_mode == BoundMode::SLD
                         ? std::vector<std::vector<double>>(d->N, std::vector<double>(d->n, 0.0))
                         : std::vector<std::vector<double>>{}),
          b_res(d->B > 0 ? std::vector<double>(d->N, static_cast<double>(d->B))
                         : std::vector<double>(d->N, 0.0)),
          fixed_mask(d->n, 0),
          d_full(d->bound_mode != BoundMode::REF
                     ? std::vector<std::vector<double>>(d->N, std::vector<double>(d->n, 0.0))
                     : std::vector<std::vector<double>>{})
    {
        const int n = d->n, N = d->N;

        // Σ q_i = Q
        linear(*this, q, IRT_EQ, d->Q);

        // Cardinalité : au plus k actifs actifs <=> au moins n-k valeurs nulles.
        count(*this, q, 0, IRT_GQ, n - d->k);

        // ── Termes produits partagés p[i][k] = q_i * q_k (i<=k), construits
        // une seule fois -- cf. docstring du fichier. ─────────────────────
        const int max_prod = d->U * d->U;
        std::vector<std::vector<IntVar>> p(n, std::vector<IntVar>(n));
        for (int i = 0; i < n; ++i) {
            IntVar sq(*this, 0, max_prod);
            sqr(*this, q[i], sq);
            p[i][i] = sq;
        }
        for (int i = 0; i < n; ++i)
            for (int kk = i + 1; kk < n; ++kk) {
                IntVar pr(*this, 0, max_prod);
                mult(*this, q[i], q[kk], pr);
                p[i][kk] = pr;
                p[kk][i] = pr;
            }

        // ── Contraintes de risque réifiées, une par scénario ──────────────
        // Scénarios presolve-infaisables : z_j fixé à 0 directement, AUCUNE
        // contrainte réifiée postée.
        for (int j = 0; j < N; ++j) {
            if (d->infeasible_scenario[j]) {
                rel(*this, z[j] == 0);
                continue;
            }
            const auto& triplets = d->sigma[j];
            const int nt = static_cast<int>(triplets.size());
            IntArgs coefs(nt);
            IntVarArgs terms(nt);
            for (int t = 0; t < nt; ++t) {
                const int i = triplets[t][0], kk = triplets[t][1], val = triplets[t][2];
                coefs[t] = (i == kk) ? val : 2 * val;
                terms[t] = p[i][kk];
            }
            linear(*this, coefs, terms, IRT_LQ, static_cast<int>(d->B),
                   Reify(z[j], RM_IMP));
        }

        // Contrainte de chance : au moins min_feas scénarios satisfaits.
        linear(*this, z, IRT_GQ, d->min_feas);

        // Objectif.
        // BUG CORRIGÉ : obj_max = Σ max(0,c_i) est la borne correcte
        // pour CCMKP (x_k binaire, 0 ou 1) mais FAUSSE pour CCPO -- q_i peut
        // valoir jusqu'à U, pas seulement 1.
        int obj_max = 0;
        for (int v : d->c_int) obj_max += std::max(0, v) * d->U;
        obj = IntVar(*this, 0, obj_max);
        linear(*this, d->c_int, q, IRT_EQ, obj);

        if (d->initial_lb_int > 0)
            rel(*this, obj >= d->initial_lb_int);

        // Branchement : BoundBrancher sur q, puis brancher standard sur z.
        IntVarArgs q_args(n);
        for (int i = 0; i < n; ++i) q_args[i] = q[i];
        ViewArray<Int::IntView> qv(*this, q_args);
        BoundBrancher::post(*this, qv);
        branch(*this, z, BOOL_VAR_NONE(), BOOL_VAL_MAX());
    }

    // ─────────────────────────────────────────────────────────────────────
    // Constructeur de clonage
    // ─────────────────────────────────────────────────────────────────────
    CCPOSpaceSLD(CCPOSpaceSLD& s)
        : IntMaximizeSpace(s), data(s.data), lambda_sld(s.lambda_sld),
          lambda_warmed(s.lambda_warmed), b_res(s.b_res), fixed_mask(s.fixed_mask),
          fp_running(s.fp_running), fixed_sum_running(s.fixed_sum_running),
          fixed_count_active_running(s.fixed_count_active_running),
          d_full(s.d_full)
    {
        q.update(*this, s.q);
        z.update(*this, s.z);
        obj.update(*this, s.obj);
    }

    Space* copy() override { return new CCPOSpaceSLD(*this); }

    IntVar cost() const override { return obj; }

    // ─────────────────────────────────────────────────────────────────────
    // more() — calcul de la borne SLD et propagation (cf. MKPSpaceSLD::more())
    // ─────────────────────────────────────────────────────────────────────
    //
    // Appelé EXPLICITEMENT par le driver DFS (jamais depuis commit(), cf.
    // docstring de BoundBrancher). Algorithme, transposé de MKPSpaceSLD :
    //   1. Scan de q : identifie les variables nouvellement fixées depuis le
    //      nœud parent, met à jour fp_running/fixed_sum_running/
    //      fixed_count_active_running/b_res[j] (incrémental, delta seulement).
    //   2. Construit le sous-problème résiduel (indices encore libres) :
    //      cost_free = c_int restreint, Sigma_free = sigma_dense restreint,
    //      Q_res = Q - fixed_sum_running, k_res = k - fixed_count_active_running.
    //   3. Warm-start lambda (projection du parent sur les indices libres) ou
    //      cold start (T_root, lambda=0) si première fois dans cette lignée.
    //   4. UB = fp_running + SLD*(sous-problème résiduel).
    //   5. Poste obj ≤ ⌊UB + ε⌋ (PAS ceil : contrairement à MKP, c_int n'est
    //      jamais re-scalé dans ce solveur -- obj est un entier EXACT, donc
    //      floor(UB) est la borne entière valide la plus SERRÉE, pas une
    //      marge de sécurité contre un arrondi c/c_int qui n'existe pas ici).
    //
    // Retourne true si une contrainte a réellement été postée (donc si une
    // repropagation via status() a un intérêt) -- même contrat que MKP.
    // ── Sous-problème résiduel (extrait de more() pour être
    // réutilisé tel quel par le driver GNN batché, gecode_bab_ccpo_gnn_batched.cpp
    // -- AUCUN changement de comportement pour more()). ──────────────────────
    using Residual = CCPOResidualData;

    // Met à jour l'état incrémental (fp_running, b_res, d_full, ...) pour les
    // variables nouvellement fixées depuis l'appel précédent, puis construit
    // le sous-problème résiduel. Retourne false si feuille (aucune variable
    // libre -- objectif déjà exact). Incrémente data->more_calls.
    bool build_residual(Residual& out) {
        const int n = data->n, N = data->N;

        std::vector<int> free_vars;
        free_vars.reserve(n);
        for (int i = 0; i < n; ++i) {
            if (!q[i].assigned()) { free_vars.push_back(i); continue; }
            if (fixed_mask[i] != 0) continue;  // déjà comptabilisé à un appel antérieur
            fixed_mask[i] = 1;
            const int v = q[i].val();
            if (v > 0) {
                fp_running += data->c_int[i] * v;
                fixed_sum_running += v;
                fixed_count_active_running += 1;
                // Met à jour le risque résiduel de chaque scénario ELIGIBLE
                // (inutile pour les presolve-infaisables, z_j=0 forcé) :
                // interaction de i avec lui-même ET avec tous les i' déjà
                // fixés actifs à un appel PRÉCÉDENT (fixed_mask[i']==1 avant
                // ce scan -- les fixations du MÊME scan sont traitées dans
                // l'ordre de la boucle, donc déjà comptabilisées au fur et
                // à mesure via ce même mécanisme).
                for (int jx = 0; jx < N; ++jx) {
                    if (data->infeasible_scenario[jx]) continue;
                    const auto& Sig = data->sigma_dense[jx];
                    double delta_risk = Sig[i][i] * static_cast<double>(v) * v;
                    for (int ip = 0; ip < i; ++ip) {
                        if (fixed_mask[ip] && q[ip].val() > 0)
                            delta_risk += 2.0 * Sig[i][ip] * v * q[ip].val();
                    }
                    for (int ip = i + 1; ip < n; ++ip) {
                        if (fixed_mask[ip] && q[ip].assigned() && q[ip].val() > 0)
                            delta_risk += 2.0 * Sig[i][ip] * v * q[ip].val();
                    }
                    b_res[jx] -= delta_risk;

                    // CORRECTIF (terme croisé fixé×libre, cf. ccpo_sld.hpp) :
                    // q_i=v induit un champ linéaire 2·v·Sig[i][l] sur TOUT
                    // autre indice l (fixé ou libre) -- accumulé dans d_full,
                    // restreint aux indices libres au moment de l'appel à
                    // solve_ccpo_sld (R_ext ci-dessous).
                    for (int l = 0; l < n; ++l)
                        if (l != i) d_full[jx][l] += Sig[i][l] * v;
                }
            }
        }

        data->more_calls++;
        if (free_vars.empty()) return false;  // feuille : objectif déjà exact

        const int n_free = static_cast<int>(free_vars.size());
        out.n_free = n_free;
        out.Q_res = data->Q - fixed_sum_running;
        out.k_res = data->k - fixed_count_active_running;

        out.cost_free.assign(n_free, 0.0);
        for (int a = 0; a < n_free; ++a) out.cost_free[a] = data->c_int[free_vars[a]];

        out.Sigma_free.assign(
            N, std::vector<std::vector<double>>(n_free, std::vector<double>(n_free)));
        out.R_ext.assign(N, std::vector<double>(n_free, 0.0));
        for (int jx = 0; jx < N; ++jx) {
            if (data->infeasible_scenario[jx]) continue;  // jamais utilisé
            for (int a = 0; a < n_free; ++a) {
                for (int b = 0; b < n_free; ++b)
                    out.Sigma_free[jx][a][b] = data->sigma_dense[jx][free_vars[a]][free_vars[b]];
                out.R_ext[jx][a] = d_full[jx][free_vars[a]];
            }
        }
        out.free_vars = std::move(free_vars);
        return true;
    }

    bool more() {
        if (data->bound_mode == BoundMode::REF || data->bound_mode == BoundMode::GNN_BATCHED) {
            data->more_calls++;
            return false;
        }
        // FEAS_ONLY partage TOUTE la construction du sous-problème résiduel
        // avec SLD (build_residual) -- seul le traitement APRÈS cette
        // construction diverge (cf. bas de la fonction).
        const int N = data->N;

        Residual res_;
        if (!build_residual(res_)) return false;  // feuille
        const std::vector<int>& free_vars = res_.free_vars;
        const int n_free = res_.n_free;
        const int Q_res = res_.Q_res, k_res = res_.k_res;
        const std::vector<double>& cost_free = res_.cost_free;
        const auto& Sigma_free = res_.Sigma_free;
        const auto& R_ext = res_.R_ext;

        // ── FEAS_ONLY : détection de faisabilité résiduelle exacte SEULE,
        // sans aucune optimisation de λ (cf. FEAS_ONLY plus haut). zeta_solve avec un coût
        // bidon (tout à zéro) -- la faisabilité qu'il retourne ne dépend
        // jamais du coût, cf. ccpo_sld.hpp -- juste pour compter combien de
        // scénarios ELIGIBLES ont au moins un q faisable. Aucune borne
        // objectif resserrée, seul un échec immédiat si trop peu de
        // scénarios restent faisables.
        if (data->bound_mode == BoundMode::FEAS_ONLY) {
            std::vector<int> order_dummy(n_free);
            std::iota(order_dummy.begin(), order_dummy.end(), 0);
            std::vector<double> zero_cost(n_free, 0.0);
            int n_feasible_eligible = 0;
            for (int jx = 0; jx < N; ++jx) {
                if (data->infeasible_scenario[jx]) continue;
                ZetaResult zr = zeta_solve_cfg(n_free, Q_res, k_res, data->L, data->U,
                                                zero_cost, Sigma_free[jx], b_res[jx],
                                                order_dummy, R_ext[jx]);
                if (zr.feasible) ++n_feasible_eligible;
            }
            if (n_feasible_eligible < data->min_feas) {
                data->fails_from_bound++;
                rel(*this, obj <= -1);
                return true;
            }
            return false;  // faisable : aucune borne resserrée, juste la propagation CP habituelle
        }

        // Warm-start : projection de lambda_sld (hérité du parent par clonage)
        // sur les indices encore libres ; cold start (0, T_root) si première
        // fois dans cette lignée.
        std::vector<std::vector<double>> lambda_init(N, std::vector<double>(n_free, 0.0));
        if (lambda_warmed) {
            for (int jx = 0; jx < N; ++jx)
                for (int a = 0; a < n_free; ++a)
                    lambda_init[jx][a] = lambda_sld[jx][free_vars[a]];
        }
        const int T_iters = lambda_warmed ? data->T_node : data->T_root;
        const bool is_first_call = !lambda_warmed;

        using Clock = std::chrono::steady_clock;
        const auto t0 = Clock::now();
        if (data->omp_threads > 1) omp_set_num_threads(data->omp_threads);
        SLDResult res = solve_ccpo_sld(
            n_free, N, k_res, Q_res, data->L, data->U, cost_free, Sigma_free,
            b_res, data->min_feas, data->infeasible_scenario, T_iters, lambda_init,
            R_ext, data->rel_tol, data->tol_window, data->omp_threads > 1);
        const double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
        data->t_bound_total_s += elapsed;
        if (is_first_call) {
            data->root_bound_time_s = elapsed;
            data->root_bound_value = res.feasible ? res.L_star : 0.0;
        }
        lambda_warmed = true;

        // Sous-problème résiduel prouvé infaisable (cf. SLDResult::feasible) :
        // moins de min_feas scénarios restent satisfaisables une fois les
        // variables déjà fixées prises en compte -- indépendant de λ. Force
        // un échec immédiat (obj.min()=0 par construction, jamais atteignable
        // par une borne < 0) au lieu de calculer une borne à partir de
        // L_star=NaN (source du crash Gecode::Int::OutOfLimits corrigé le
        // floor(-infini) en int est un comportement indéfini).
        if (!res.feasible) {
            data->fails_from_bound++;
            rel(*this, obj <= -1);
            return true;
        }

        // Reprojection de lambda_star sur l'espace complet (n) pour l'héritage
        // par les enfants -- mêmes positions FIXÉES inchangées (non pertinentes).
        for (int jx = 0; jx < N; ++jx)
            for (int a = 0; a < n_free; ++a)
                lambda_sld[jx][free_vars[a]] = res.lambda_star[jx][a];

        const double ub = fp_running + res.L_star;
        const int ub_int = static_cast<int>(std::floor(ub + 1e-6));

        const bool tightens = ub_int < obj.max();
        if (tightens) {
            data->ub_tightened++;
            if (ub_int < obj.min()) data->fails_from_bound++;
            rel(*this, obj <= ub_int);
        }
        return tightens;
    }
};

// ═════════════════════════════════════════════════════════════════════════
// solve_gecode_bab_ccpo — DFS manuelle (cf. gecode_bab_ccpo.cpp)
// ═════════════════════════════════════════════════════════════════════════

// Incumbent glouton (défini dans gecode_bab_ccpo.cpp, exposé pour que le
// driver GNN batché démarre avec le MÊME incumbent initial que REF/SLD).
struct GreedyCCPOResult {
    double value;
    std::vector<int> q;
    bool feasible;
};
GreedyCCPOResult greedy_ccpo(
    int n, int N, int k, int Q, int L, int U,
    const std::vector<int>& c_int,
    const std::vector<std::vector<SigmaTriplet>>& sigma,
    long long B, int min_feas);

CCPOResult solve_gecode_bab_ccpo(
    int n, int N, int k, int Q, int L, int U, long long B, int min_feas,
    const std::vector<int>& c_int,
    const std::vector<std::vector<SigmaTriplet>>& sigma,
    double time_lim,
    BoundMode bound_mode = BoundMode::REF,
    int T_root = 100,
    int T_node = 20,
    int omp_threads = 1);
