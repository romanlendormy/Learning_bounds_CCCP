#pragma once
/*
 * gecode_bab_sld.hpp — B&B Gecode pour le CCMKP avec borne SLD à chaque nœud.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * ARCHITECTURE GÉNÉRALE
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Ce fichier combine deux mécanismes :
 *
 *  1. Un modèle CP Gecode (MKPSpaceSLD, hérite de IntMaximizeSpace) :
 *     - encode le CCMKP comme un programme en variables entières/booléennes
 *     - Gecode pilote l'arbre B&B, gère la propagation et le clonage des espaces
 *
 *  2. Un bornage Lagrangien (méthode more()) :
 *     - appelé après chaque décision de branchement
 *     - calcule SLD sur le sous-problème résiduel (variables encore libres)
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
 * BORNE SLD AUX NŒUDS DE L'ARBRE
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Après chaque fixation de variable x_k = v par le brancher, more() est appelé :
 *
 *   Soit F₁ = {k : x_k fixé à 1},  F₀ = {k : x_k fixé à 0},  L = {k : x_k libre}.
 *
 *   1. Profit fixé       : fp = Σ_{k ∈ F₁} c_k
 *   2. Capacité réduite  : b̃_i[j] = b_i[j] − Σ_{k ∈ F₁} A_i[j][k]
 *   3. Sous-problème résiduel sur L (avec b̃_i, A restreinte aux colonnes L)
 *   4. SLD* = solve_sld(sous-problème, T_node itérations)
 *   5. UB = fp + SLD*
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

#include "../../solver_generic/sld_base.hpp"
#include "../../solver_generic/sld_solver.hpp"
#include "../../solver_generic/dld_base.hpp"
#include "../../solver_generic/dld_solver.hpp"
#include "../../solver_generic/mkp/mkp_problem_bnb.hpp"
#include "../../solver_generic/mkp/mkp_problem_bnb_surrogate.hpp"
#include "../../solver_generic/mkp/mkp_dld_problem.hpp"
#include "gnn_bound_backend.hpp"

#include <gecode/int.hh>
#include <gecode/minimodel.hh>
#include <gecode/search.hh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

using namespace Gecode;

// ── Mode de calcul de la borne aux nœuds ─────────────────────────────────────
// SLD  : ζᵢ résolu exactement (B&B multi-dim)   → borne serrée, lent
// DLD : ζᵢⱼ LP-relax 1D (min sur j)             → borne lâche, rapide
// GNN   : borne one-shot prédite par un GNN pré-entraîné (gnn_bound_backend.hpp)
//         — pas de descente de sous-gradient, un seul aller-retour vers un
//         process Python persistant par appel more() — À CHAQUE nœud.
// GNN_ROOT : même inférence GNN one-shot, mais UNE SEULE FOIS (au premier
//         appel more() de chaque lignée, cf. lambda_warmed) — tous les appels
//         suivants se comportent comme REF (aucune borne recalculée/postée).
//         Teste si la seule borne de racine, sans la repayer à chaque nœud,
//         suffit à élaguer utilement l'arbre (compromis coût/qualité opposé
//         au mode GNN plein).
// REF   : squelette SLD NU — même classe, même heuristique de branchement
//         (ordre RATIO), même amorçage d'incumbent glouton, même moteur DFS
//         maison (solve_gecode_bab_mkp) — mais more() ne reconstruit AUCUN
//         sous-problème et ne calcule/poste AUCUNE borne. Sert de référence
//         pour comparer "Gecode base" (BAB natif Gecode, ordre de déclaration,
//         pas d'incumbent initial) à SLD/DLD à heuristique/moteur FIXÉS :
//         l'écart base vs REF isole l'effet du branchement+incumbent+moteur ;
//         l'écart REF vs SLD/DLD isole l'effet de la borne seule.
// GNN_ROOT_DLD : symétrique de GNN_ROOT, mais avec le checkpoint/backend
// DLD (ζ DP 1D exact) au lieu de SLD — cf. GNNDLDSdEditNative
// (gnn_dld_sd_edit_native.hpp), adaptateur mono-nœud autour de
// GNNDLDSdEditNativeBatched (query_batch, taille de batch 1). Ajouté pour
// obtenir une borne "forward GNN root-only" DLD comparable à GNN_ROOT côté
// SLD (cf. mémoire project_ccmkp_trillium_migration / test_paper_trillium
// GNN-vs-sous-gradient).
enum class BoundMode { SLD, DLD, GNN, GNN_ROOT, GNN_ROOT_DLD, REF };

// ═════════════════════════════════════════════════════════════════════════════
// MKPSharedData — données partagées entre tous les espaces Gecode
// ═════════════════════════════════════════════════════════════════════════════
//
// Contient les données du problème sous deux formes :
//   - flottantes (c, p, A, b) : utilisées par SLD (more())
//   - entières  (c_int, A_int, b_int, SCALE=1000) : utilisées par la formulation
//     CP Gecode, qui n'accepte que des entiers dans les contraintes linéaires.
//
// Durée de vie : doit rester valide pendant toute la recherche B&B.
// Ce struct est construit dans solve_gecode_sld_mkp et son adresse est transmise
// à la racine MKPSpaceSLD, puis copiée (comme pointeur) à chaque clone.

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
    // On multiplie tous les coefficients par SCALE pour conserver les décimales.
    // Attention : obj est aussi scalé, il faudra diviser par SCALE en fin de résolution.
    //
    // SCALE=10000 (4 décimales) et NON 1000 (3 décimales) : les datasets
    // ccmk-song-pisinger-*.txt ont des coûts c[] ET coefficients A[] à 4
    // décimales (ex. c=470.7849). À SCALE=1000, llround(c[k]*1000) perd la
    // 4e décimale : la somme flottante EXACTE `fp` (utilisée par more()/la
    // borne GNN pour décider obj<=ceil(ub*SCALE)) peut alors différer de la
    // somme des c_int RÉELLEMENT vue par Gecode (obj.min()) — jusqu'à un plein
    // SCALE unit sur quelques items. Sur l'instance n=20/N=20 rep=0
    // (ccmk-song-pisinger-n20-m5-a50-we-r42-S20.txt), cet écart d'arrondi
    // faisait basculer un nœud-limite (ub calculé EXACTEMENT égal à
    // l'optimum, à 0.001 près) vers une contradiction CP immédiate en mode
    // GNN batché (bab_mkp_gecode_gnn_batched), éliminant silencieusement la
    // branche optimale et produisant un faux certificat OPTIMAL à 5130.397
    // au lieu du vrai 5166.381 — cf. mémoire
    // project_gnn_batched_false_optimal_bug_investigation. Confirmé par test
    // direct : SCALE=10000 corrige ce cas (job SLURM 16385256, 2026-07-16),
    // TF32 cuBLAS/cuDNN désactivé NE corrige PAS (écarté). Vérifié aussi que
    // SCALE=10000 ne fait pas déborder int32 sur tous les datasets
    // disponibles (headroom ≥ 5.9x sur obj_max et sur la somme de chaque
    // contrainte linéaire).
    static constexpr int SCALE = 10000;
    std::vector<int>                           c_int;   // c × SCALE
    std::vector<std::vector<std::vector<int>>> A_int;   // A × SCALE (arrondi)
    std::vector<std::vector<int>>              b_int;   // b × SCALE (floor, conservatif)
    int obj_max;   // borne supérieure sur obj (= Σ max(0, c_int[k])), pour le domaine de obj

    // ── Options SLD pour les appels dans more() ──────────────────────────────
    BoundMode bound_mode = BoundMode::SLD;  // SLD (exact) ou DLD (LP-relax)
    // Mode SLD uniquement (ignoré en DLD) : méthode utilisée pour ζ_i à
    // chaque nœud. false (défaut) = MKPProblemBnB (Dantzig + LP-relax par
    // dimension, méthode historique/prod). true = MKPProblemBnBSurrogate
    // (même ζ_i exact, borne de branchement surrogate — cf.
    // mkp_surrogate_bound.hpp) : comparaison expérimentale demandée pour
    // évaluer si une borne de branchement plus serrée réduit le temps ζ_i
    // dans le VRAI B&B (pas seulement le benchmark isolé bench_sld_subproblems).
    bool   use_surrogate_zeta = false;
    int    T_root     = 200;   // itérations sous-gradient au premier appel (λ=0, cold start)
    int    T_node     = 30;    // itérations sous-gradient aux nœuds suivants (warm-start)
    double rel_tol    = 1e-4;  // tolérance d'arrêt anticipé du sous-gradient
    int    tol_window = 30;    // fenêtre pour vérifier la tolérance

    // gnn_compare_bound : arrêt anticipé du sous-gradient À LA RACINE
    // (is_first) dès que la borne courante atteint cette valeur ABSOLUE
    // (même échelle que root_bound_value, i.e. fp + L1 — PAS juste L1) —
    // typiquement la borne déjà obtenue par un forward GNN root-only
    // (gnn_root/gnn_root_dld), passée via la variable d'env
    // GNN_COMPARE_BOUND (lue dans solve_gecode_bab_mkp, gecode_bab.cpp).
    // 1e18 (défaut) = désactivé (jamais atteint, comportement historique
    // inchangé). Ignoré aux nœuds suivants (T_node) — la comparaison au GNN
    // ne concerne que la racine. Ajouté suite au constat que le sous-gradient
    // SLD/ζ exact devient ~500x plus cher par itération à n=25 qu'à n=15
    // (cf. mémoire project_ccmkp_trillium_migration) : courir jusqu'à
    // convergence complète ou jusqu'à T_root=3000 est impraticable, alors
    // qu'on ne cherche qu'à savoir à QUELLE itération le sous-gradient
    // rattrape le GNN.
    double gnn_compare_bound = 1e18;

    // ── Mode patience ─────────────────────────────────────────────────────────
    // Si true, chaque appel more() aux nœuds (hors root) tourne jusqu'à ce que
    // la borne converge à patience_tol près (arrêt anticipé par rel_tol),
    // avec un plafond de patience_max_iters itérations.
    bool   patience_mode      = false;
    double patience_tol       = 0.1;    // tolérance de convergence (10^-1 par défaut)
    int    patience_max_iters = 500;    // plafond d'itérations en mode patience

    // ── Compteurs de statistiques (mutable : modifiés dans more() via const*) ─
    //
    // ub_tightened : nombre de fois où la borne SLD est strictement inférieure
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
    // Si > 0, on poste obj >= initial_lb_int dans le constructeur de MKPSpaceSLD.
    // Cela évite d'explorer des branches dont l'objectif ne peut dépasser une
    // solution déjà connue (typiquement : la solution gloutonne).
    int initial_lb_int = 0;

    mutable int more_calls          = 0;    // nombre total d'appels à more()
    mutable int ub_tightened        = 0;    // appels où la borne réduit obj.max()
    mutable int fails_from_bound    = 0;    // appels où la borne crée une contradiction directe
    mutable double best_open_ub     = 0.0;  // meilleure UB vue sur un nœud non-élagué
    mutable double root_bound_time_s = 0.0; // durée du cold start (premier appel more() où la borne est CALCULÉE, T_root iters)
    mutable double root_bound_value  = 0.0; // valeur ub calculée au premier appel more() où la borne est calculée ; 0.0 en mode REF (pas de borne).
                                             // = valeur à la racine SAUF si gnn_min_depth_frac > 0 (mode GNN) : dans ce cas
                                             // c'est la valeur au premier nœud atteignant le seuil de profondeur, pas la racine.
    mutable double t_bound_total_s   = 0.0; // temps total cumulé dans tous les appels more()

    // ── Diagnostic : coût du clonage d'espace (Space::copy(), un par nœud) ────
    // Isole le temps passé à copier NOTRE état (lambda_sld/lambda_dld/b_res/
    // fixed_mask), distinct du reste de t_bab_pure_s (clonage natif Gecode des
    // BoolVarArray/propagateurs + propagation CP + dispatch du BoundBrancher).
    mutable int64_t clone_count      = 0;   // nb d'appels à MKPSpaceSLD::copy()
    mutable double  t_clone_state_s  = 0.0; // temps cumulé à copier lambda/b_res/fixed_mask
    mutable double  t_clone_total_s  = 0.0; // temps cumulé du clonage COMPLET (base Gecode + notre état)

    // ── Statistiques par profondeur ───────────────────────────────────────────
    // Objectif : détecter une profondeur au-delà de laquelle calculer la borne
    // (SLD/DLD/GNN) ne provoque plus aucun fail, pour évaluer si on peut
    // arrêter de la recalculer/poster passé ce seuil (compromis coût/qualité).
    // Indexées par depth = nombre de variables x fixées au moment de l'appel
    // more() (0 = racine, n = feuille) — même définition que /*depth=*/ dans
    // le bloc GNN ci-dessous. Taille n+1, allouée à la construction. Non
    // peuplées en mode REF ni pour les appels GNN_ROOT court-circuités après
    // le premier (aucune borne calculée à ces appels — cf. more()).
    mutable std::vector<int64_t> nodes_by_depth;           // more() ayant atteint l'étape 1 (scan) à cette depth
    mutable std::vector<int64_t> ub_tightened_by_depth;    // sous-ensemble : borne resserre obj.max()
    mutable std::vector<int64_t> fails_from_bound_by_depth; // sous-ensemble : fail immédiat dû à la borne
    // Temps cumulé (s) passé à calculer+poster la borne à cette depth — même
    // "elapsed_bound" que t_bound_total_s, mais réparti par profondeur. Sert à
    // juger la RENTABILITÉ (coût/utilité), pas seulement l'utilité : une depth
    // sans fail mais où la borne est déjà quasi gratuite n'a pas le même
    // verdict qu'une depth sans fail où elle reste coûteuse (ex. GNN : le
    // coût par nœud ne décroît pas nécessairement avec la profondeur comme le
    // fait celui de SLD/DLD, dont le sous-problème résiduel rétrécit).
    mutable std::vector<double>  t_bound_by_depth;

    // ── Mode GNN (bound_mode == BoundMode::GNN) ───────────────────────────────
    // gnn_client n'est PAS possédé ici (durée de vie gérée par
    // solve_gecode_bab_mkp, qui outlive tous les clones d'espace — même
    // patron que ce struct lui-même vis-à-vis des MKPSpaceSLD). Interface
    // commune aux deux backends (IPC ou natif LibTorch) — cf. gnn_bound_backend.hpp.
    GNNBoundBackend* gnn_client = nullptr;
    // Seuil de profondeur relative (depth/n) en dessous duquel l'appel GNN est
    // sauté en mode GNN (comportement REF local pour CE nœud : aucune borne
    // calculée/postée, more_calls et nodes_by_depth restent comptabilisés).
    // Motivation empirique : sweep de robustesse job 16186510 (8 instances,
    // n∈{10,20} × N∈{20,50,100,500}) — fails_from_bound_by_depth == 0 pour
    // depth/n < ~0.5 dans 6/8 configs (les 2 exceptions, à N=500, dévient de
    // ±10 points sur des comptes de 1-2 fails — probablement du bruit
    // d'échantillonnage sur une seule instance). Défaut 0.0 = comportement
    // GNN historique inchangé (borne calculée à chaque nœud dès la racine).
    // Ignoré en mode GNN_ROOT (déjà gated par lambda_warmed, cf. more()).
    double gnn_min_depth_frac = 0.0;
    // Seuil sur un proxy LOCAL (par nœud, pas par profondeur) : gap_rel =
    // (obj.max() - obj.min()) / obj.max(), lu APRÈS propagation CP héritée
    // des ancêtres mais AVANT tout calcul de borne à ce nœud (donc gratuit,
    // aucun appel GNN nécessaire pour l'évaluer). L'appel GNN est sauté si
    // gap_rel > gnn_min_gap_frac (domaine encore trop large pour qu'une borne
    // avec l'écart typique du GNN — ~20% à la racine — puisse faire fail).
    // Motivation empirique (trace GNN_TRACE_DEBUG, n=20/N=100, job 16437043,
    // analyse offline) : gap_rel médian ≈5.1% sur les nœuds où la borne cause
    // un fail contre ≈9.6% sur les autres, séparation stable à profondeur
    // fixée (pas un artefact de profondeur) — cf. mémoire "gnn_local_proxy_gate".
    // Défaut 0.0 = comportement historique (borne calculée à chaque nœud).
    double gnn_min_gap_frac = 0.0;
    // Symétrique de gnn_min_depth_frac : seuil de profondeur relative
    // au-DESSUS duquel l'appel GNN est sauté (arrête de calculer/poster la
    // borne une fois "assez profond", au lieu de près de la racine). Motivation :
    // à n=20/N=100 (job 16206556), la majorité des nœuds more() ET des fails
    // imputables à la borne se concentrent aux profondeurs profondes
    // (depth/n ≈ 0.5-0.95), pas près de la racine — gnn_min_depth_frac=0.5 n'y
    // gagnait donc que 3-8% (la zone gatée pesait peu). Couper plutôt le fond
    // de l'arbre teste le compromis inverse : économiser le gros du coût de
    // borne (là où il y a le plus de nœuds) au prix de la perte du pruning là
    // où la borne s'avère le plus souvent utile. Défaut 1.0 = jamais sauté
    // (comportement historique, depth <= n toujours).
    double gnn_max_depth_frac = 1.0;
    // Si true, BoundBrancher::commit() ne calcule PLUS la borne lui-même
    // (fixation de la variable seulement) — c'est au driver (boucle DFS de
    // solve_gecode_bab_mkp) d'appeler explicitement more() APRÈS un premier
    // status() de propagation CP pure. Motivation (cf. revue niveau 1 point
    // A) : dans le comportement historique (false), more() — et donc le coût
    // GNN complet (graphe+forward+ζ) — est payé pour CHAQUE nœud AVANT que
    // la propagation CP native (contraintes de capacité réifiées, chance
    // constraint) ait la moindre chance de le faire échouer gratuitement ;
    // fails_from_cp (= fails − fails_from_bound) mesure combien de ces
    // appels sont gaspillés sur des nœuds que le CP seul aurait de toute
    // façon tués. gecode_bab_gnn_batched.cpp appliquait déjà ce découplage
    // (SimpleBrancher::commit() ne calcule aucune borne) mais jamais
    // reporté ici. Défaut false : AUCUN changement de comportement pour les
    // autres consommateurs de MKPSpaceSLD/BoundBrancher (gecode_bab_bucketed.cpp,
    // gecode_bab_collect.cpp), qui gardent le fusionnement commit()→more()
    // historique. Mis à true uniquement dans solve_gecode_bab_mkp.
    bool defer_bound_to_driver = false;
    // Décomposition du temps de borne en mode GNN, cumulée sur tous les
    // appels more() : construction de graphe, forward pass GNN "pur", calcul
    // de L1(λ) (inclut la résolution ζ exacte côté serveur Python), et
    // overhead IPC (= t_bound_total_s − les trois précédents), qui isole la
    // latence de communication du temps de calcul réel.
    mutable double t_gnn_graph_build_s   = 0.0;
    mutable double t_gnn_graph_features_s = 0.0;  // sous-ensemble de t_gnn_graph_build_s (backend natif)
    mutable double t_gnn_graph_edges_s    = 0.0;  // sous-ensemble de t_gnn_graph_build_s (backend natif)
    mutable double t_gnn_forward_s       = 0.0;
    mutable double t_gnn_bound_compute_s = 0.0;
    mutable double t_gnn_ipc_overhead_s  = 0.0;

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
        // Statistiques par profondeur : depth ∈ [0, n]
        nodes_by_depth.assign(n + 1, 0);
        ub_tightened_by_depth.assign(n + 1, 0);
        fails_from_bound_by_depth.assign(n + 1, 0);
        t_bound_by_depth.assign(n + 1, 0.0);

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
// GecodeSLDResult — statistiques de résolution retournées à l'appelant
// ═════════════════════════════════════════════════════════════════════════════

struct GecodeSLDResult {
    double      objective;     // valeur objective de la meilleure solution (non scalée)
    bool        optimal;       // true si optimalité prouvée (arbre épuisé)
    bool        feasible;      // true si au moins une solution faisable trouvée
    double      greedy_value;  // valeur de la solution gloutonne utilisée comme incumbent initial
    double      best_open_ub;  // meilleure UB SLD vue sur un nœud non-élagué (≥ objective si optimal)
    double      solve_time_s;  // temps total de résolution (construction + B&B)
    int64_t     nodes;              // nœuds totaux Gecode (engine.statistics().node)
    int64_t     fails;              // nœuds échoués (propagation + borne, au total)
    int64_t     fails_from_bound;   // sous-ensemble : fails causés directement par notre borne SLD
    int64_t     fails_from_cp;      // estimé = fails − fails_from_bound (propagation CP native)
    int64_t     incumbent_improvements; // nombre de fois où une meilleure solution a été trouvée
    int64_t     more_calls;           // appels totaux à more() (= branches explorées avec commit OK)
    int64_t     ub_tightened;         // sous-ensemble : borne SLD a réellement réduit obj.max()
    int64_t     depth;                // profondeur maximale atteinte dans l'arbre
    double      root_bound_time_s;    // temps du premier appel à more() (cold start, T_root iters)
    double      root_bound_value = 0.0; // valeur de la borne au premier appel more() (racine) ; 0.0 en mode REF
    double      t_bound_total_s;      // temps cumulé dans tous les appels more()
    double      t_bab_pure_s;         // temps B&B pur = t_total - t_bound_total
    int64_t     clone_count = 0;      // nb d'appels à MKPSpaceSLD::copy() (~1/nœud, parfois plus)
    double      t_clone_state_s = 0.0; // sous-ensemble de t_bab_pure_s : copie de lambda/b_res/fixed_mask
    double      t_clone_total_s = 0.0; // sous-ensemble de t_bab_pure_s : clonage complet (base + état)
    std::string status;               // "OPTIMAL" | "FEASIBLE" | "INFEASIBLE" | "TIMEOUT"

    // ── Mode GNN uniquement (0 sinon) — cf. MKPSharedData::t_gnn_*_s ──────────
    double      t_gnn_graph_build_s   = 0.0;
    double      t_gnn_graph_features_s = 0.0;
    double      t_gnn_graph_edges_s    = 0.0;
    double      t_gnn_forward_s       = 0.0;
    double      t_gnn_bound_compute_s = 0.0;
    double      t_gnn_ipc_overhead_s  = 0.0;
    // Pré-chauffage TorchScript (backend natif uniquement, 0 sinon) : temps
    // passé à compiler toutes les formes n_g=1..n avant de démarrer le B&B
    // (cf. commentaire dans solve_gecode_bab_mkp). Inclus dans solve_time_s
    // (le chrono t0 démarre avant), mais PAS dans t_bab_pure_s ni
    // t_bound_total_s — sinon il serait attribué à tort au dispatch
    // brancher/CP.
    double      t_gnn_warmup_s        = 0.0;

    // ── Statistiques par profondeur — cf. MKPSharedData::*_by_depth ──────────
    // Taille n+1, index = depth (nombre de x fixés, 0 = racine). Vide en mode
    // REF. Sert à repérer la depth au-delà de laquelle la borne ne provoque
    // plus aucun fail (fails_from_bound_by_depth[d] == 0).
    std::vector<int64_t> nodes_by_depth;
    std::vector<int64_t> ub_tightened_by_depth;
    std::vector<int64_t> fails_from_bound_by_depth;
    std::vector<double>  t_bound_by_depth;  // s cumulées, mêmes index — cf. MKPSharedData::t_bound_by_depth
};

// ═════════════════════════════════════════════════════════════════════════════
// MKPSpaceSLD — espace de recherche Gecode pour le CCMKP + borne SLD
// ═════════════════════════════════════════════════════════════════════════════
//
// Hérite de IntMaximizeSpace : Gecode pilote automatiquement la maximisation
// (il maintient une borne inférieure = meilleure solution, et élaguer les
// branches dont l'objectif ne peut dépasser cette borne).

class MKPSpaceSLD : public IntMaximizeSpace {
public:
    BoolVarArray x;    // x[k] ∈ {0,1} : item k sélectionné ou non
    BoolVarArray z;    // z[i] ∈ {0,1} : scénario i satisfait ou non
    IntVar       obj;  // objectif scalé = Σ_k c_int[k] * x[k]  (÷ SCALE pour la vraie valeur)

    // Pointeur vers les données partagées (read-only, non cloné).
    const MKPSharedData* data;

    // Multiplicateurs Lagrangiens — un seul est utilisé selon bound_mode.
    // SLD  : lambda_sld  (N × n)      — hérité/mis à jour par more() en mode SLD
    // DLD : lambda_dld (N × m × n)  — hérité/mis à jour par more() en mode DLD
    // L'autre champ reste vide (Vec2d{} ou Vec3d{}) → aucun surcoût mémoire.
    Vec2d lambda_sld;    // (N, n)      — mode SLD
    Vec3d lambda_dld;   // (N, m, n)   — mode DLD

    // false → premier appel : cold start (T_root iters) ; true → warm-start (T_node iters)
    bool lambda_warmed = false;

    // ── État résiduel incrémental (évite de reconstruire b_adj depuis data->b
    // en resommant TOUT fixed_one à chaque nœud — cf. discussion coût more()) ─
    //
    // b_res[i][j]   : capacité résiduelle du scénario i, contrainte j, après
    //                 déduction des items fixés à 1 (hérité/mis à jour par more()).
    // fixed_mask[k] : 0 = pas encore comptabilisé, 1 = déjà comptabilisé (fixé à
    //                 0 ou 1) — sert à détecter, à chaque appel, UNIQUEMENT les
    //                 variables nouvellement fixées depuis le nœud parent (delta,
    //                 généralement 1, parfois plus si la propagation CP a fixé
    //                 des variables en plus du branchement explicite).
    // fp_running    : profit fixé (Σ c[k] pour k fixé à 1), mis à jour au même
    //                 rythme que b_res au lieu d'être resommé à chaque more().
    Vec2d                b_res;
    std::vector<int8_t>  fixed_mask;
    double               fp_running = 0.0;

    // ─────────────────────────────────────────────────────────────────────────
    // BoundBrancher — brancher custom qui déclenche le bornage SLD
    // ─────────────────────────────────────────────────────────────────────────
    //
    // Gecode utilise des "branchers" pour choisir quelle variable instancier et
    // avec quelle valeur. Ce brancher gère uniquement les variables x[k].
    //
    // Rôle principal : après chaque commit (fixation de x[k] = v), appeler
    // more() sur l'espace pour calculer la borne SLD et la poster comme
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
            const auto& order = static_cast<MKPSpaceSLD&>(home).data->branch_order;
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
        // pour calculer la borne SLD sur le sous-problème résiduel et la poster
        // comme contrainte sur obj. Si cette contrainte est infaisable (borne < LB),
        // Gecode détecte ES_FAILED lors de la propagation suivante.
        //
        // SAUF si data->defer_bound_to_driver == true (cf. commentaire sur ce
        // champ, MKPSharedData) : dans ce cas, on fixe la variable et on
        // s'arrête là — c'est au driver (boucle DFS) d'appeler more()
        // lui-même, APRÈS avoir laissé la propagation CP pure (status())
        // une première chance de faire échouer le nœud sans payer le coût de
        // la borne. Défaut false : comportement historique inchangé pour les
        // autres consommateurs (gecode_bab_bucketed.cpp, gecode_bab_collect.cpp).
        ExecStatus commit(Space& home, const Choice& c, unsigned int a) override {
            const PosVal& pv = static_cast<const PosVal&>(c);
            int pos = pv.pos;
            int val = (a == 0) ? 1 : 0;

            // Tente de fixer vars[pos] = val ; retourne ES_FAILED si contradiction
            ExecStatus es = me_failed(vars[pos].eq(home, val))
                            ? ES_FAILED : ES_OK;

            auto& sp = static_cast<MKPSpaceSLD&>(home);
            // Ne calcule la borne que si la fixation est cohérente avec les contraintes
            if (es == ES_OK && !sp.data->defer_bound_to_driver)
                sp.more();

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

    explicit MKPSpaceSLD(const MKPSharedData* d)
        : x(*this, d->n, 0, 1),
          z(*this, d->N, 0, 1),
          obj(*this, 0, d->obj_max),
          data(d),
          lambda_sld (d->bound_mode == BoundMode::SLD
                       ? Vec2d(d->N, Vec1d(d->n, 0.0)) : Vec2d{}),
          lambda_dld(d->bound_mode == BoundMode::DLD
                       ? Vec3d(d->N, Vec2d(d->m, Vec1d(d->n, 0.0))) : Vec3d{}),
          b_res(d->b),                    // racine : aucun item fixé, capacité pleine
          fixed_mask(d->n, 0)
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

    MKPSpaceSLD(MKPSpaceSLD& s)
        : IntMaximizeSpace(s), data(s.data),
          lambda_warmed(s.lambda_warmed)
    {
        x.update(*this, s.x);
        z.update(*this, s.z);
        obj.update(*this, s.obj);

        // Diagnostic : isoler le temps de copie de NOTRE état (listes
        // d'initialisation remplacées par des affectations chronométrées dans
        // le corps — coût identique, juste mesuré séparément du reste du
        // clonage Gecode natif).
        using Clock = std::chrono::steady_clock;
        auto t_clone_start = Clock::now();
        lambda_sld  = s.lambda_sld;
        lambda_dld = s.lambda_dld;
        b_res        = s.b_res;
        fixed_mask   = s.fixed_mask;
        fp_running   = s.fp_running;
        data->t_clone_state_s += std::chrono::duration<double>(
            Clock::now() - t_clone_start).count();
        data->clone_count++;
    }

    // copy() : appelé par Gecode pour cloner cet espace. Doit retourner un nouvel
    // espace construit avec le constructeur de clonage.
    // Diagnostic : chronomètre le clonage COMPLET (base IntMaximizeSpace + nos
    // membres), pour isoler par soustraction ce qui n'est PAS notre état
    // (t_clone_total_s - t_clone_state_s = clonage natif Gecode des
    // BoolVarArray/IntVar/propagateurs) du reste de t_bab_pure_s (dispatch
    // BoundBrancher::choice()/commit() hors more(), propagation CP déclenchée
    // par le moteur de recherche).
    Space* copy() override {
        using Clock = std::chrono::steady_clock;
        auto t0 = Clock::now();
        Space* s = new MKPSpaceSLD(*this);
        data->t_clone_total_s += std::chrono::duration<double>(Clock::now() - t0).count();
        return s;
    }

    // cost() : retourne la variable que Gecode doit maximiser.
    // IntMaximizeSpace utilise cette variable pour maintenir la borne inférieure
    // (incumbent) et élaguer les branches dont obj ne peut la dépasser.
    IntVar  cost()  const override { return obj; }

    // ─────────────────────────────────────────────────────────────────────────
    // more() — calcul de la borne SLD et propagation dans l'espace courant
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
    //   4. Appeler solve_sld() sur ce sous-problème pour obtenir SLD*.
    //   5. UB = fp + SLD*
    //   6. Poster  obj ≤ ⌈UB × SCALE⌉  dans l'espace courant.
    //      Si cette contrainte est contradictoire avec le domaine actuel de obj,
    //      la propagation Gecode détectera un échec → branche élagée.
    //
    // Note : pas de warm-start ici (lambda_init = {}). Chaque appel repart de λ=0.
    // Conséquence : la borne est moins précise que dans bab_mkp.cpp (qui warm-start),
    // mais l'implémentation est plus simple.
    //
    // Retourne true si une contrainte obj ≤ ub_int a réellement été postée
    // (donc si une repropagation via status() a un intérêt), false sinon —
    // REF, GNN_ROOT déjà réchauffé, feuille en x, gnn_min_depth_frac non
    // atteint, ou borne calculée mais qui n'améliore pas obj.max(). Utilisé
    // par le driver quand data->defer_bound_to_driver == true (cf.
    // solve_gecode_bab_mkp) pour ne re-status() que si nécessaire ; ignoré
    // (valeur jetée) quand more() est encore appelé depuis
    // BoundBrancher::commit() (defer_bound_to_driver == false).

    bool more() {
        // Mode REF : squelette SLD nu — même heuristique de branchement, même
        // incumbent glouton, même moteur DFS que SLD/DLD, mais AUCUNE
        // reconstruction de sous-problème et AUCUNE borne calculée/postée.
        // Court-circuit avant même le scan des variables (Étape 1) : ce scan
        // et la mise à jour incrémentale de b_res/fp_running n'existent que
        // pour servir le calcul de borne, donc n'ont pas lieu d'être ici.
        if (data->bound_mode == BoundMode::REF) {
            data->more_calls++;
            return false;
        }

        // Mode GNN_ROOT / GNN_ROOT_DLD : la borne n'est calculée qu'UNE FOIS
        // par lignée (au premier appel more(), cf. lambda_warmed hérité par
        // clonage — même notion de "premier appel" que is_first plus bas /
        // T_root vs T_node). Une fois lambda_warmed==true, on se comporte
        // comme REF : même court-circuit AVANT le scan des variables (Étape 1),
        // qui n'a d'utilité que pour alimenter un calcul de borne ultérieur.
        if ((data->bound_mode == BoundMode::GNN_ROOT ||
             data->bound_mode == BoundMode::GNN_ROOT_DLD) && lambda_warmed) {
            data->more_calls++;
            return false;
        }

        const int n = data->n;
        const int m = data->m;
        const int N = data->N;

        // Étape 1 : identifier l'état courant des variables x, ET mettre à jour
        // l'état résiduel incrémental (b_res, fp_running) pour les SEULES
        // variables nouvellement fixées depuis le nœud parent (delta — en
        // général 1 seule, le branchement explicite ; parfois plus si la
        // propagation CP a fixé des variables en plus). Remplace la
        // reconstruction complète de b_adj/fp depuis data->b/data->c à chaque
        // nœud (coût O(N·m·n) auparavant, resommé sur TOUT fixed_one) par un
        // coût O(N·m·|delta|) — b_res/fixed_mask sont clonés d'un nœud à l'autre
        // comme lambda_sld/lambda_dld, donc portent déjà l'état du parent.
        //
        // x_mask sert uniquement au mode GNN (0=fixé_0, 1=fixé_1, 2=libre, sur
        // l'instance COMPLÈTE) — coût O(n) négligeable, on le construit toujours
        // pour ne pas complexifier cette boucle avec un branchement par mode.
        std::vector<int> free_vars;   // indices des x[k] non encore fixés
        std::vector<int8_t> x_mask(n, 2);
        free_vars.reserve(n);

        for (int k = 0; k < n; ++k) {
            if (!x[k].assigned()) {
                free_vars.push_back(k);
                continue;
            }
            int val = x[k].val();
            x_mask[k] = static_cast<int8_t>(val);

            if (fixed_mask[k] != 0) continue;  // déjà comptabilisé à un appel antérieur
            fixed_mask[k] = 1;
            if (val == 1) {
                for (int i = 0; i < N; ++i)
                    for (int j = 0; j < m; ++j)
                        b_res[i][j] -= data->A[i][j][k];
                fp_running += data->c[k];
            }
        }

        // Comptabilise cet appel (même les feuilles où on sort tôt)
        data->more_calls++;

        // À une feuille (tout fixé), pas de borne à calculer : l'objectif est exact.
        if (free_vars.empty()) return false;

        const int n_free = static_cast<int>(free_vars.size());

        // depth = nombre de variables x déjà fixées (0 = racine). Utilisé pour
        // la borne GNN (query()) ET pour les statistiques par profondeur
        // ci-dessous — un seul calcul, une seule définition.
        const int depth = n - n_free;
        data->nodes_by_depth[depth]++;

        // Étape 2 : profit des items déjà sélectionnés — maintenu incrémentalement
        // dans fp_running (étape 1) ; plus besoin de resommer fixed_one ici.
        double fp = fp_running;

        // ── Mode GNN / GNN_ROOT : borne one-shot, AVANT de construire le
        // sous-problème réduit (Étape 3) — celui-ci ne sert qu'aux modes
        // SLD/DLD, le construire quand même ici gaspillerait
        // O(N·m·n_free) par nœud pour rien. Un seul aller-retour vers le
        // serveur Python persistant (data->gnn_client), pas de warm-start ni
        // de raffinement itératif : la borne prédite est postée telle
        // quelle. x_mask porte sur l'instance complète — le serveur
        // reconstruit lui-même le sous-problème résiduel depuis ce masque
        // (cf. build_graph côté Python, qui accepte un masque plutôt qu'un
        // rétrécissement).
        // En GNN_ROOT, ce bloc n'est atteint que lorsque lambda_warmed était
        // encore false (garanti par le court-circuit ci-dessus) : c'est donc
        // toujours le premier (et dernier) appel de la lignée.
        // Mode GNN uniquement : sous le seuil de profondeur relative, on saute
        // l'appel GNN pour ce nœud (cf. doc sur MKPSharedData::gnn_min_depth_frac
        // — les mesures existantes montrent que la borne n'y provoque
        // pratiquement jamais de fail immédiat). nodes_by_depth est déjà
        // comptabilisé ci-dessus ; aucune borne n'est calculée/postée ici, donc
        // ni t_bound_by_depth ni ub_tightened_by_depth ne bougent pour cette depth.
        if (data->bound_mode == BoundMode::GNN &&
            depth < static_cast<int>(std::ceil(data->gnn_min_depth_frac * n))) {
            return false;
        }

        // Symétrique : sous mode GNN, au-delà du seuil gnn_max_depth_frac, on
        // arrête de calculer/poster la borne (cf. doc sur
        // MKPSharedData::gnn_max_depth_frac ci-dessus). Même comptabilisation
        // que le seuil min (nodes_by_depth déjà incrémenté, aucune borne
        // calculée ici).
        if (data->bound_mode == BoundMode::GNN &&
            depth > static_cast<int>(std::floor(data->gnn_max_depth_frac * n))) {
            return false;
        }

        // Mode GNN uniquement : proxy local gap_rel = (obj.max()-obj.min())/obj.max(),
        // disponible sans coût (obj déjà propagé par le CP hérité des ancêtres,
        // avant tout calcul de borne à CE nœud). Si le domaine est encore trop
        // large, on saute l'appel GNN — cf. doc sur MKPSharedData::gnn_min_gap_frac.
        if (data->bound_mode == BoundMode::GNN && data->gnn_min_gap_frac > 0.0) {
            const double gap_rel = static_cast<double>(obj.max() - obj.min()) /
                                    static_cast<double>(obj.max());
            if (gap_rel > data->gnn_min_gap_frac) {
                return false;
            }
        }

        if (data->bound_mode == BoundMode::GNN || data->bound_mode == BoundMode::GNN_ROOT ||
            data->bound_mode == BoundMode::GNN_ROOT_DLD) {
            using Clock = std::chrono::steady_clock;
            auto t_bound_start = Clock::now();
            const bool is_first = !lambda_warmed;

            GNNCallTiming timing;
            // b_res (this->b_res) est déjà à jour à ce point (maintenu
            // incrémentalement à l'Étape 1 ci-dessus) — le transmettre
            // directement au backend au lieu de le laisser recalculer depuis
            // x_mask (cf. revue niveau 1 point B : coût O(N·m·|F1|) redondant,
            // croissant avec la profondeur, qui expliquait pourquoi le coût
            // par nœud du mode GNN ne décroissait pas avec la profondeur
            // contrairement à SLD/DLD).
            double ub_relative = data->gnn_client->query(x_mask, b_res, fp, /*depth=*/depth, timing);
            double ub = fp + ub_relative;

            // Debug : trace nœud par nœud (diagnostic bug faux-optimal batché,
            // cf. mémoire "gnn_batched_robustness_sweep_result") — opt-in via
            // GNN_TRACE_DEBUG, aucun coût si absent. Format CSV parseable :
            // TRACE,<x_mask complet>,<depth>,<fp>,<ub_relative>,<ub>,<obj_min>,<obj_max>
            // obj_min/obj_max : domaine (unités SCALE) juste APRÈS l'appel GNN
            // mais AVANT le rel(obj <= ub_int) ci-dessous — reflète donc l'état
            // obtenu par la seule propagation CP héritée des ancêtres, utilisé
            // pour tester un proxy local (gap = obj_max - obj_min) candidat à
            // décider si l'appel GNN vaut le coût (cf. mémoire "gnn_local_proxy_gate").
            if (std::getenv("GNN_TRACE_DEBUG")) {
                std::string mstr; mstr.reserve(n);
                for (int k = 0; k < n; ++k) mstr += static_cast<char>('0' + x_mask[k]);
                std::fprintf(stderr, "TRACE,%s,%d,%.6f,%.6f,%.6f,%d,%d\n",
                             mstr.c_str(), depth, fp, ub_relative, ub, obj.min(), obj.max());
            }

            data->t_gnn_graph_build_s   += timing.t_graph_build_s;
            data->t_gnn_graph_features_s += timing.t_graph_features_s;
            data->t_gnn_graph_edges_s    += timing.t_graph_edges_s;
            data->t_gnn_forward_s       += timing.t_forward_s;
            data->t_gnn_bound_compute_s += timing.t_bound_compute_s;

            double elapsed_bound = std::chrono::duration<double>(
                Clock::now() - t_bound_start).count();
            data->t_gnn_ipc_overhead_s += elapsed_bound -
                (timing.t_graph_build_s + timing.t_forward_s + timing.t_bound_compute_s);

            data->t_bound_total_s += elapsed_bound;
            data->t_bound_by_depth[depth] += elapsed_bound;
            if (is_first) { data->root_bound_time_s = elapsed_bound; data->root_bound_value = ub; }
            lambda_warmed = true;

            if (ub > data->best_open_ub) data->best_open_ub = ub;
            int ub_int = static_cast<int>(std::ceil(ub * data->SCALE));
            const bool posted = ub_int < obj.max();
            if (posted) {
                data->ub_tightened++;
                data->ub_tightened_by_depth[depth]++;
                if (ub_int < obj.min()) {
                    data->fails_from_bound++;
                    data->fails_from_bound_by_depth[depth]++;
                }
                rel(*this, obj <= ub_int);
            }
            return posted;
        }

        // Étape 3 : construire le sous-problème réduit (flottant, pour SLD/DLD).
        // Chrono démarré ICI (avant la reconstruction O(N·m·n_free) de A_free/b_adj),
        // pas seulement autour de solve_sld/solve_dld : cette reconstruction est un
        // coût du calcul de la borne (l'équivalent existe côté GNN, dans build_graph
        // sur le serveur Python, et EST compté dans t_gnn_graph_build_s) — l'ancien
        // placement du chrono (juste avant l'étape 4) la comptait à tort dans
        // t_bab_pure_s, gonflant artificiellement le "B&B pur" de SLD/DLD par
        // rapport au mode GNN.
        using Clock = std::chrono::steady_clock;
        auto t_bound_start = Clock::now();

        Vec1d c_free(n_free);
        for (int kf = 0; kf < n_free; ++kf)
            c_free[kf] = data->c[free_vars[kf]];

        // Sous-matrice A restreinte aux variables libres — reste O(N·m·n_free) :
        // pas encore résolu par l'incrément 1 (nécessiterait de passer un accès
        // indexé à solve_sld/solve_dld plutôt qu'un tableau pré-empaqueté ;
        // laissé pour un incrément séparé si le gain ci-dessous ne suffit pas).
        Vec3d A_free(N, Vec2d(m, Vec1d(n_free, 0.0)));
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < m; ++j)
                for (int kf = 0; kf < n_free; ++kf)
                    A_free[i][j][kf] = data->A[i][j][free_vars[kf]];

        // b_adj == b_res : déjà à jour depuis l'étape 1 (mise à jour incrémentale),
        // plus de reconstruction depuis data->b + somme sur fixed_one ici.
        const Vec2d& b_adj = b_res;

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

        double ub;  // borne supérieure retournée par le solveur Lagrangien choisi

        if (data->bound_mode == BoundMode::SLD) {
            // ── Mode SLD : ζᵢ exact (B&B multi-dim), λ ∈ R^{N×n} ─────────────
            Vec2d lambda_init(N, Vec1d(n_free, 0.0));
            if (!is_first)
                for (int i = 0; i < N; ++i)
                    for (int kf = 0; kf < n_free; ++kf)
                        lambda_init[i][kf] = lambda_sld[i][free_vars[kf]];

            // Choix de la méthode ζ_i, cf. doc sur MKPSharedData::use_surrogate_zeta.
            // solve_sld() dispatche via SLDProblem& (virtuel) : polymorphisme
            // requis pour choisir le type concret à l'exécution sans dupliquer
            // l'appel solve_sld ci-dessous.
            std::unique_ptr<MKPProblem> prob_ptr = data->use_surrogate_zeta
                ? std::unique_ptr<MKPProblem>(std::make_unique<MKPProblemBnBSurrogate>(
                      c_free, data->p, A_free, b_adj, data->epsilon))
                : std::unique_ptr<MKPProblem>(std::make_unique<MKPProblemBnB>(
                      c_free, data->p, A_free, b_adj, data->epsilon));
            // log_interval=1 UNIQUEMENT à la racine (is_first) : res.convergence
            // porte alors la trace complète L1(t) pour t=0..T_root-1, dumpée
            // ci-dessous sous ROOT_SUBGRAD_TRACE (comparaison au forward GNN
            // root-only, cf. mémoire project_ccmkp_trillium_migration). Aux
            // nœuds suivants (warm-start, T_node), log_interval reste à 0
            // (comportement de production inchangé — pas de coût de logging
            // répété à chaque nœud de l'arbre).
            // target_relative : cf. doc MKPSharedData::gnn_compare_bound —
            // uniquement à la racine (is_first), converti de l'échelle
            // absolue (fp + L1) vers l'échelle relative (L1 seule) attendue
            // par solve_sld. Désactivé (1e18) aux nœuds suivants.
            double target_relative = is_first ? (data->gnn_compare_bound - fp) : 1e18;
            SLDResult res = solve_sld(*prob_ptr, T_iters, is_first ? 1 : 0, false, -1.0,
                                        node_tol, data->tol_window, -1e18,
                                        lambda_init, -1.0, 50, target_relative);

            for (int i = 0; i < N; ++i)
                for (int kf = 0; kf < n_free; ++kf)
                    lambda_sld[i][free_vars[kf]] = res.lambda_star[i][kf];
            ub = fp + res.SLD_star;

            // Trace opt-in (ROOT_SUBGRAD_TRACE=1) : best_upper(t) = fp + min
            // des L1 vus jusqu'à l'itération t — même quantité que
            // root_bound_value, mais à CHAQUE itération plutôt que seulement à
            // la fin. Sert à trouver t* = min{t : best_upper(t) <= borne_GNN
            // root} en post-traitement (aucun coût si la variable d'env est
            // absente).
            if (is_first && std::getenv("ROOT_SUBGRAD_TRACE")) {
                double running_best = 1e18;
                for (size_t t = 0; t < res.convergence.size(); ++t) {
                    running_best = std::min(running_best, res.convergence[t]);
                    std::fprintf(stderr, "SUBGRAD_TRACE,sld,%zu,%.6f\n",
                                 t + 1, fp + running_best);
                }
            }

        } else {
            // ── Mode DLD : ζᵢⱼ LP-relax 1D, λ ∈ R^{N×m×n} ───────────────────
            Vec3d lambda_init(N, Vec2d(m, Vec1d(n_free, 0.0)));
            if (!is_first)
                for (int i = 0; i < N; ++i)
                    for (int j = 0; j < m; ++j)
                        for (int kf = 0; kf < n_free; ++kf)
                            lambda_init[i][j][kf] = lambda_dld[i][j][free_vars[kf]];

            MKPDLDProblem prob(c_free, data->p, A_free, b_adj, data->epsilon);
            // log_interval, target_relative : cf. commentaires symétriques côté SLD ci-dessus.
            double target_relative = is_first ? (data->gnn_compare_bound - fp) : 1e18;
            DLDResult res = solve_dld(prob, T_iters, is_first ? 1 : 0, false, -1.0,
                                          node_tol, data->tol_window, -1e18,
                                          lambda_init, target_relative);

            for (int i = 0; i < N; ++i)
                for (int j = 0; j < m; ++j)
                    for (int kf = 0; kf < n_free; ++kf)
                        lambda_dld[i][j][free_vars[kf]] = res.lambda_star[i][j][kf];
            ub = fp + res.DLD_star;

            // Trace opt-in : cf. commentaire symétrique côté SLD ci-dessus.
            if (is_first && std::getenv("ROOT_SUBGRAD_TRACE")) {
                double running_best = 1e18;
                for (size_t t = 0; t < res.convergence.size(); ++t) {
                    running_best = std::min(running_best, res.convergence[t]);
                    std::fprintf(stderr, "SUBGRAD_TRACE,dld,%zu,%.6f\n",
                                 t + 1, fp + running_best);
                }
            }
        }

        double elapsed_bound = std::chrono::duration<double>(
            Clock::now() - t_bound_start).count();
        data->t_bound_total_s += elapsed_bound;
        data->t_bound_by_depth[depth] += elapsed_bound;
        if (is_first) {
            data->root_bound_time_s = elapsed_bound;
            data->root_bound_value  = ub;
        }

        lambda_warmed = true;
        if (ub > data->best_open_ub) data->best_open_ub = ub;  // mise à jour de la meilleure UB vue
        int    ub_int = static_cast<int>(std::ceil(ub * data->SCALE));

        // La borne est utile uniquement si elle réduit le domaine courant de obj.
        // obj.max() est la valeur maximale encore atteignable selon Gecode.
        const bool posted = ub_int < obj.max();
        if (posted) {
            data->ub_tightened++;
            data->ub_tightened_by_depth[depth]++;

            // Si ub_int est déjà en dessous du minimum de obj, la contradiction est
            // immédiate : ce fail vient directement de notre borne, pas de la propagation CP.
            if (ub_int < obj.min()) {
                data->fails_from_bound++;
                data->fails_from_bound_by_depth[depth]++;
            }

            rel(*this, obj <= ub_int);
        }
        // Si ub_int >= obj.max(), la borne ne change rien : on ne la poste pas.
        return posted;
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
// solve_gecode_sld_mkp — point d'entrée public
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
//   T_node    : itérations de sous-gradient SLD par nœud (défaut 30)

// gnn_checkpoint_path : requis si bound_mode == BoundMode::GNN (chemin vers le
//   .pt) — lève std::invalid_argument si vide dans ce mode.
// gnn_backend / gnn_variant : seul le backend natif "native" (LibTorch en-process,
//   TorchScript) avec le graphe "sd_edit" est disponible (gnn_variant ignoré en
//   GNN_ROOT_DLD). gnn_python_exe / gnn_server_script : conservés pour la
//   compatibilité de signature, inutilisés.
// gnn_min_depth_frac : mode GNN uniquement (ignoré ailleurs, y compris
//   GNN_ROOT) — saute l'appel GNN tant que depth/n < gnn_min_depth_frac (cf.
//   doc sur MKPSharedData::gnn_min_depth_frac dans gecode_bab.hpp). Défaut
//   0.0 = borne calculée à chaque nœud dès la racine (comportement historique).
// gnn_min_gap_frac : mode GNN uniquement — saute l'appel GNN si le proxy local
//   gap_rel = (obj.max()-obj.min())/obj.max() dépasse ce seuil (cf. doc sur
//   MKPSharedData::gnn_min_gap_frac). Défaut 0.0 = jamais sauté.
// gnn_max_depth_frac : symétrique de gnn_min_depth_frac — saute l'appel GNN
//   dès que depth/n > gnn_max_depth_frac (cf. doc sur
//   MKPSharedData::gnn_max_depth_frac). Défaut 1.0 = jamais sauté.
// use_surrogate_zeta : mode SLD uniquement — cf. doc sur
//   MKPSharedData::use_surrogate_zeta. Défaut false = comportement historique
//   (MKPProblemBnB).
GecodeSLDResult solve_gecode_bab_mkp(
    int n, int m, int N, double epsilon,
    const Vec1d& c, const Vec1d& p,
    const Vec3d& A, const Vec2d& b,
    double    time_lim      = 300.0,
    int       T_root        = 200,
    int       T_node        = 30,
    BoundMode bound_mode    = BoundMode::SLD,
    bool      patience_mode = false,
    double    patience_tol  = 0.1,
    std::string gnn_checkpoint_path = "",
    std::string gnn_python_exe      = "python3",
    std::string gnn_device          = "cpu",
    std::string gnn_server_script   = "gnn/gnn_inference_server.py",
    std::string gnn_backend         = "native",
    std::string gnn_variant         = "sd_edit",
    double gnn_min_depth_frac       = 0.0,
    double gnn_min_gap_frac         = 0.0,
    double gnn_max_depth_frac       = 1.0,
    bool   use_surrogate_zeta       = false);
