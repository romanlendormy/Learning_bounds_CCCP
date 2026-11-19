#pragma once
/*
 * ccpo_sld.hpp — Borne lagrangienne SLD pour CCPO (décomposition par scénario).
 *
 * Fidèle au patron MKP (solver_generic/sld_base.hpp, sld_solver.cpp,
 * mkp_problem_bnb.hpp) : mêmes rôles pour θ_j/ζ_j/δ_j, même master (sac-à-dos
 * min-coût sur z), même boucle de sous-gradient. Le point de rupture face à
 * MKP, discuté et tranché avant ce fichier :
 *
 *   - MKP : ζ_i est un sac-à-dos 0/1 LINÉAIRE -> B&B avec borne LP par
 *     dimension (Dantzig), rapide.
 *   - CCPO : ζ_j = max_{q∈D} cost'q  s.c.  q'Σ_j q ≤ B  -- QUADRATIQUE.
 *
 * ζ_j DOIT être résolu EXACTEMENT sur le même ensemble entier D que θ_j (pas
 * de relaxation continue retournée telle quelle) : sinon rien ne garantit
 * ζ_j(λ) ≤ θ_j(λ), l'inégalité δ_j=θ_j-ζ_j≥0 casse, et la reformulation en
 * sac-à-dos min-coût (qui suppose des coûts δ_j≥0) n'est plus valide -- la
 * borne L(λ) perdrait sa garantie L(λ)≥v*. Décision actée en conversation :
 * ζ_j exact, SANS budget de temps/nœuds interne, quoi qu'il en coûte.
 *
 * La relaxation continue (ellipsoïde décalé, KKT) sert UNIQUEMENT de borne
 * d'élagage À L'INTÉRIEUR du B&B qui résout ζ_j exactement -- jamais comme
 * valeur retournée -- exactement le rôle de la borne LP dans MKPProblemBnB.
 *
 * CORRECTIF -- terme croisé fixé×libre manquant : à un nœud
 * résiduel (≥1 actif déjà fixé ACTIF par l'arbre englobant), la vraie
 * contrainte est q_free'Sigma_j q_free + 2·r_j'q_free ≤ B_res, où r_j[a] =
 * Σ_{i fixé actif} v_i·Sigma_j[i][free_vars[a]] (terme LINÉAIRE, signe
 * quelconque). L'ancienne version de more()/zeta_solve ne retirait de B que
 * la partie fixé-fixé (constante) et omettait ENTIÈREMENT r_j -- une borne
 * potentiellement INVALIDE (trop optimiste si r_j'q_free>0 pour le q optimal)
 * dès qu'au moins une variable est fixée active. zeta_solve reçoit maintenant
 * r_init (= r_j) et l'utilise À LA FOIS dans la borne d'élagage (déjà câblée
 * pour un terme linéaire générique) ET dans la vérification EXACTE à la
 * feuille (qui ne l'utilisait pas du tout auparavant) -- cf. zeta_solve.
 *
 * Choix de conception (pas celui envisagé en conversation) : la borne
 * d'élagage utilise directement la sous-matrice DENSE de Sigma_int (celle que
 * REF vérifie), factorisée par Cholesky à la volée -- PAS une reconstruction
 * séparée via (beta_j, sigma_f_j). Différence cruciale : Sigma_int est arrondi
 * ENTRÉE PAR ENTRÉE à la génération (§3 de la spec), donc légèrement différent
 * d'une factorisation rang-1 idéalisée reconstruite a posteriori (écart
 * mesuré : λ2/λ1 < 5%, cf. ccpo_stochastic tests). Utiliser Sigma_int dense
 * directement pour LA BORNE garantit qu'elle reste valide PAR RAPPORT À CE
 * QUE REF VÉRIFIE RÉELLEMENT -- pas de risque d'élaguer à tort une branche
 * qui serait vraiment faisable. À n≤30, une factorisation Cholesky par nœud
 * (O(n³) ≤ 27000 flops) est négligeable ; pas besoin de Sherman-Morrison ni
 * d'étendre scenarios.txt avec beta_int/sigma_f_int.
 */

#include "ccpo_presolve_exact.hpp"
#include <algorithm>
#include <array>
#include <functional>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <omp.h>
#include <vector>

using SigmaTriplet3 = std::array<int, 3>;  // (i, k, val), i<=k -- même type que gecode_bab_ccpo.hpp

// ═════════════════════════════════════════════════════════════════════════
// theta_solve — max_{q∈D} cost'q, SANS contrainte de risque (exact, entier)
// ═════════════════════════════════════════════════════════════════════════
//
// D = { q∈Z^n : Σq_i=Q, |supp(q)|≤k, q_i∈{0}∪{L,...,U} }.
//
// Argument d'échange (cf. discussion) : à taille de support s fixée,
// l'allocation optimale est du "water-filling" -- L à chacun, puis le budget
// restant (Q-sL) va en priorité au coût le plus élevé du support jusqu'à U,
// puis au suivant. Le meilleur support de taille s est le top-s par coût
// (échanger un membre du support contre un item hors-support de coût
// supérieur ne peut qu'améliorer). On essaie toutes les tailles s valides.

struct ThetaResult {
    double value;
    std::vector<int> q;  // (n,), solution entière atteignant `value`
    // false ssi AUCUNE taille de support n'est structurellement réalisable
    // (peut arriver sur un sous-problème RÉSIDUEL : k_res ne diminue que sur
    // les fixations ACTIVES, pas sur les exclusions, donc n_free peut devenir
    // < min(k_res, Q_res/L) -- bug latent trouvé sous gdb :
    // max_support = min(k,Q/L) sans plafond par `n` provoquait un accès hors
    // bornes de `order[idx]` quand s > n. Corrigé en plafonnant par `n`, ET
    // en signalant explicitement l'infaisabilité plutôt que de continuer
    // silencieusement avec best.value=-infini.
    bool feasible = true;
};

inline ThetaResult theta_solve(
    int n, int Q, int k, int L, int U, const std::vector<double>& cost)
{
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return cost[a] > cost[b]; });

    const int min_support = std::max(1, (Q + U - 1) / U);  // ceil(Q/U)
    const int max_support = std::min({k, Q / L, n});        // plafonné par n (cf. ci-dessus)

    ThetaResult best{-std::numeric_limits<double>::infinity(), std::vector<int>(n, 0), false};

    for (int s = min_support; s <= max_support; ++s) {
        const int R = Q - s * L;  // budget au-delà du plancher L, à distribuer
        if (R < 0 || R > s * (U - L)) continue;  // taille s infaisable pour Q

        std::vector<int> q(n, 0);
        double value = 0.0;
        int remaining = R;
        for (int idx = 0; idx < s; ++idx) {
            const int i = order[idx];
            const int extra = std::min(U - L, remaining);
            q[i] = L + extra;
            remaining -= extra;
            value += cost[i] * q[i];
        }
        // remaining doit être exactement 0 ici (R <= s*(U-L) garanti ci-dessus,
        // et on distribue toujours au maximum avant de passer au suivant).
        if (!best.feasible || value > best.value) {
            best.value = value;
            best.q = std::move(q);
            best.feasible = true;
        }
    }
    return best;
}

// ═════════════════════════════════════════════════════════════════════════
// Cholesky dense (M symétrique définie positive, garanti par construction :
// diag(psi)>0 domine, plus un terme rang-1 PSD ajouté par-dessus).
// ═════════════════════════════════════════════════════════════════════════

// Décompose M = LL', retourne L (triangulaire inférieure). Lève si M n'est
// pas définie positive (ne devrait jamais arriver ici -- cf. ci-dessus).
inline std::vector<std::vector<double>> cholesky(const std::vector<std::vector<double>>& M) {
    const int m = static_cast<int>(M.size());
    std::vector<std::vector<double>> Lmat(m, std::vector<double>(m, 0.0));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = M[i][j];
            for (int t = 0; t < j; ++t) s -= Lmat[i][t] * Lmat[j][t];
            if (i == j) {
                if (s <= 0.0) s = 1e-12;  // garde-fou numérique (PD attendu par construction)
                Lmat[i][j] = std::sqrt(s);
            } else {
                Lmat[i][j] = s / Lmat[j][j];
            }
        }
    }
    return Lmat;
}

// Résout M x = b via la factorisation de Cholesky L (descente puis remontée).
inline std::vector<double> cholesky_solve(const std::vector<std::vector<double>>& Lmat,
                                           const std::vector<double>& b) {
    const int m = static_cast<int>(b.size());
    std::vector<double> y(m), x(m);
    for (int i = 0; i < m; ++i) {
        double s = b[i];
        for (int t = 0; t < i; ++t) s -= Lmat[i][t] * y[t];
        y[i] = s / Lmat[i][i];
    }
    for (int i = m - 1; i >= 0; --i) {
        double s = y[i];
        for (int t = i + 1; t < m; ++t) s -= Lmat[t][i] * x[t];
        x[i] = s / Lmat[i][i];
    }
    return x;
}

// ═════════════════════════════════════════════════════════════════════════
// solve_quadratic_relaxation_bound — borne d'ÉLAGAGE (majorant), pas ζ_j
// ═════════════════════════════════════════════════════════════════════════
//
//   max_{q∈R^m} cost'q   s.c.   1'q = R,   q'Mq + 2 r'q ≤ Bres
//
// M dense symétrique définie positive (sous-matrice de Sigma_int_j restreinte
// aux indices encore libres à ce nœud -- cf. docstring du fichier : PAS de
// reconstruction rang-1, la sous-matrice dense exacte). Résolu par KKT :
// stationnarité donne q(mu,rho) = [M^{-1}cost - mu M^{-1}1]/(2rho) - M^{-1}r,
// puis mu(rho) déduit de la contrainte de somme ; rho* en FORME CLOSE (ex-bissection)
// pour saturer la contrainte quadratique (toujours active ici : sans bornes
// de boîte, le problème est non borné si elle ne l'était pas).

inline double solve_quadratic_relaxation_bound(
    const std::vector<std::vector<double>>& M,
    const std::vector<double>& cost,
    const std::vector<double>& r,
    double R, double Bres,
    std::vector<double>* q_out = nullptr,   // si non nul : solution q* de la relaxation (diag. branchement)
    bool* relax_infeasible_out = nullptr)   // si non nul : true ssi min{q'Mq+2r'q : 1'q=R} > Bres (+marge) -> sous-arbre
                                            // PROUVÉ infaisable (relaxation ⊇ ensemble faisable) ; retourne alors -inf
{
    const int m = static_cast<int>(cost.size());
    if (m == 0) { if (q_out) q_out->clear(); return 0.0; }  // rien de libre : pas de contribution résiduelle

    // PD vérifiée STRICTEMENT. cholesky() remplace un pivot <= 0 par 1e-12 (garde-fou
    // silencieux) -> minimum / borne faux si M n'est pas numériquement définie positive. Dans ce cas : aucune preuve
    // d'infaisabilité, borne +inf (aucun élagage) -- prudent. Sans effet (bit à bit) quand M est PD.
    {
        std::vector<std::vector<double>> Lchk;
        if (!ccpo_presolve_exact::chol_strict(M, Lchk)) {
            if (relax_infeasible_out) *relax_infeasible_out = false;
            if (q_out) q_out->assign(m, static_cast<double>(R) / m);
            return std::numeric_limits<double>::infinity();
        }
    }
    auto Lmat = cholesky(M);
    std::vector<double> ones(m, 1.0);
    std::vector<double> a = cholesky_solve(Lmat, cost);  // M^{-1} cost
    std::vector<double> e = cholesky_solve(Lmat, ones);  // M^{-1} 1
    std::vector<double> b = cholesky_solve(Lmat, r);     // M^{-1} r

    double sum_a = std::inner_product(a.begin(), a.end(), ones.begin(), 0.0);
    double sum_e = std::inner_product(e.begin(), e.end(), ones.begin(), 0.0);
    double sum_b = std::inner_product(b.begin(), b.end(), ones.begin(), 0.0);

    // Infaisabilité de la relaxation continue : minimum EXACT (forme close, problème convexe sur le sous-espace
    // affine 1'q=R) de q'Mq+2r'q : q_min = nu*M^-1 1 - M^-1 r, nu=(R+sum_b)/sum_e. Si ce minimum dépasse Bres
    // (marge >= 1e-6, la tolérance du test de feuille, + marge relative numérique), aucun q entier ne convient.
    if (relax_infeasible_out) {
        const double nu = (R + sum_b) / sum_e;
        std::vector<double> qm(m);
        for (int t = 0; t < m; ++t) qm[t] = nu * e[t] - b[t];
        double qMq = 0.0, rq = 0.0;
        for (int i = 0; i < m; ++i) {
            double Mq_i = 0.0;
            for (int t = 0; t < m; ++t) Mq_i += M[i][t] * qm[t];
            qMq += qm[i] * Mq_i;
            rq += r[i] * qm[i];
        }
        const double rmin = qMq + 2.0 * rq;
        *relax_infeasible_out = (rmin > Bres + 1e-6 + 1e-9 * (std::abs(Bres) + std::abs(rmin)));
        if (*relax_infeasible_out) {
            if (q_out) q_out->clear();
            return -std::numeric_limits<double>::infinity();
        }
    }

    // FORME CLOSE exacte (remplace la bissection sur rho). Avec q_min = nu*M^-1 1 - M^-1 r
    // (minimum du risque sur 1'q=R) et d = M^-1 cost - (sum_a/sum_e) M^-1 1 (1'd = 0), la stationnarité donne
    // q(rho) = q_min + t d (t = 1/(2 rho)) ; comme M q_min + r = nu 1 et 1'd = 0 :
    //   risque(q_min + t d) = rmin + t^2 g,  cost'(q_min + t d) = cost'q_min + t g,  g = d'Md = cost'd >= 0.
    // Max de cost'q sous risque <= Bres :  cost'q_min + sqrt(g * (Bres - rmin)).  Exact pour tout g >= 0 (g = 0 :
    // coût constant sur l'hyperplan, ex. une seule variable libre -- cas où la bissection, ou son encadrement, dérivait).
    // Relaxation vide (Bres < rmin) : ensemble faisable vide, toute valeur est un majorant ; on garde cost'q_min (prudent).
    const double nu_c = (R + sum_b) / sum_e, ratio = sum_a / sum_e;
    std::vector<double> qmin_c(m), d_c(m);
    for (int t = 0; t < m; ++t) { qmin_c[t] = nu_c * e[t] - b[t]; d_c[t] = a[t] - ratio * e[t]; }
    double rmin_c = 0.0;
    for (int i = 0; i < m; ++i) {
        double Mq_i = 0.0;
        for (int t = 0; t < m; ++t) Mq_i += M[i][t] * qmin_c[t];
        rmin_c += qmin_c[i] * (Mq_i + 2.0 * r[i]);
    }
    const double g_c = std::max(0.0, std::inner_product(cost.begin(), cost.end(), d_c.begin(), 0.0));
    const double slack_c = std::max(0.0, Bres - rmin_c);
    // q* (sert seulement au choix de branchement) : si g est au niveau du bruit d'arrondi (coût ~constant sur
    // l'hyperplan), d n'est que du bruit et t* = sqrt(slack/g) l'amplifierait -> q* = q_min. La VALEUR renvoyée garde g tel quel.
    const double g_scale = std::abs(std::inner_product(cost.begin(), cost.end(), a.begin(), 0.0));
    const double tstar = (g_c > 1e-12 * g_scale && g_c > 0.0) ? std::sqrt(slack_c / g_c) : 0.0;
    if (q_out) {
        q_out->assign(m, 0.0);
        for (int t = 0; t < m; ++t) (*q_out)[t] = qmin_c[t] + tstar * d_c[t];
    }
    return std::inner_product(cost.begin(), cost.end(), qmin_c.begin(), 0.0) + std::sqrt(g_c * slack_c);
}

// ═════════════════════════════════════════════════════════════════════════
// zeta_solve — max_{q∈D} cost'q  s.c.  q'Σ_j q ≤ B   (EXACT, sans budget)
// ═════════════════════════════════════════════════════════════════════════
//
// B&B sur l'ordre `order` (permutation de branchement, ex. branch_order de
// CCPOSharedData) ; à chaque nœud, valeur essayée de U à L puis 0. Élagage
// via solve_quadratic_relaxation_bound sur la sous-matrice DENSE (exacte,
// pas de reconstruction rang-1) restreinte aux indices encore libres.
// AUCUNE limite de temps/nœuds : tourne jusqu'à preuve d'optimalité, choix
// assumé (cf. docstring du fichier).

struct ZetaResult {
    double value;
    std::vector<int> q;
    bool feasible;
    int64_t nodes_explored = 0;  // DEBUG : diagnostic de coût
    // DIAGNOSTIC (warm_mode>=1) : valeur de l'incumbent glouton initial (NaN si aucun trouvé).
    double warm_value = std::numeric_limits<double>::quiet_NaN();
    // true ssi la recherche a été INTERROMPUE par le budget de nœuds (zeta_solve_ex, node_limit>0) : value/q = meilleur
    // incumbent trouvé (feasible=true) ou rien (feasible=false) ; ce n'est PAS un résultat exact.
    bool aborted = false;
};

namespace ccpo_zeta_detail {

struct State {
    int n, Q, k, L, U;
    const std::vector<double>* cost;
    const std::vector<std::vector<double>>* Sigma;  // dense n×n, symétrique
    double B;
    const std::vector<int>* order;
    // r_init[i] : correction linéaire EXTERNE -- interaction entre l'actif i
    // (encore libre dans CE zeta_solve) et les actifs déjà fixés par l'arbre
    // englobant (more(), gecode_bab_ccpo.hpp), cf. discussion "terme croisé" :
    //   q'Sigma_j q = C_ff (constante, déjà retirée de B avant l'appel)
    //               + 2 r_init'q_free   (LINÉAIRE, ici)
    //               + q_free'Sigma_free q_free
    // Nul (vecteur de zéros) à la racine, où aucune variable n'est encore
    // fixée à l'extérieur.
    const std::vector<double>* r_init;

    std::vector<int> q;
    double best_value = -std::numeric_limits<double>::infinity();
    std::vector<int> best_q;
    bool found = false;
    int64_t nodes_explored = 0;  // DEBUG : diagnostic de coût
    // Majorant global ζ ≤ θ (θ = même problème SANS contrainte de risque) : dès que l'incumbent
    // l'atteint, il est optimal -> arrêt. +inf = désactivé (warm_mode<2).
    double cap = std::numeric_limits<double>::infinity();
};

// Vérifie qu'il existe une façon de distribuer `remaining_q` parmi
// `remaining_k` slots actifs au plus, choisis parmi `n_free` indices libres,
// chacun dans {0}∪{L,...,U}. Condition nécessaire et suffisante : il existe
// s ∈ [max(0,?), min(remaining_k,n_free)] avec s*L ≤ remaining_q ≤ s*U et
// remaining_q ≤ (n_free choisis...) -- on cherche juste s'il existe s valide.
inline bool feasible_to_complete(int remaining_q, int remaining_k, int n_free, int L, int U) {
    if (remaining_q == 0) return true;  // s=0 marche toujours
    int s_max = std::min(remaining_k, n_free);
    for (int s = 1; s <= s_max; ++s)
        if (s * L <= remaining_q && remaining_q <= s * U) return true;
    return false;
}

inline void recurse(State& st, int pos, int remaining_q, int remaining_k,
                     double fixed_cost, double fixed_risk, double linear_contrib,
                     std::vector<double>& r)
{
    // fixed_risk       = q_S'Sigma_SS q_S       (pur interne à CE zeta_solve, inchangé)
    // linear_contrib   = Σ_{i∈S} r_init[i]*q_i  (interaction q_S avec l'EXTÉRIEUR --
    //                    utilise r_init CONSTANT, jamais le `r` accumulé ci-dessous,
    //                    qui lui mélange r_init et la contribution interne S->F et
    //                    ne doit servir QU'à la borne d'élagage, cf. docstring State).
    st.nodes_explored++;
    const int n = st.n;
    if (st.found && st.best_value >= st.cap - 1e-9) return;  // incumbent = θ : optimal, rien à prouver
    if (pos == n) {
        if (remaining_q == 0 && fixed_risk + 2.0 * linear_contrib <= st.B + 1e-6
            && fixed_cost > st.best_value) {
            st.best_value = fixed_cost;
            st.best_q = st.q;
            st.found = true;
        }
        return;
    }

    const int n_free = n - pos;
    if (!feasible_to_complete(remaining_q, remaining_k, n_free, st.L, st.U)) return;

    // Borne d'élagage : sous-problème résiduel sur les indices libres. Budget
    // restant = B moins le quadratique interne DÉJÀ engagé ET le linéaire
    // externe DÉJÀ engagé par S (2*linear_contrib) -- r_free porte le champ
    // linéaire restant (r_init + contribution interne de S) pour chaque
    // indice encore libre, cf. solve_quadratic_relaxation_bound.
    std::vector<int> free_idx(st.order->begin() + pos, st.order->end());
    const int m = static_cast<int>(free_idx.size());
    std::vector<std::vector<double>> M(m, std::vector<double>(m));
    std::vector<double> cost_free(m), r_free(m);
    for (int a = 0; a < m; ++a) {
        cost_free[a] = (*st.cost)[free_idx[a]];
        r_free[a] = r[free_idx[a]];
        for (int b = 0; b < m; ++b) M[a][b] = (*st.Sigma)[free_idx[a]][free_idx[b]];
    }
    const double bound = fixed_cost + solve_quadratic_relaxation_bound(
                                           M, cost_free, r_free,
                                           static_cast<double>(remaining_q),
                                           st.B - fixed_risk - 2.0 * linear_contrib);
    if (st.found && bound <= st.best_value + 1e-9) return;

    const int i = (*st.order)[pos];

    // q[i] = v, v de U à L (valeur haute d'abord -- trouve vite un bon incumbent)
    for (int v = st.U; v >= st.L; --v) {
        if (v > remaining_q || remaining_k <= 0) continue;
        double delta_risk = (*st.Sigma)[i][i] * v * v;
        for (int p = 0; p < pos; ++p) {
            const int j = (*st.order)[p];
            if (st.q[j] != 0) delta_risk += 2.0 * (*st.Sigma)[i][j] * v * st.q[j];
        }
        st.q[i] = v;
        std::vector<double> new_r = r;
        for (int l : free_idx)
            if (l != i) new_r[l] += (*st.Sigma)[i][l] * v;
        recurse(st, pos + 1, remaining_q - v, remaining_k - 1,
                fixed_cost + (*st.cost)[i] * v, fixed_risk + delta_risk,
                linear_contrib + (*st.r_init)[i] * v, new_r);
    }
    // q[i] = 0 (exclu) -- ne contribue ni au risque interne ni au terme externe
    st.q[i] = 0;
    recurse(st, pos + 1, remaining_q, remaining_k, fixed_cost, fixed_risk,
            linear_contrib, r);
}

// ── Incumbent initial GLOUTON (warm_mode>=1) ─────────────────────────────────
// Ne change JAMAIS la valeur retournée par zeta_solve : le B&B cherche ensuite strictement
// mieux ; l'incumbent n'est accepté que s'il est RE-VÉRIFIÉ exactement (somme, cardinalité,
// valeurs ∈ {0}∪[L,U], risque q'Σq+2r'q ≤ B+1e-6, même test que la feuille de recurse()).
// Principe : pour chaque taille de support s, point de départ = water-filling top-s de θ
// (optimal sans risque) ; phase "réparation" : tant que le risque dépasse B, appliquer le
// transfert (i->j, d unités) au meilleur ratio réduction de risque / perte de coût ; phase
// "amélioration" : transferts faisables qui augmentent le coût (meilleure amélioration).
// Un transfert (i->j,d) préserve Σq=Q ; il couvre décalage d'une unité, échange de support,
// ouverture (j inactif reçoit d>=L) et fermeture (i donne tout). Coût O(n²·U) par itération.
inline double risk_exact(int n, const std::vector<std::vector<double>>& S,
                         const std::vector<double>& r, const std::vector<int>& q) {
    double v = 0.0;
    for (int i = 0; i < n; ++i) {
        if (q[i] == 0) continue;
        double gi = 0.0;
        for (int j = 0; j < n; ++j) gi += S[i][j] * q[j];
        v += q[i] * (gi + 2.0 * r[i]);
    }
    return v;
}

struct GreedyIncumbent {
    bool found = false;
    double value = -std::numeric_limits<double>::infinity();
    std::vector<int> q;
};

inline GreedyIncumbent greedy_incumbent(
    int n, int Q, int k, int L, int U, const std::vector<double>& cost,
    const std::vector<std::vector<double>>& S, double B, const std::vector<double>& r)
{
    GreedyIncumbent best;
    best.q.assign(n, 0);
    std::vector<int> ord(n);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](int a, int b) { return cost[a] > cost[b]; });
    const int s_min = std::max(1, (Q + U - 1) / U);
    const int s_max = std::min({k, Q / L, n});
    const double tolB = B + 1e-6;

    for (int s = s_min; s <= s_max; ++s) {
        const int Rb = Q - s * L;
        if (Rb < 0 || Rb > s * (U - L)) continue;
        std::vector<int> q(n, 0);
        int rem = Rb;
        for (int t = 0; t < s; ++t) {
            const int extra = std::min(U - L, rem);
            q[ord[t]] = L + extra; rem -= extra;
        }
        std::vector<double> g(n, 0.0);  // g = Σ q
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) g[i] += S[i][j] * q[j];
        auto risk_of = [&]() {
            double v = 0.0;
            for (int i = 0; i < n; ++i) v += q[i] * (g[i] + 2.0 * r[i]);
            return v;
        };
        int active = s;
        double risk = risk_of();

        // énumère les transferts valides (i->j, d) ; f(i,j,d,d_risk,d_cost,active') -> void
        auto for_moves = [&](auto&& f) {
            for (int i = 0; i < n; ++i) {
                if (q[i] == 0) continue;
                for (int j = 0; j < n; ++j) {
                    if (j == i) continue;
                    for (int d = 1; d <= q[i]; ++d) {
                        const int qi2 = q[i] - d, qj2 = q[j] + d;
                        if (qi2 != 0 && qi2 < L) continue;
                        if (qj2 > U || (qj2 < L)) continue;  // qj2>=1 ; doit être >=L
                        const int act2 = active - (qi2 == 0 ? 1 : 0) + (q[j] == 0 ? 1 : 0);
                        if (act2 > k) continue;
                        const double dr = 2.0 * d * (g[j] - g[i] + r[j] - r[i])
                                          + double(d) * d * (S[j][j] + S[i][i] - 2.0 * S[i][j]);
                        f(i, j, d, dr, (cost[j] - cost[i]) * d, act2);
                    }
                }
            }
        };
        auto apply = [&](int i, int j, int d, int act2) {
            q[i] -= d; q[j] += d; active = act2;
            for (int t = 0; t < n; ++t) g[t] += d * (S[t][j] - S[t][i]);
            risk = risk_of();  // recalcul exact depuis g (O(n)), pas de dérive
        };

        // phase réparation
        bool ok = (risk <= tolB);
        for (int it = 0; !ok && it < 100; ++it) {
            double best_score = 0.0; int bi = -1, bj = -1, bd = 0, ba = 0;
            for_moves([&](int i, int j, int d, double dr, double dc, int a2) {
                if (dr >= -1e-12) return;
                const double loss = -dc;
                const double score = (loss <= 1e-12) ? 1e30 - dr : (-dr) / loss;
                if (score > best_score) { best_score = score; bi = i; bj = j; bd = d; ba = a2; }
            });
            if (bi < 0) break;
            apply(bi, bj, bd, ba);
            ok = (risk <= tolB);
        }
        if (!ok) continue;
        // phase amélioration (coût strictement croissant -> terminaison ; plafond de sécurité)
        for (int it = 0; it < 200; ++it) {
            double best_gain = 1e-12; int bi = -1, bj = -1, bd = 0, ba = 0;
            for_moves([&](int i, int j, int d, double dr, double dc, int a2) {
                if (dc <= best_gain) return;
                if (risk + dr > tolB) return;
                best_gain = dc; bi = i; bj = j; bd = d; ba = a2;
            });
            if (bi < 0) break;
            apply(bi, bj, bd, ba);
        }
        // vérification EXACTE indépendante (pas de confiance dans les mises à jour incrémentales)
        int sum = 0, act = 0; bool valid = true;
        for (int i = 0; i < n; ++i) {
            sum += q[i];
            if (q[i] != 0) { ++act; if (q[i] < L || q[i] > U) valid = false; }
        }
        if (!valid || sum != Q || act > k) continue;
        if (risk_exact(n, S, r, q) > tolB) continue;
        double val = 0.0;
        for (int i = 0; i < n; ++i) val += cost[i] * q[i];
        if (!best.found || val > best.value) { best.found = true; best.value = val; best.q = q; }
    }
    return best;
}

}  // namespace ccpo_zeta_detail

inline ZetaResult zeta_solve(
    int n, int Q, int k, int L, int U,
    const std::vector<double>& cost,
    const std::vector<std::vector<double>>& Sigma,  // dense n×n
    double B,
    const std::vector<int>& order,
    const std::vector<double>& r_init,  // (n,) -- correction linéaire externe, 0 à la racine
    int warm_mode = 0)  // 0 = comportement historique ; 1 = + incumbent glouton ; 2 = + plafond θ
{
    ccpo_zeta_detail::State st;
    st.n = n; st.Q = Q; st.k = k; st.L = L; st.U = U;
    st.cost = &cost; st.Sigma = &Sigma; st.B = B; st.order = &order;
    st.r_init = &r_init;
    st.q.assign(n, 0);

    double warm_value = std::numeric_limits<double>::quiet_NaN();
    if (warm_mode >= 1) {
        ccpo_zeta_detail::GreedyIncumbent gi =
            ccpo_zeta_detail::greedy_incumbent(n, Q, k, L, U, cost, Sigma, B, r_init);
        if (gi.found) {
            st.best_value = gi.value; st.best_q = gi.q; st.found = true;
            warm_value = gi.value;
        }
    }
    if (warm_mode >= 2) {
        ThetaResult th = theta_solve(n, Q, k, L, U, cost);
        if (th.feasible) st.cap = th.value;
    }

    std::vector<double> r = r_init;  // seed : le champ linéaire "vu" par un indice
                                      // encore libre commence à r_init, puis accumule
                                      // la contribution des fixations internes S->F.
    ccpo_zeta_detail::recurse(st, 0, Q, k, 0.0, 0.0, 0.0, r);

    if (!st.found) return {0.0, std::vector<int>(n, 0), false, st.nodes_explored, warm_value};
    return {st.best_value, st.best_q, true, st.nodes_explored, warm_value};
}

// ── Configuration de zeta_solve pour toutes les évaluations de L(λ) ──
// DÉFAUT DE PRODUCTION = (warm_mode 2, branching 1) : incumbent glouton + plafond ζ ≤ θ + branchement
// DYNAMIQUE (tous EXACTS : ζ identique à l'ancien solveur, vérifié sur 64 nœuds x 7 stratégies, 0 écart ;
// cf. notes/zeta_solve_etat_de_l_art.md). Utilisée par evaluate_ccpo_lagrangian(_omp) -> SLD, driver GNN batché,
// bindings Python, ET par le mode FEAS_ONLY. N'AFFECTE PAS le mode REF (Gecode seul, pas de zeta_solve).
// Retour à l'ancien comportement : CCPO_ZETA_WARM=0 CCPO_ZETA_BRANCH=0 (variables d'environnement, lues UNE FOIS
// à la première utilisation -> lecture seule ensuite, thread-safe sous OpenMP), ou ccpo_zeta_set_config(0,0).
//   warm_mode : 0 historique | 1 + incumbent glouton | 2 + incumbent glouton + plafond θ
//   branching : 0 ordre `order` fourni, valeurs U..L,0 | 1 DYNAMIQUE (zeta_solve_ex : argmax q* du nœud,
//               valeurs par proximité de q*)
//   infeas_prune : 1 = zeta_solve_ex élague les sous-arbres à relaxation continue infaisable (exact ; sans effet si branching=0)
//   inherit_infeas : 1 = une preuve d'infaisabilité d'un scénario est HÉRITÉE (λ-indépendante, monotone dans l'arbre) :
//                    z_j fixé à 0 dans l'espace Gecode (driver GNN) et exclu des itérations suivantes (boucle SLD)
//   scip_mode : 0 = désactivé (défaut) ; 1 = HYBRIDE : zeta_solve_ex avec budget de nœuds scip_nodes ; si dépassé ET si un incumbent
//               faisable existe ET n_free >= scip_nf0 -> SCIP (via le hook, cf. scip_zeta.cpp) ; repli sur le B&B complet si
//               SCIP échoue. N'a d'effet que si le binaire a installé le hook (CCPO_WITH_SCIP) et si branching=1.
//   scip_nodes / scip_nf0 / scip_tlim : budget de nœuds (~1,8 µs/nœud : 6000 ≈ 10 ms), seuil n_free, limite de temps SCIP (s)
struct CCPOZetaConfig {
    int warm_mode = 2; int branching = 1; int infeas_prune = 1; int inherit_infeas = 1;
    int scip_mode = 0; long long scip_nodes = 6000; int scip_nf0 = 11; double scip_tlim = 20.0;
    // évaluation de L(λ) PILOTÉE PAR L'ÉLAGAGE dans le driver batché (cf. evaluate_ccpo_lagrangian_lazy_omp).
    // lazy = 0 : calcul exact de tous les ζ_j (défaut, comportement historique) ; 1 : encadrement + raffinement jusqu'à décision.
    // lazy_check = 1 : calcule AUSSI L exact et compte les incohérences (validation ; coûte le calcul exact).
    int lazy = 0; int lazy_check = 0;
    // node_theta = 1 -> dans zeta_solve_ex, borne de nœud = coût fixé + min(relaxation continue, θ résiduel),
    // θ résiduel = max entier du coût sur les actifs libres SANS contrainte de variance (theta_solve, exact : bornes L..U,
    // cardinalité, intégralité). Majorant valide -> ζ inchangé (exact) ; 0 = comportement historique.
    int node_theta = 0;
};
inline CCPOZetaConfig ccpo_zeta_config_from_env() {
    CCPOZetaConfig c;
    if (const char* e = std::getenv("CCPO_ZETA_WARM"))   c.warm_mode = std::atoi(e);
    if (const char* e = std::getenv("CCPO_ZETA_BRANCH")) c.branching = std::atoi(e);
    if (const char* e = std::getenv("CCPO_ZETA_INFEAS_PRUNE")) c.infeas_prune = std::atoi(e);
    if (const char* e = std::getenv("CCPO_ZETA_INHERIT")) c.inherit_infeas = std::atoi(e);
    if (const char* e = std::getenv("CCPO_ZETA_SCIP")) c.scip_mode = std::atoi(e);
    if (const char* e = std::getenv("CCPO_ZETA_SCIP_NODES")) c.scip_nodes = std::atoll(e);
    if (const char* e = std::getenv("CCPO_ZETA_SCIP_NF0")) c.scip_nf0 = std::atoi(e);
    if (const char* e = std::getenv("CCPO_ZETA_SCIP_TLIM")) c.scip_tlim = std::atof(e);
    if (const char* e = std::getenv("CCPO_ZETA_LAZY")) c.lazy = std::atoi(e);
    if (const char* e = std::getenv("CCPO_ZETA_NODE_THETA")) c.node_theta = std::atoi(e);
    if (const char* e = std::getenv("CCPO_ZETA_LAZY_CHECK")) c.lazy_check = std::atoi(e);
    return c;
}
inline CCPOZetaConfig& ccpo_zeta_config() { static CCPOZetaConfig c = ccpo_zeta_config_from_env(); return c; }

// ═════════════════════════════════════════════════════════════════════════
// zeta_solve_ex — variante EXPÉRIMENTALE (diagnostic de l'ordre de branchement)
// ═════════════════════════════════════════════════════════════════════════
// Même problème, même borne d'élagage, même exactitude que zeta_solve ; seuls changent le choix de la
// variable (var_sel : 0 = 1re de la liste libre, ordre `order` ; 1 = DYNAMIQUE : plus grande composante
// de la solution q* de la relaxation du nœud) et l'ordre des valeurs (val_sel : 0 = U..L puis 0 ;
// 1 = par proximité de q*_i, ex-aequo : valeur la plus haute d'abord). Les actifs libres sont tenus dans
// une liste (plus un suffixe de `order` en mode dynamique). Non utilisée en production.
namespace ccpo_zeta_ex {
struct StateX {
    int n, Q, k, L, U;
    const std::vector<double>* cost;
    const std::vector<std::vector<double>>* Sigma;
    double B;
    const std::vector<double>* r_init;
    int var_sel = 0, val_sel = 0;
    bool prune_infeas = false;   // élagage des sous-arbres dont la relaxation continue est infaisable
    bool node_theta = false;     // borne de nœud renforcée par θ résiduel (cf. CCPOZetaConfig::node_theta)
    int64_t node_limit = 0;      // >0 : abandon (aborted=true) au-delà de ce nombre de nœuds (hybride SCIP)
    bool aborted = false;
    std::vector<int> q;
    double best_value = -std::numeric_limits<double>::infinity();
    std::vector<int> best_q;
    bool found = false;
    int64_t nodes = 0;
    double cap = std::numeric_limits<double>::infinity();
};

inline void recurse_x(StateX& st, const std::vector<int>& free_list, int remaining_q, int remaining_k,
                      double fixed_cost, double fixed_risk, double linear_contrib, std::vector<double>& r)
{
    if (st.aborted) return;
    st.nodes++;
    if (st.node_limit > 0 && st.nodes > st.node_limit) { st.aborted = true; return; }
    if (st.found && st.best_value >= st.cap - 1e-9) return;
    const int m = static_cast<int>(free_list.size());
    if (m == 0) {
        if (remaining_q == 0 && fixed_risk + 2.0 * linear_contrib <= st.B + 1e-6 && fixed_cost > st.best_value) {
            st.best_value = fixed_cost; st.best_q = st.q; st.found = true;
        }
        return;
    }
    if (!ccpo_zeta_detail::feasible_to_complete(remaining_q, remaining_k, m, st.L, st.U)) return;

    std::vector<std::vector<double>> M(m, std::vector<double>(m));
    std::vector<double> cost_free(m), r_free(m);
    for (int a = 0; a < m; ++a) {
        cost_free[a] = (*st.cost)[free_list[a]];
        r_free[a] = r[free_list[a]];
        for (int b = 0; b < m; ++b) M[a][b] = (*st.Sigma)[free_list[a]][free_list[b]];
    }
    std::vector<double> qstar;
    bool relax_infeasible = false;
    double bound = fixed_cost + solve_quadratic_relaxation_bound(
        M, cost_free, r_free, static_cast<double>(remaining_q),
        st.B - fixed_risk - 2.0 * linear_contrib, &qstar, st.prune_infeas ? &relax_infeasible : nullptr);
    if (relax_infeasible) return;   // aucun q (même continu) ne respecte le risque : sous-arbre infaisable
    if (st.node_theta) {            // θ résiduel : max entier sans variance sur les libres (majorant exact du sous-arbre)
        const double th_res = (remaining_q == 0) ? 0.0
            : theta_solve(m, remaining_q, remaining_k, st.L, st.U, cost_free).value;   // −∞ impossible ici (feasible_to_complete)
        if (std::isfinite(th_res)) bound = std::min(bound, fixed_cost + th_res);
    }
    if (st.found && bound <= st.best_value + 1e-9) return;

    int pick = 0;
    if (st.var_sel == 1)
        for (int a = 1; a < m; ++a) if (qstar[a] > qstar[pick]) pick = a;
    const int i = free_list[pick];
    std::vector<int> rest = free_list;
    rest.erase(rest.begin() + pick);

    std::vector<int> vals;
    for (int v = st.U; v >= st.L; --v) vals.push_back(v);
    vals.push_back(0);
    if (st.val_sel == 1) {
        const double t = qstar[pick];
        std::stable_sort(vals.begin(), vals.end(), [&](int a, int b) {
            return std::abs(a - t) < std::abs(b - t);
        });
    }
    for (int v : vals) {
        if (st.aborted) return;
        if (v == 0) {
            st.q[i] = 0;
            recurse_x(st, rest, remaining_q, remaining_k, fixed_cost, fixed_risk, linear_contrib, r);
            continue;
        }
        if (v > remaining_q || remaining_k <= 0) continue;
        // r[i]-r_init[i] = Σ_{j fixé actif} Σ_ij q_j  (cf. mise à jour de new_r ci-dessous)
        const double delta_risk = (*st.Sigma)[i][i] * v * v + 2.0 * v * (r[i] - (*st.r_init)[i]);
        st.q[i] = v;
        std::vector<double> new_r = r;
        for (int l : rest) new_r[l] += (*st.Sigma)[i][l] * v;
        recurse_x(st, rest, remaining_q - v, remaining_k - 1, fixed_cost + (*st.cost)[i] * v,
                  fixed_risk + delta_risk, linear_contrib + (*st.r_init)[i] * v, new_r);
    }
    st.q[i] = 0;
}
}  // namespace ccpo_zeta_ex

inline ZetaResult zeta_solve_ex(
    int n, int Q, int k, int L, int U, const std::vector<double>& cost,
    const std::vector<std::vector<double>>& Sigma, double B, const std::vector<int>& order,
    const std::vector<double>& r_init, int warm_mode, int var_sel, int val_sel, int infeas_prune = 0,
    int64_t node_limit = 0)
{
    ccpo_zeta_ex::StateX st;
    st.prune_infeas = (infeas_prune != 0);
    st.node_theta = (ccpo_zeta_config().node_theta != 0);
    st.node_limit = node_limit;
    st.n = n; st.Q = Q; st.k = k; st.L = L; st.U = U;
    st.cost = &cost; st.Sigma = &Sigma; st.B = B; st.r_init = &r_init;
    st.var_sel = var_sel; st.val_sel = val_sel;
    st.q.assign(n, 0);
    double warm_value = std::numeric_limits<double>::quiet_NaN();
    if (warm_mode >= 1) {
        ccpo_zeta_detail::GreedyIncumbent gi = ccpo_zeta_detail::greedy_incumbent(n, Q, k, L, U, cost, Sigma, B, r_init);
        if (gi.found) { st.best_value = gi.value; st.best_q = gi.q; st.found = true; warm_value = gi.value; }
    }
    if (warm_mode >= 2) {
        ThetaResult th = theta_solve(n, Q, k, L, U, cost);
        if (th.feasible) st.cap = th.value;
    }
    std::vector<double> r = r_init;
    ccpo_zeta_ex::recurse_x(st, order, Q, k, 0.0, 0.0, 0.0, r);
    ZetaResult out;
    out.nodes_explored = st.nodes; out.warm_value = warm_value; out.aborted = st.aborted;
    if (!st.found) { out.value = 0.0; out.q.assign(n, 0); out.feasible = false; }
    else { out.value = st.best_value; out.q = st.best_q; out.feasible = true; }
    return out;
}


// Hook SCIP (installé par le binaire si compilé avec SCIP, cf. scip_zeta.cpp) : résout max c'q (même problème que zeta_solve) par un
// solveur MIQCP, avec (init_q, init_val) comme solution initiale. Retour : 0 optimal (primal/dual valides, q vérifié exactement),
// 1 infaisable, 2 limite, 3 erreur. Pas de #ifdef ici (évite toute violation d'ODR entre unités de compilation).
using ScipZetaFn = int (*)(int n, int Q, int k, int L, int U, const std::vector<double>& cost,
                           const std::vector<std::vector<double>>& Sigma, double B, const std::vector<double>& r,
                           const std::vector<int>& init_q, double init_val, double tlim,
                           std::vector<int>& q_out, double& primal, double& dual);
inline ScipZetaFn& ccpo_scip_hook() { static ScipZetaFn f = nullptr; return f; }

struct CCPOZetaStats {                       // compteurs globaux (diagnostic ; atomiques, lisibles en fin d'exécution)
    std::atomic<long long> calls{0};         // appels zeta_solve_cfg en mode hybride (n_free >= nf0)
    std::atomic<long long> aborted{0};       // budget de nœuds dépassé
    std::atomic<long long> scip_ok{0};       // résolus par SCIP (optimal, vérifié)
    std::atomic<long long> scip_fail{0};     // SCIP en échec/limite/non vérifié -> repli B&B complet
    std::atomic<long long> no_incumbent{0};  // budget dépassé sans incumbent -> B&B complet (pas de SCIP)
    std::atomic<long long> scip_ns{0};       // temps passé dans SCIP (ns, cumulé sur les threads)
    // évaluation pilotée par l'élagage (lazy) :
    std::atomic<long long> lazy_evals{0};        // évaluations de L en mode lazy (hors nœuds prouvés infaisables d'emblée)
    std::atomic<long long> lazy_prune_cheap{0};  // tranchées « élaguer » par l'encadrement initial (aucun ζ exact)
    std::atomic<long long> lazy_keep_cheap{0};   // tranchées « garder » par l'encadrement initial (aucun ζ exact)
    std::atomic<long long> lazy_refined{0};      // évaluations ayant nécessité des ζ exacts
    std::atomic<long long> lazy_exact_solves{0}; // ζ exacts résolus (somme)
    std::atomic<long long> lazy_scen{0};         // scénarios éligibles vus (somme)
    std::atomic<long long> lazy_check_bad{0};    // lazy_check : décision ≠ exacte, ou L_lazy < L_exact
    std::atomic<long long> lazy_check_n{0};      // lazy_check : évaluations contrôlées
    std::atomic<long long> lazy_fallback{0};     // garde-fou : L⁺ non fini -> repli sur l'évaluation exacte
};
inline CCPOZetaStats& ccpo_zeta_stats() { static CCPOZetaStats st; return st; }

inline ZetaResult zeta_solve_cfg(
    int n, int Q, int k, int L, int U, const std::vector<double>& cost,
    const std::vector<std::vector<double>>& Sigma, double B, const std::vector<int>& order,
    const std::vector<double>& r_init)
{
    const CCPOZetaConfig& c = ccpo_zeta_config();
    if (c.branching == 1) {
        const bool hybrid = (c.scip_mode == 1 && ccpo_scip_hook() != nullptr && n >= c.scip_nf0 && c.scip_nodes > 0);
        if (!hybrid)
            return zeta_solve_ex(n, Q, k, L, U, cost, Sigma, B, order, r_init, c.warm_mode, 1, 1, c.infeas_prune);
        ccpo_zeta_stats().calls++;
        ZetaResult zr = zeta_solve_ex(n, Q, k, L, U, cost, Sigma, B, order, r_init, c.warm_mode, 1, 1, c.infeas_prune,
                                      static_cast<int64_t>(c.scip_nodes));
        if (!zr.aborted) return zr;                       // fini sous le budget : exact, comme sans hybride
        ccpo_zeta_stats().aborted++;
        if (zr.feasible) {                                // jamais de preuve d'infaisabilité déléguée à SCIP
            std::vector<int> qs; double primal = 0.0, dual = 0.0;
            const auto t0 = std::chrono::steady_clock::now();
            const int rc = ccpo_scip_hook()(n, Q, k, L, U, cost, Sigma, B, r_init, zr.q, zr.value, c.scip_tlim, qs, primal, dual);
            ccpo_zeta_stats().scip_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
            // accepté seulement si optimal, vérifié (par le hook) et écart primal-dual négligeable ; la valeur retournée est le MAJORANT
            // max(primal, dual) (valide pour L(lambda), cf. notes/zeta_solve_etat_de_l_art.md), le q primal sert au sous-gradient.
            const double tol = 1e-6 * std::max(1.0, std::abs(primal));
            if (rc == 0 && !qs.empty() && std::isfinite(primal) && std::isfinite(dual) && dual >= primal - tol && dual - primal <= tol
                && primal >= zr.value - tol) {
                ccpo_zeta_stats().scip_ok++;
                ZetaResult out;
                out.value = std::max(primal, dual); out.q = qs; out.feasible = true;
                out.nodes_explored = zr.nodes_explored; out.warm_value = zr.warm_value;
                return out;
            }
            ccpo_zeta_stats().scip_fail++;
        } else {
            ccpo_zeta_stats().no_incumbent++;
        }
        // repli : B&B complet sans budget (exact)
        return zeta_solve_ex(n, Q, k, L, U, cost, Sigma, B, order, r_init, c.warm_mode, 1, 1, c.infeas_prune);
    }
    return zeta_solve(n, Q, k, L, U, cost, Sigma, B, order, r_init, c.warm_mode);
}

// ═════════════════════════════════════════════════════════════════════════
// Presolve — élimine les scénarios PROVABLEMENT infaisables (z_j=0 forcé)
// ═════════════════════════════════════════════════════════════════════════
//
// La faisabilité de ζ_j (∃q∈D : q'Σ_j q ≤ B) ne dépend PAS de λ -- c'est une
// propriété fixe du scénario, calculable UNE FOIS avant REF/BOUND (cf.
// discussion : fixer des variables dans l'arbre externe ne peut que
// RÉTRÉCIR l'ensemble faisable, jamais l'élargir -- un scénario infaisable
// à la racine le reste à tout nœud descendant).
//
// Test : min_{0≤q≤U, Σq=Q} q'Σ_j q  >  B  ⟹  scénario prouvé infaisable.
// Relaxation valide (D ⊆ {0≤q≤U, Σq=Q} : boîte sans plancher L, sans
// cardinalité) -- un majorant du vrai minimum entier n'existerait pas ici,
// c'est un MINORANT (le vrai minimum sur D ne peut être que ≥ ce résultat),
// donc si même ce minorant dépasse B, le vrai minimum aussi -- sans risque
// de faux positif (éliminer à tort un scénario faisable).
//
// Actif-set convexe (borne, boîte, égalité) -- convergence en au plus n+2
// itérations, chacune un solve Cholesky O(n³). Mesuré : ~0.02ms/scénario à
// n=30 -- négligeable face au coût de zeta_solve sur un cas dur.

inline double presolve_min_variance_box(
    int n, double Q, double U, const std::vector<std::vector<double>>& Sigma)
{
    std::vector<char> fixed(n, 0);  // 0=libre, 1=fixé à 0, 2=fixé à U
    std::vector<double> qval(n, 0.0);

    for (int iter = 0; iter < n + 2; ++iter) {
        std::vector<int> free_idx;
        double fixed_sum = 0.0;
        for (int i = 0; i < n; ++i) {
            if (fixed[i] == 0) free_idx.push_back(i);
            else fixed_sum += qval[i];
        }
        const int m = static_cast<int>(free_idx.size());
        if (m == 0) break;
        const double R = Q - fixed_sum;

        std::vector<std::vector<double>> M(m, std::vector<double>(m));
        std::vector<double> r(m, 0.0);
        for (int a = 0; a < m; ++a) {
            for (int b = 0; b < m; ++b) M[a][b] = Sigma[free_idx[a]][free_idx[b]];
            for (int i = 0; i < n; ++i)
                if (fixed[i]) r[a] += Sigma[free_idx[a]][i] * qval[i];
        }

        auto Lmat = cholesky(M);
        std::vector<double> ones(m, 1.0);
        std::vector<double> e = cholesky_solve(Lmat, ones);
        std::vector<double> b = cholesky_solve(Lmat, r);
        const double sum_e = std::inner_product(e.begin(), e.end(), ones.begin(), 0.0);
        const double sum_b = std::inner_product(b.begin(), b.end(), ones.begin(), 0.0);
        const double mu = 2.0 * (R + sum_b) / sum_e;

        bool any_violation = false;
        for (int a = 0; a < m; ++a) {
            const double q_a = mu / 2.0 * e[a] - b[a];
            if (q_a < -1e-9) {
                fixed[free_idx[a]] = 1; qval[free_idx[a]] = 0.0; any_violation = true;
            } else if (q_a > U + 1e-9) {
                fixed[free_idx[a]] = 2; qval[free_idx[a]] = U; any_violation = true;
            } else {
                qval[free_idx[a]] = q_a;
            }
        }
        if (!any_violation) break;
    }

    double value = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) value += Sigma[i][j] * qval[i] * qval[j];
    return value;
}

// ═════════════════════════════════════════════════════════════════════════
// solve_ccpo_sld — boucle de sous-gradient, décomposition par scénario
// ═════════════════════════════════════════════════════════════════════════
//
// Fidèle au patron MKP (sld_solver.cpp) -- même reformulation en sac-à-dos
// min-coût sur z (problème-agnostique), même reparamétrage de centrage,
// même règle de pas diminuant 1/√(t+1), même "meilleur itéré" (best_upper).
// Seuls θ_j/ζ_j changent de nature (cf. ccpo_sld.hpp, ci-dessus).
//
// p_j = 1/N (uniforme) -- cohérent avec la contrainte de chance SAA
// Σ_j z_j ≥ ⌈(1-ε)N⌉ (un compte, pas une somme pondérée).
//
// Scénarios presolve-infaisables (infeasible_scenario[j]=true) : jamais
// éligibles pour z_j=1 (exclus du sac-à-dos), contribuent toujours θ_j --
// zeta_solve n'est JAMAIS appelé dessus (économie directe, cf. discussion).

struct SLDResult {
    double L_star;                                     // meilleure (= plus petite) borne vue
    std::vector<std::vector<double>> lambda_star;      // (N,n) -- lambda l'ayant atteinte
    int iterations_run;
    // false ssi ce sous-problème RÉSIDUEL est prouvé infaisable (moins de
    // min_feas scénarios éligibles restent réellement satisfaisables une
    // fois les variables déjà fixées prises en compte -- indépendant de λ,
    // cf. discussion : la faisabilité de zeta_j ne dépend jamais de λ, donc
    // une seule évaluation suffit à conclure). Si false, L_star/lambda_star
    // n'ont pas de sens -- l'appelant doit traiter le nœud comme un échec
    // immédiat, PAS poster une borne calculée à partir de valeurs infinies
    // (cause d'un crash Gecode::Int::OutOfLimits corrigé).
    bool feasible = true;
};

// ═════════════════════════════════════════════════════════════════════════
// evaluate_ccpo_lagrangian — évaluation "un coup" de L(λ), λ DONNÉ
// ═════════════════════════════════════════════════════════════════════════
//
// Brique de base extraite de la boucle de sous-gradient (utilisée par
// solve_ccpo_sld ci-dessous) et destinée à être appelée directement depuis
// l'extérieur -- notamment depuis un binding Python pour l'entraînement d'un
// GNN (cf. discussion "GNN sur le modèle de MKP") : `L(λ)` est une borne
// SUPÉRIEURE VALIDE pour N'IMPORTE QUEL λ (dualité faible, cf. §5 de la
// discussion GNN) -- un GNN peut donc proposer n'importe quel λ sans risque
// de casser la validité, seule la SERRURE de la borne dépend de la qualité
// de λ. Le sous-gradient retourné permet un entraînement par théorème de
// l'enveloppe (backward analytique, sans différentier le code combinatoire
// theta_solve/zeta_solve lui-même) -- même mécanisme que MKP
// (lagrangian_sld_cpp + loss.backward()).
//
// `order` (ordre de branchement pour zeta_solve) est un paramètre plutôt que
// recalculé ici : ne dépend que de cost_base (fixe pour tout le sous-arbre),
// évite de le retrier à chaque appel dans une boucle d'entraînement.

struct LagrangianEval {
    double L;                                     // L(λ) -- borne valide pour CE λ
    std::vector<std::vector<double>> subgrad;     // (N,n) : ∂L/∂λ_j[i] = x_j*[i] - x̄[i]
    bool feasible = true;                         // false ssi sous-problème résiduel prouvé infaisable
    // (N,) : 1 ssi zeta_solve a PROUVÉ le scénario j infaisable à ce nœud (indépendant de λ ; vrai aussi pour
    // tout descendant) -> à hériter (cf. CCPOZetaConfig::inherit_infeas). Vide si non renseigné.
    std::vector<uint8_t> infeasible_found;
};

inline LagrangianEval evaluate_ccpo_lagrangian(
    int n, int N, int k, int Q, int L, int U,
    const std::vector<double>& cost_base,
    const std::vector<std::vector<std::vector<double>>>& Sigma,
    const std::vector<double>& B_vec,
    int min_feas,
    const std::vector<bool>& infeasible_scenario,
    const std::vector<std::vector<double>>& lambda,
    const std::vector<int>& order,
    const std::vector<std::vector<double>>& R_ext)  // (N,n) -- correction linéaire
                                                      // externe par scénario, cf.
                                                      // zeta_solve::r_init. Toujours
                                                      // (N,n) de zéros à la racine.
{
    const std::vector<double> p(N, 1.0 / N);

    std::vector<double> Lambda_sum(n, 0.0);
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < n; ++i) Lambda_sum[i] += lambda[j][i];

    std::vector<double> theta_val(N, 0.0), delta(N, 0.0);
    std::vector<std::vector<int>> theta_q(N), sel_q(N);
    std::vector<uint8_t> infeas_found(N, 0);
    int n_feasible_eligible = 0;

    for (int j = 0; j < N; ++j) {
        std::vector<double> cost_j(n);
        for (int i = 0; i < n; ++i)
            cost_j[i] = cost_base[i] + lambda[j][i] / p[j] - Lambda_sum[i];

        ThetaResult tr = theta_solve(n, Q, k, L, U, cost_j);
        theta_val[j] = tr.value;
        theta_q[j] = tr.q;

        if (infeasible_scenario[j]) {
            delta[j] = std::numeric_limits<double>::infinity();  // jamais sélectionné
            sel_q[j] = tr.q;                                     // sans objet (z_j=0 toujours)
            continue;
        }
        ZetaResult zr = zeta_solve_cfg(n, Q, k, L, U, cost_j, Sigma[j], B_vec[j], order, R_ext[j]);
        if (!zr.feasible) {
            infeas_found[j] = 1;
            // Peut arriver légitimement sur un sous-problème RÉSIDUEL (cf.
            // docstring SLDResult::feasible) -- la faisabilité de ζ_j ne
            // dépend pas de λ, donc cette conclusion est valide pour tout λ.
            delta[j] = std::numeric_limits<double>::infinity();
            sel_q[j] = tr.q;
            continue;
        }
        ++n_feasible_eligible;
        delta[j] = std::max(0.0, theta_val[j] - zr.value);
        sel_q[j] = zr.q;
    }

    if (n_feasible_eligible < min_feas) {
        // Moins de min_feas scénarios peuvent être satisfaits même en
        // ignorant tout compromis de rendement -- sous-problème RÉSIDUEL
        // prouvé infaisable, indépendamment de λ. NE PAS continuer vers
        // knap_cost=+∞/L=-∞ (source du crash Gecode::Int::OutOfLimits
        // corrigé).
        return {std::numeric_limits<double>::quiet_NaN(), {}, false, infeas_found};
    }

    // ── Master : sac-à-dos min-coût, exact par tri (cf. sld_solver.cpp) ──
    std::vector<int> eligible;
    for (int j = 0; j < N; ++j)
        if (!infeasible_scenario[j]) eligible.push_back(j);
    std::sort(eligible.begin(), eligible.end(),
              [&](int a, int b) { return delta[a] < delta[b]; });

    std::vector<char> z(N, 0);
    double knap_cost = 0.0;
    for (int idx = 0; idx < min_feas && idx < static_cast<int>(eligible.size()); ++idx) {
        const int j = eligible[idx];
        z[j] = 1;
        knap_cost += p[j] * delta[j];
    }

    double sum_p_theta = 0.0;
    for (int j = 0; j < N; ++j) sum_p_theta += p[j] * theta_val[j];
    const double L1 = sum_p_theta - knap_cost;

    // x̄ et sous-gradient : x_j* = solution zeta si z_j=1 (retenu), sinon theta.
    std::vector<double> x_bar(n, 0.0);
    std::vector<std::vector<double>> x_star(N, std::vector<double>(n));
    for (int j = 0; j < N; ++j) {
        const std::vector<int>& chosen = z[j] ? sel_q[j] : theta_q[j];
        for (int i = 0; i < n; ++i) {
            x_star[j][i] = chosen[i];
            x_bar[i] += p[j] * chosen[i];
        }
    }

    std::vector<std::vector<double>> subgrad(N, std::vector<double>(n));
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < n; ++i)
            subgrad[j][i] = x_star[j][i] - x_bar[i];

    return {L1, subgrad, true, infeas_found};
}

// ═════════════════════════════════════════════════════════════════════════
// evaluate_ccpo_lagrangian_omp — IDENTIQUE à evaluate_ccpo_lagrangian
// ci-dessus, boucle scénario (theta_solve + zeta_solve) parallélisée par
// OpenMP (même schéma que solve_zeta_batch côté MKP, solver_bab/mkp/bnb_solver.cpp).
//
// Chaque itération j écrit UNIQUEMENT dans theta_val[j]/theta_q[j]/
// delta[j]/sel_q[j] (indices disjoints, pré-alloués avant la boucle) --
// aucune dépendance entre itérations, donc parallélisable sans section
// critique ni verrou. Seul n_feasible_eligible est partagé -- réduction.
// theta_solve/zeta_solve sont des fonctions pures (aucun état
// statique/global mutable, vérifié) -- thread-safe telles quelles.
//
// La partie "master" (tri + sac-à-dos min-coût sur δ_j, après la boucle)
// reste séquentielle : coût O(N log N), négligeable face à N appels
// zeta_solve (chacun potentiellement des secondes à un B&B quadratique exact
// sans budget -- jusqu'à ~55 s par appel mesurés sur les plus grandes cellules).
// ═════════════════════════════════════════════════════════════════════════

inline LagrangianEval evaluate_ccpo_lagrangian_omp(
    int n, int N, int k, int Q, int L, int U,
    const std::vector<double>& cost_base,
    const std::vector<std::vector<std::vector<double>>>& Sigma,
    const std::vector<double>& B_vec,
    int min_feas,
    const std::vector<bool>& infeasible_scenario,
    const std::vector<std::vector<double>>& lambda,
    const std::vector<int>& order,
    const std::vector<std::vector<double>>& R_ext)
{
    const std::vector<double> p(N, 1.0 / N);

    std::vector<double> Lambda_sum(n, 0.0);
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < n; ++i) Lambda_sum[i] += lambda[j][i];

    std::vector<double> theta_val(N, 0.0), delta(N, 0.0);
    std::vector<std::vector<int>> theta_q(N), sel_q(N);
    std::vector<uint8_t> infeas_found(N, 0);
    int n_feasible_eligible = 0;

    #pragma omp parallel for schedule(dynamic) reduction(+:n_feasible_eligible)
    for (int j = 0; j < N; ++j) {
        std::vector<double> cost_j(n);
        for (int i = 0; i < n; ++i)
            cost_j[i] = cost_base[i] + lambda[j][i] / p[j] - Lambda_sum[i];

        ThetaResult tr = theta_solve(n, Q, k, L, U, cost_j);
        theta_val[j] = tr.value;
        theta_q[j] = tr.q;

        if (infeasible_scenario[j]) {
            delta[j] = std::numeric_limits<double>::infinity();
            sel_q[j] = tr.q;
            continue;
        }
        ZetaResult zr = zeta_solve_cfg(n, Q, k, L, U, cost_j, Sigma[j], B_vec[j], order, R_ext[j]);
        if (!zr.feasible) {
            infeas_found[j] = 1;
            delta[j] = std::numeric_limits<double>::infinity();
            sel_q[j] = tr.q;
            continue;
        }
        ++n_feasible_eligible;
        delta[j] = std::max(0.0, theta_val[j] - zr.value);
        sel_q[j] = zr.q;
    }

    if (n_feasible_eligible < min_feas) {
        return {std::numeric_limits<double>::quiet_NaN(), {}, false, infeas_found};
    }

    std::vector<int> eligible;
    for (int j = 0; j < N; ++j)
        if (!infeasible_scenario[j]) eligible.push_back(j);
    std::sort(eligible.begin(), eligible.end(),
              [&](int a, int b) { return delta[a] < delta[b]; });

    std::vector<char> z(N, 0);
    double knap_cost = 0.0;
    for (int idx = 0; idx < min_feas && idx < static_cast<int>(eligible.size()); ++idx) {
        const int j = eligible[idx];
        z[j] = 1;
        knap_cost += p[j] * delta[j];
    }

    double sum_p_theta = 0.0;
    for (int j = 0; j < N; ++j) sum_p_theta += p[j] * theta_val[j];
    const double L1 = sum_p_theta - knap_cost;

    std::vector<double> x_bar(n, 0.0);
    std::vector<std::vector<double>> x_star(N, std::vector<double>(n));
    for (int j = 0; j < N; ++j) {
        const std::vector<int>& chosen = z[j] ? sel_q[j] : theta_q[j];
        for (int i = 0; i < n; ++i) {
            x_star[j][i] = chosen[i];
            x_bar[i] += p[j] * chosen[i];
        }
    }

    std::vector<std::vector<double>> subgrad(N, std::vector<double>(n));
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < n; ++i)
            subgrad[j][i] = x_star[j][i] - x_bar[i];

    return {L1, subgrad, true, infeas_found};
}

// ═════════════════════════════════════════════════════════════════════════
// evaluate_ccpo_lagrangian_lazy_omp — évaluation de L(λ) PILOTÉE PAR L'ÉLAGAGE
// ═════════════════════════════════════════════════════════════════════════
// Le driver B&B n'a besoin de L(λ) que pour décider « L < tau » (nœud élagué) et pour poster obj ≤ ⌊cost_fixed + L⌋.
// L(δ) = Σ p θ_j − (somme des min_feas plus petits p δ_j), δ_j = θ_j − ζ_j, est DÉCROISSANT en chaque δ_j.
// Pour chaque scénario éligible, sans recherche :
//   ζ_j⁻ = valeur de l'incumbent glouton (point faisable) ≤ ζ_j ; −∞ si aucun ;
//   ζ_j⁺ = min(θ_j, borne de la relaxation continue à la racine) ≥ ζ_j ;
//   relaxation infaisable (si infeas_prune) -> scénario PROUVÉ infaisable (même preuve que zeta_solve_ex à la racine).
// D'où δ_j ∈ [θ_j − ζ_j⁺, θ_j − ζ_j⁻] et L⁻ = L(δ⁺) ≤ L(λ) ≤ L⁺ = L(δ⁻).
//   L⁺ < tau  -> élaguer (décision identique au calcul exact) ;
//   L⁻ ≥ tau  -> pas d'élagage possible même avec ζ exacts ; on renvoie L⁺ (borne VALIDE, éventuellement plus lâche) ;
//   sinon     -> ζ exacts (zeta_solve_cfg, inchangé) sur les scénarios à intervalle le plus large, par paquets, jusqu'à décision.
// Tous les scénarios raffinés -> L exact. Les ζ calculés restent EXACTS ; seuls les scénarios non nécessaires ne sont pas résolus.
// tau = −∞ : aucun élagage possible -> renvoie L⁺ sans aucun ζ exact. Sous-gradient NON calculé (inutile au driver).
// feasible = false ssi moins de min_feas scénarios non prouvés infaisables (preuves exactes uniquement), comme l'exact.
inline double ccpo_L_from_delta(const std::vector<double>& theta_val, const std::vector<double>& delta,
                                const std::vector<int>& eligible, int min_feas, int N)
{
    const double p = 1.0 / N;
    std::vector<double> d;
    d.reserve(eligible.size());
    for (int j : eligible) d.push_back(delta[j]);
    if (static_cast<int>(d.size()) < min_feas) return -std::numeric_limits<double>::infinity();
    std::nth_element(d.begin(), d.begin() + std::max(0, min_feas - 1), d.end());
    double knap = 0.0;
    for (int i = 0; i < min_feas; ++i) {
        if (!std::isfinite(d[i])) return -std::numeric_limits<double>::infinity();
        knap += p * d[i];
    }
    double sum_p_theta = 0.0;
    for (int j = 0; j < N; ++j) sum_p_theta += p * theta_val[j];
    return sum_p_theta - knap;
}

inline LagrangianEval evaluate_ccpo_lagrangian_lazy_omp(
    int n, int N, int k, int Q, int L, int U,
    const std::vector<double>& cost_base,
    const std::vector<std::vector<std::vector<double>>>& Sigma,
    const std::vector<double>& B_vec,
    int min_feas,
    const std::vector<bool>& infeasible_scenario,
    const std::vector<std::vector<double>>& lambda,
    const std::vector<int>& order,
    const std::vector<std::vector<double>>& R_ext,
    double tau)
{
    const double INF = std::numeric_limits<double>::infinity();
    const double p = 1.0 / N;
    const CCPOZetaConfig& cfg = ccpo_zeta_config();
    std::vector<double> Lambda_sum(n, 0.0);
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < n; ++i) Lambda_sum[i] += lambda[j][i];

    std::vector<std::vector<double>> cost(N, std::vector<double>(n));
    std::vector<double> theta_val(N, 0.0), dlo(N, INF), dhi(N, INF);
    std::vector<uint8_t> infeas_found(N, 0), exact(N, 0), elig_flag(N, 0);

    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < n; ++i) cost[j][i] = cost_base[i] + lambda[j][i] / p - Lambda_sum[i];
        const ThetaResult thr = theta_solve(n, Q, k, L, U, cost[j]);
        theta_val[j] = thr.value;
        if (infeasible_scenario[j]) continue;                       // δ = +∞ (jamais retenu), comme l'exact
        // ensemble D vide (Σq = Q, cardinalité, bornes irréalisables) : ζ_j ≤ θ_j = −∞ -> scénario PROUVÉ infaisable
        // (correctif : sinon δ⁻ = −∞ − (−∞) = NaN -> L NaN -> Gecode::Int::OutOfLimits)
        if (!thr.feasible || !std::isfinite(thr.value)) { infeas_found[j] = 1; exact[j] = 1; continue; }
        // borne supérieure : relaxation continue à la racine (mêmes arguments que recurse_x à la racine)
        bool relax_inf = false;
        const double rb = solve_quadratic_relaxation_bound(Sigma[j], cost[j], R_ext[j], static_cast<double>(Q), B_vec[j],
                                                           nullptr, cfg.infeas_prune ? &relax_inf : nullptr);
        if (relax_inf) { infeas_found[j] = 1; exact[j] = 1; continue; }   // preuve exacte -> δ = +∞
        elig_flag[j] = 1;
        const double zhi = (std::isnan(rb) || rb == -std::numeric_limits<double>::infinity()) ? theta_val[j] : std::min(theta_val[j], rb);
        dlo[j] = std::max(0.0, theta_val[j] - zhi);
        const ccpo_zeta_detail::GreedyIncumbent gi = ccpo_zeta_detail::greedy_incumbent(n, Q, k, L, U, cost[j], Sigma[j], B_vec[j], R_ext[j]);
        dhi[j] = gi.found ? std::max(0.0, theta_val[j] - gi.value) : INF;
        if (gi.found && gi.value >= theta_val[j] - 1e-9 * std::max(1.0, std::abs(theta_val[j]))) { dlo[j] = dhi[j]; exact[j] = 1; }
    }
    std::vector<int> eligible;                                       // non prouvés infaisables (presolve ou relaxation)
    for (int j = 0; j < N; ++j) if (elig_flag[j]) eligible.push_back(j);
    // feasibilité : même règle que l'exact, avec les seules preuves exactes disponibles
    if (static_cast<int>(eligible.size()) < min_feas)
        return {std::numeric_limits<double>::quiet_NaN(), {}, false, infeas_found};

    auto& st = ccpo_zeta_stats();
    st.lazy_evals++; st.lazy_scen += static_cast<long long>(eligible.size());
    double Lhi = ccpo_L_from_delta(theta_val, dlo, eligible, min_feas, N);
    double Llo = ccpo_L_from_delta(theta_val, dhi, eligible, min_feas, N);
    if (std::isnan(Lhi) || !std::isfinite(Lhi)) {                   // garde-fou : jamais de borne non finie -> calcul exact
        st.lazy_fallback++;
        return evaluate_ccpo_lagrangian_omp(n, N, k, Q, L, U, cost_base, Sigma, B_vec, min_feas, infeasible_scenario,
                                            lambda, order, R_ext);
    }
    if (Lhi < tau) { st.lazy_prune_cheap++; return {Lhi, {}, true, infeas_found}; }
    if (Llo >= tau) { st.lazy_keep_cheap++; return {Lhi, {}, true, infeas_found}; }

    st.lazy_refined++;
    std::vector<int> todo;
    for (int j : eligible) if (!exact[j]) todo.push_back(j);
    std::sort(todo.begin(), todo.end(), [&](int a, int b) { return dhi[a] - dlo[a] > dhi[b] - dlo[b]; });   // ∞ d'abord
    const int chunk = std::max(1, omp_get_max_threads());
    size_t pos = 0;
    while (pos < todo.size()) {
        const size_t end = std::min(todo.size(), pos + static_cast<size_t>(chunk));
        #pragma omp parallel for schedule(dynamic)
        for (size_t t = pos; t < end; ++t) {
            const int j = todo[t];
            ZetaResult zr = zeta_solve_cfg(n, Q, k, L, U, cost[j], Sigma[j], B_vec[j], order, R_ext[j]);
            if (!zr.feasible) { infeas_found[j] = 1; dlo[j] = dhi[j] = INF; }
            else dlo[j] = dhi[j] = std::max(0.0, theta_val[j] - zr.value);
            exact[j] = 1;
        }
        st.lazy_exact_solves += static_cast<long long>(end - pos);
        pos = end;
        std::vector<int> elig2;                                      // les infaisables découverts sortent (δ = +∞ dans les deux)
        for (int j : eligible) if (!(infeas_found[j])) elig2.push_back(j);
        if (static_cast<int>(elig2.size()) < min_feas)
            return {std::numeric_limits<double>::quiet_NaN(), {}, false, infeas_found};
        Lhi = ccpo_L_from_delta(theta_val, dlo, eligible, min_feas, N);
        Llo = ccpo_L_from_delta(theta_val, dhi, eligible, min_feas, N);
        if (Lhi < tau || Llo >= tau) break;
    }
    return {Lhi, {}, true, infeas_found};
}

// ═════════════════════════════════════════════════════════════════════════
// solve_ccpo_sld — boucle de sous-gradient (appelle evaluate_ccpo_lagrangian
// à chaque itération, sans dupliquer sa logique)
// ═════════════════════════════════════════════════════════════════════════

inline SLDResult solve_ccpo_sld(
    int n, int N, int k, int Q, int L, int U,
    const std::vector<double>& cost_base,                          // c, SANS lambda
    const std::vector<std::vector<std::vector<double>>>& Sigma,    // (N,n,n) dense par scénario
    const std::vector<double>& B_vec,                              // budget résiduel PAR scénario
                                                                    // (= B partout à la racine ;
                                                                    // B - risque_fixé_j à un nœud)
    int min_feas,
    const std::vector<bool>& infeasible_scenario,
    int T,
    std::vector<std::vector<double>> lambda,                       // (N,n) -- init (0 = cold start)
    const std::vector<std::vector<double>>& R_ext,                 // (N,n) -- cf. evaluate_ccpo_lagrangian
    double rel_tol = 1e-4, int tol_window = 30,
    bool use_omp = false)   // evalue L(lambda) avec la boucle scenario OpenMP (comparaison a parallelisme egal avec le chemin GNN)
{
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return cost_base[a] > cost_base[b]; });

    // Faisabilité STRUCTURELLE du sous-problème résiduel (indépendante du
    // scénario ET de λ -- ne dépend que de n,Q,k,L,U) : si theta_solve ne
    // trouve aucune taille de support réalisable, aucun q ne peut compléter
    // ce nœud, quel que soit le coût utilisé. Vérifiée UNE FOIS ici plutôt
    // que redondamment à chaque scénario/itération (cf. bug latent trouvé
    // sous gdb).
    {
        ThetaResult probe = theta_solve(n, Q, k, L, U, cost_base);
        if (!probe.feasible)
            return {std::numeric_limits<double>::quiet_NaN(), lambda, 0, false};
    }

    double best_L = std::numeric_limits<double>::infinity();
    std::vector<std::vector<double>> best_lambda = lambda;
    std::vector<double> L_history;

    // Héritage des preuves d'infaisabilité entre itérations : la faisabilité de zeta_j ne dépend pas de λ, donc un
    // scénario prouvé infaisable à l'itération t l'est à toutes les suivantes (inutile de le re-prouver T fois).
    std::vector<bool> infeas_local = infeasible_scenario;
    const bool inherit = ccpo_zeta_config().inherit_infeas != 0;

    int iter = 0;
    for (; iter < T; ++iter) {
        LagrangianEval ev = use_omp
            ? evaluate_ccpo_lagrangian_omp(n, N, k, Q, L, U, cost_base, Sigma, B_vec, min_feas,
                                           infeas_local, lambda, order, R_ext)
            : evaluate_ccpo_lagrangian(n, N, k, Q, L, U, cost_base, Sigma, B_vec, min_feas,
                                       infeas_local, lambda, order, R_ext);

        if (!ev.feasible)
            return {std::numeric_limits<double>::quiet_NaN(), lambda, iter, false};
        if (inherit)
            for (int j = 0; j < N && j < static_cast<int>(ev.infeasible_found.size()); ++j)
                if (ev.infeasible_found[j]) infeas_local[j] = true;

        if (ev.L < best_L) {
            best_L = ev.L;
            best_lambda = lambda;
        }
#ifdef CCPO_SLD_TRACE
        std::cerr << "iter=" << iter << " L1=" << ev.L << " best_L=" << best_L << "\n";
#endif

        // Critère d'arrêt : amélioration relative de best_L sur une fenêtre glissante.
        L_history.push_back(best_L);
        if (static_cast<int>(L_history.size()) > tol_window) {
            const double old = L_history[L_history.size() - 1 - tol_window];
            const double rel_impr = std::abs(old - best_L) / std::max(1.0, std::abs(old));
            if (rel_impr < rel_tol) { ++iter; break; }
        }

        // Pas diminuant + mise à jour (λ_j ← λ_j − pas·sous-gradient_j).
        const double step = 1.0 / std::sqrt(static_cast<double>(iter + 1));
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < n; ++i)
                lambda[j][i] -= step * ev.subgrad[j][i];
    }

    return {best_L, best_lambda, iter};
}

// Retourne, pour chaque scénario j (0..N-1), true si PROUVÉ infaisable
// (z_j doit être fixé à 0). `sigma` : triangle supérieur par scénario, même
// format que CCPOSharedData::sigma. `dense_buf` réutilisé entre appels pour
// éviter n reallocations O(n²) par scénario.
inline std::vector<bool> presolve_infeasible_scenarios(
    int n, int Q, int U, double B,
    const std::vector<std::vector<SigmaTriplet3>>& sigma)
{
    const int N = static_cast<int>(sigma.size());
    std::vector<bool> infeasible(N, false);
    std::vector<std::vector<double>> dense(n, std::vector<double>(n, 0.0));
    // Défaut : presolve exact certifié ; CCPO_PRESOLVE=legacy : ancien presolve (reproduction seulement, peut écarter à tort).
    const bool exact = ccpo_presolve_exact::use_exact_from_env();

    for (int j = 0; j < N; ++j) {
        for (auto& row : dense) std::fill(row.begin(), row.end(), 0.0);
        for (const auto& t : sigma[j]) {
            dense[t[0]][t[1]] = t[2];
            dense[t[1]][t[0]] = t[2];
        }
        if (exact) {
            // borne inférieure CERTIFIÉE (exacte) du min boîte, cf. ccpo_presolve_exact.hpp.
            const auto r = ccpo_presolve_exact::box_min_certified(n, Q, U, dense);
            infeasible[j] = r.lower > B * (1.0 + ccpo_presolve_exact::kRelTol);
        } else {
            const double var_min = presolve_min_variance_box(n, Q, U, dense);
            infeasible[j] = var_min > B;
        }
    }
    std::fprintf(stderr, "[presolve] mode=%s n_infeasible=%d/%d\n", exact ? "exact" : "legacy",
                 static_cast<int>(std::count(infeasible.begin(), infeasible.end(), true)), N);
    return infeasible;
}
