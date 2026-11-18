#pragma once
/*
 * dld_base.hpp — Interface abstraite DLDProblem + struct résultat.
 *
 * DLD est la version décomposée de SLD : au lieu d'un multiplicateur λ_i
 * par scénario, on introduit un multiplicateur λ_{i,j} par couple
 * (scénario i, contrainte j).
 *
 * Hypothèse structurelle : le domaine faisable du scénario i est une
 * intersection de m contraintes :
 *
 *   X^i = X^i_1 ∩ X^i_2 ∩ … ∩ X^i_m
 *
 * Exemples :
 *   MKP  → X^i_j = { x ∈ {0,1}^n : A[i][j] x ≤ b[i][j] }
 *
 * Le Lagrangien relaxe le consensus x_{i,j} = x̄ :
 *
 *   L1(λ) = Σ_{i,j} p_{i,j} θ_{i,j}(λ)
 *          − min_y { Σ_i p_i Δ_i(λ) y_i | Σ_i p_i y_i ≥ 1−ε }
 *
 * avec p_{i,j} = p_i / m (répartition uniforme sur les m contraintes),
 *
 *   θ_{i,j}(λ) = max_{x dans le domaine de base} cost_{i,j}^T x
 *   ζ_{i,j}(λ) = max_{x ∈ X^i_j}               cost_{i,j}^T x
 *   cost_{i,j}  = c + λ_{i,j}/p_{i,j} − Λ_sum
 *   Λ_sum        = Σ_{i,j} λ_{i,j}
 *   δ_{i,j}     = max(0, θ_{i,j} − ζ_{i,j})
 *   Δ_i          = (1/m) Σ_j δ_{i,j}
 *
 * Implémenter solve_theta et solve_zeta suffit pour brancher un problème.
 */

#include "types.hpp"
#include <utility>

// ── Résultat ──────────────────────────────────────────────────────────────────

struct DLDResult {
    double DLD_star;      // meilleure borne supérieure trouvée
    Vec3d  lambda_star;     // λ* optimal : (N, m, d)
    Vec1d  convergence;     // historique de L1 aux pas logués
    Vec1d  times_s;         // temps (s) aux pas logués
    Vec1d  gnorm_history;   // ||g|| aux pas logués
    Vec1d  z_changes;       // nb de scénarios où y* a changé
    double solve_time_s;
    int    iterations_run;
    // Temps cumulé (s) passé dans le calcul de θ_{i,j}/ζ_{i,j} (étape B de la
    // boucle, cf. docstring dld_solver.cpp) sur toutes les itérations de cet
    // appel -- ajouté 2026-08-20 pour comparer la répartition du coût à celle
    // du mode GNN (graph_build/forward/zeta_theta), à la demande de
    // l'utilisateur (comparaison REF/DLD/GNN à taille croissante). Le reste
    // (solve_time_s - t_zeta_theta_s) couvre Λ_sum, knapsack, sous-gradient,
    // pas de descente, logging (étapes A,C-H) -- l'équivalent DLD de ce que
    // "forward" produit côté GNN (mise à jour de λ).
    double t_zeta_theta_s = 0.0;
};

// ── Classe abstraite ──────────────────────────────────────────────────────────

class DLDProblem {
public:
    virtual ~DLDProblem() = default;

    virtual int           n_vars()        const = 0;  // dimension d de x aplati
    virtual int           N_scen()        const = 0;  // nombre de scénarios N
    virtual int           n_constraints() const = 0;  // nombre de contraintes m
    virtual double        epsilon()       const = 0;
    virtual const Vec1d&  p()             const = 0;  // probabilités (N,)
    virtual const Vec1d&  c()             const = 0;  // coûts déterministes (d,)

    /*
     * solve_theta(j, cost)
     *   Max sur le domaine de base SANS la contrainte j.
     *   Pour MKP, cela revient à un max non contraint sur {0,1}^n.
     *   Le paramètre j est fourni pour les problèmes où le domaine de base
     *   dépend de l'indice de contrainte.
     *   Retourne (valeur optimale, x*).
     */
    virtual std::pair<double, Vec1d>
    solve_theta(int j, const Vec1d& cost) const = 0;

    /*
     * solve_zeta(i, j, cost)
     *   Max sur le domaine de base AVEC la contrainte j du scénario i.
     *   Retourne (valeur optimale, x*).
     */
    virtual std::pair<double, Vec1d>
    solve_zeta(int i, int j, const Vec1d& cost) const = 0;
};
