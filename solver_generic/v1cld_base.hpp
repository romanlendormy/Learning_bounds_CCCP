#pragma once
/*
 * v1cld_base.hpp — Interface abstraite V1CLDProblem + struct résultat.
 *
 * v1CLD est la version décomposée de v1LD : au lieu d'un multiplicateur λ_i
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
 *   SSP  → X^i_j = { x séquence : x appartient à l'automate j du scénario i }
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

struct V1CLDResult {
    double v1CLD_star;      // meilleure borne supérieure trouvée
    Vec3d  lambda_star;     // λ* optimal : (N, m, d)
    Vec1d  convergence;     // historique de L1 aux pas logués
    Vec1d  times_s;         // temps (s) aux pas logués
    Vec1d  gnorm_history;   // ||g|| aux pas logués
    Vec1d  z_changes;       // nb de scénarios où y* a changé
    double solve_time_s;
    int    iterations_run;
};

// ── Classe abstraite ──────────────────────────────────────────────────────────

class V1CLDProblem {
public:
    virtual ~V1CLDProblem() = default;

    virtual int           n_vars()        const = 0;  // dimension d de x aplati
    virtual int           N_scen()        const = 0;  // nombre de scénarios N
    virtual int           n_constraints() const = 0;  // nombre de contraintes m
    virtual double        epsilon()       const = 0;
    virtual const Vec1d&  p()             const = 0;  // probabilités (N,)
    virtual const Vec1d&  c()             const = 0;  // coûts déterministes (d,)

    /*
     * solve_theta(j, cost)
     *   Max sur le domaine de base SANS la contrainte j.
     *   Dans la plupart des cas (MKP, SSP), cela revient à un max non contraint
     *   sur {0,1}^n (MKP) ou sur l'ensemble des séquences (SSP).
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
