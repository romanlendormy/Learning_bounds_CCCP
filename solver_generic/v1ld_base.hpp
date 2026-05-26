#pragma once
/*
 * v1ld_base.hpp — Interface abstraite V1LDProblem + struct résultat.
 *
 * Implémenter solve_theta et solve_zeta suffit pour brancher un nouveau
 * problème dans le solveur générique solve_v1ld.
 *
 * Convention :
 *   • x est représenté comme un vecteur réel aplati de dimension n_vars()
 *   • l'objectif est linéaire : cost^T x
 *   • λ ∈ R^{N × n_vars}, un vecteur par scénario
 */

#include "types.hpp"
#include <utility>

// ── Résultat ──────────────────────────────────────────────────────────────────

struct V1LDResult {
    double v1LD_star;
    Vec2d  lambda_star;
    Vec1d  convergence;
    Vec1d  times_s;
    Vec1d  gnorm_history;
    Vec1d  z_changes;
    double solve_time_s;
    int    iterations_run;
};

// ── Classe abstraite ──────────────────────────────────────────────────────────

class V1LDProblem {
public:
    virtual ~V1LDProblem() = default;

    virtual int           n_vars()  const = 0;   // dimension de x aplati
    virtual int           N_scen()  const = 0;   // nombre de scénarios
    virtual double        epsilon() const = 0;
    virtual const Vec1d&  p()       const = 0;   // probabilités (N,)
    virtual const Vec1d&  c()       const = 0;   // coûts déterministes (n_vars,)

    // θ : max_{x ∈ S} cost^T x  — ensemble faisable sans contrainte stochastique
    virtual std::pair<double, Vec1d> solve_theta(const Vec1d& cost) const = 0;

    // ζ : max_{x ∈ X^i} cost^T x  — contraintes du scénario i
    virtual std::pair<double, Vec1d> solve_zeta(int i, const Vec1d& cost) const = 0;
};
