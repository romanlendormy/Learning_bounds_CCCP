#pragma once
/*
 * kp_problem.hpp — V1LDProblem pour le knapsack chance-contraint (CC-KP, m=1).
 *
 * Cas particulier m=1 du MKP : x ∈ {0,1}^n, c déterministe, a_i^T x ≤ b_i
 * stochastique (une seule contrainte par scénario, pas m comme pour MKPProblem).
 *
 * solve_theta : greedy — x_k = 1 ssi cost_k > 0                       O(n)
 * solve_zeta  : knapsack 0/1 exact 1-contrainte — B&B (Dantzig), réel  O(B&B)
 *               (réutilise solve_zeta_i_exact_bb de mkp_subproblems.hpp)
 *
 * Pourquoi pas CP-SAT (comme MKPProblem à m>1) : à une seule contrainte, le
 * B&B trié-par-ratio + borne de Dantzig est exact, ne nécessite aucun arrondi
 * entier (contrairement à un DP), et n'a pas le coût fixe de construction de
 * modèle + démarrage solveur de CP-SAT — plusieurs ordres de grandeur plus
 * rapide pour n petit, appelé N × T fois par la descente de sous-gradient,
 * à chaque nœud du B&B externe. Cf. kp_problem_dp.hpp pour la variante DP
 * (arrondie, opt-in).
 */

#include "v1ld_base.hpp"

class KPProblem : public V1LDProblem {
public:
    // c : (n,)   p : (N,)   A : (N, n)   b : (N,)
    KPProblem(Vec1d c, Vec1d p, Vec2d A, Vec1d b, double epsilon);

    int           n_vars()  const override { return n_; }
    int           N_scen()  const override { return N_; }
    double        epsilon() const override { return epsilon_; }
    const Vec1d&  p()       const override { return p_; }
    const Vec1d&  c()       const override { return c_; }

    std::pair<double, Vec1d> solve_theta(const Vec1d& cost) const override;
    std::pair<double, Vec1d> solve_zeta(int i, const Vec1d& cost) const override;

protected:
    int    n_, N_;
    double epsilon_;
    Vec1d  c_, p_;
    Vec1d  b_;    // (N,)
    Vec2d  A_;    // (N, n)
};
