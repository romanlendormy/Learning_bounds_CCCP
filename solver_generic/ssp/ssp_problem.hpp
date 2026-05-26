#pragma once
/*
 * ssp_problem.hpp — V1LDProblem pour le CCSSP à contraintes automates.
 *
 * x ∈ {0,1}^{n×m} one-hot par période, aplati en (n*m,).
 * Profit déterministe profit[t][a] → objectif Σ_{t,a} cost[t,a] x[t,a].
 *
 * solve_theta : greedy par période — argmax_a cost[t,a]        O(n·m)
 * solve_zeta  : DP sur l'automate produit des k DFAs          O(n · Q^k · m)
 *
 * Optimisation : tables de transition des automates produits (Q^k × m)
 * précalculées une fois dans le constructeur (O(N · Q^k · m · k)).
 */

#include "v1ld_base.hpp"
#include <vector>

// ── Structures de données ─────────────────────────────────────────────────────

struct Automaton {
    Vec2i delta;              // (Q, m)  —  delta[q][a] ∈ {-1,...,Q-1}
    std::vector<bool> accepting; // (Q,)
};

struct SSPScenario {
    std::vector<Automaton> automata;  // k automates
};

// ── Classe concrète ───────────────────────────────────────────────────────────

class SSPProblem : public V1LDProblem {
public:
    /*
     * Paramètres :
     *   n, m, k, Q : dimensions du problème
     *   profit     : (n*m,) aplati (row = période, col = activité)
     *   p          : (N,) probabilités
     *   epsilon    : niveau de risque
     *   scenarios  : N scénarios, chacun avec k automates
     *
     * Le constructeur précalcule l'automate produit de chaque scénario.
     */
    SSPProblem(int n, int m, int k, int Q,
               Vec1d profit, Vec1d p, double epsilon,
               std::vector<SSPScenario> scenarios);

    int           n_vars()  const override { return n_ * m_; }
    int           N_scen()  const override { return N_; }
    double        epsilon() const override { return epsilon_; }
    const Vec1d&  p()       const override { return p_; }
    const Vec1d&  c()       const override { return c_; }  // = profit aplati

    std::pair<double, Vec1d> solve_theta(const Vec1d& cost) const override;
    std::pair<double, Vec1d> solve_zeta(int i, const Vec1d& cost) const override;

private:
    int    n_, m_, k_, Q_, N_;
    double epsilon_;
    Vec1d  c_;   // profit aplati (n*m,)
    Vec1d  p_;

    // Automate produit précalculé pour chaque scénario
    struct ProductAut {
        Vec2i             trans;  // (Q^k, m) — trans[s][a] ∈ {-1,...,Q^k-1}
        std::vector<bool> acc;    // (Q^k,)
    };
    std::vector<ProductAut> product_auts_;  // taille N
    int total_states_;                      // = Q^k

    int    encode(const Vec1i& qs) const;
    Vec1i  decode(int s)           const;
    ProductAut build_product_automaton(const SSPScenario& scen) const;
};
