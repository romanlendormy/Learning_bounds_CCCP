#pragma once
/*
 * ssp_cld_problem.hpp — V1CLDProblem pour le CCSSP à contraintes automates.
 *
 * Décomposition CLD : au lieu de l'automate produit (intersection), on résout
 * séparément chaque automate j du scénario i.  n_constraints() = k.
 *
 * solve_theta(j, cost) : greedy par période — argmax_a cost[t,a]   O(n·m)
 * solve_zeta(i,j,cost) : DP sur l'automate j seul                  O(n·Q·m)
 */

#include "v1cld_base.hpp"
#include "ssp_problem.hpp"   // Automaton, SSPScenario

class SSPCLDProblem : public V1CLDProblem {
public:
    SSPCLDProblem(int n, int m, int k, int Q,
                  Vec1d profit, Vec1d p, double epsilon,
                  std::vector<SSPScenario> scenarios);

    int           n_vars()        const override { return n_ * m_; }
    int           N_scen()        const override { return N_; }
    int           n_constraints() const override { return k_; }
    double        epsilon()       const override { return epsilon_; }
    const Vec1d&  p()             const override { return p_; }
    const Vec1d&  c()             const override { return c_; }

    std::pair<double, Vec1d> solve_theta(int j, const Vec1d& cost) const override;
    std::pair<double, Vec1d> solve_zeta(int i, int j, const Vec1d& cost) const override;

private:
    int    n_, m_, k_, Q_, N_;
    double epsilon_;
    Vec1d  c_;
    Vec1d  p_;
    std::vector<SSPScenario> scenarios_;
};
