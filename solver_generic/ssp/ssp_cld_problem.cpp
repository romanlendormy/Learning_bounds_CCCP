/*
 * ssp_cld_problem.cpp — Implémentation de SSPCLDProblem.
 *
 * solve_zeta(i, j, cost) : DP sur l'automate j seul du scénario i.
 * Complexité : O(n · Q · m) par appel (vs O(n · Q^k · m) pour v1LD).
 */

#include "ssp_cld_problem.hpp"
#include <algorithm>

// ── Constructeur ──────────────────────────────────────────────────────────────

SSPCLDProblem::SSPCLDProblem(int n, int m, int k, int Q,
                               Vec1d profit, Vec1d p, double epsilon,
                               std::vector<SSPScenario> scenarios)
    : n_(n), m_(m), k_(k), Q_(Q),
      N_(static_cast<int>(p.size())),
      epsilon_(epsilon),
      c_(std::move(profit)), p_(std::move(p)),
      scenarios_(std::move(scenarios))
{}

// ── θ : greedy par période (j ignoré) ────────────────────────────────────────

std::pair<double, Vec1d>
SSPCLDProblem::solve_theta(int /*j*/, const Vec1d& cost) const
{
    Vec1d x(n_ * m_, 0.0);
    double val = 0.0;
    for (int t = 0; t < n_; ++t) {
        int best_a = 0;
        double best_c = cost[t * m_];
        for (int a = 1; a < m_; ++a)
            if (cost[t * m_ + a] > best_c) { best_c = cost[t * m_ + a]; best_a = a; }
        x[t * m_ + best_a] = 1.0;
        val += best_c;
    }
    return {val, std::move(x)};
}

// ── ζ_{i,j} : DP sur l'automate j du scénario i ──────────────────────────────

std::pair<double, Vec1d>
SSPCLDProblem::solve_zeta(int i, int j, const Vec1d& cost) const
{
    const Automaton& aut = scenarios_[i].automata[j];
    constexpr double NEG_INF = -1e30;

    Vec1d dp(Q_, NEG_INF);
    dp[0] = 0.0;   // état initial : q = 0

    // Tables de reconstruction : ba[t*Q+q] = action choisie, ps[t*Q+q] = état précédent
    std::vector<int16_t> best_action(n_ * Q_, -1);
    std::vector<int32_t> prev_state (n_ * Q_, -1);

    Vec1d dp_new(Q_);

    for (int t = 0; t < n_; ++t) {
        std::fill(dp_new.begin(), dp_new.end(), NEG_INF);
        int16_t* ba = best_action.data() + t * Q_;
        int32_t* ps = prev_state.data()  + t * Q_;

        for (int q = 0; q < Q_; ++q) {
            if (dp[q] < NEG_INF / 2) continue;
            const Vec1i& tr = aut.delta[q];
            for (int a = 0; a < m_; ++a) {
                int q_next = tr[a];
                if (q_next < 0) continue;
                double v = dp[q] + cost[t * m_ + a];
                if (v > dp_new[q_next]) {
                    dp_new[q_next] = v;
                    ba[q_next]     = static_cast<int16_t>(a);
                    ps[q_next]     = static_cast<int32_t>(q);
                }
            }
        }
        std::swap(dp, dp_new);
    }

    // Meilleur état final acceptant
    double best_val   = NEG_INF;
    int    best_state = -1;
    for (int q = 0; q < Q_; ++q) {
        if (aut.accepting[q] && dp[q] > best_val) {
            best_val   = dp[q];
            best_state = q;
        }
    }
    if (best_state < 0)
        return solve_theta(j, cost);   // aucune séquence faisable — fallback

    // Reconstruction de la séquence par remontée
    std::vector<int> actions(n_);
    int q = best_state;
    for (int t = n_ - 1; t >= 0; --t) {
        actions[t] = best_action[t * Q_ + q];
        q          = prev_state [t * Q_ + q];
    }

    Vec1d x(n_ * m_, 0.0);
    for (int t = 0; t < n_; ++t)
        x[t * m_ + actions[t]] = 1.0;

    return {best_val, std::move(x)};
}
