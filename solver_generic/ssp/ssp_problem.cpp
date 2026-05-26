/*
 * ssp_problem.cpp — Implémentation de SSPProblem.
 *
 * DP sur l'automate produit : O(n · Q^k · m) par appel solve_zeta.
 * Pour Q=20, k=2, m=10, n=50 : ~200K ops/appel → ~0.1-0.5 ms en C++
 */

#include "ssp_problem.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

// ── Constructeur ──────────────────────────────────────────────────────────────

SSPProblem::SSPProblem(int n, int m, int k, int Q,
                       Vec1d profit, Vec1d p, double epsilon,
                       std::vector<SSPScenario> scenarios)
    : n_(n), m_(m), k_(k), Q_(Q),
      N_(static_cast<int>(p.size())),
      epsilon_(epsilon),
      c_(std::move(profit)), p_(std::move(p))
{
    // total_states = Q^k
    total_states_ = 1;
    for (int j = 0; j < k_; ++j) total_states_ *= Q_;

    // Précalcul des automates produit
    product_auts_.reserve(N_);
    for (const auto& scen : scenarios)
        product_auts_.push_back(build_product_automaton(scen));
}

// ── encode / decode ───────────────────────────────────────────────────────────

int SSPProblem::encode(const Vec1i& qs) const {
    int s = 0;
    for (int j = 0; j < k_; ++j) s = s * Q_ + qs[j];
    return s;
}

Vec1i SSPProblem::decode(int s) const {
    Vec1i qs(k_);
    for (int j = k_ - 1; j >= 0; --j) { qs[j] = s % Q_; s /= Q_; }
    return qs;
}

// ── Précalcul de l'automate produit ──────────────────────────────────────────

SSPProblem::ProductAut
SSPProblem::build_product_automaton(const SSPScenario& scen) const
{
    int S = total_states_;
    ProductAut pa;
    pa.trans.assign(S, Vec1i(m_, -1));
    pa.acc.assign(S, false);

    for (int s = 0; s < S; ++s) {
        Vec1i qs = decode(s);
        // état acceptant si tous les automates acceptent
        bool all_acc = true;
        for (int j = 0; j < k_; ++j)
            if (!scen.automata[j].accepting[qs[j]]) { all_acc = false; break; }
        pa.acc[s] = all_acc;

        for (int a = 0; a < m_; ++a) {
            Vec1i next_qs(k_);
            bool valid = true;
            for (int j = 0; j < k_; ++j) {
                int q_next = scen.automata[j].delta[qs[j]][a];
                if (q_next < 0) { valid = false; break; }
                next_qs[j] = q_next;
            }
            if (valid) pa.trans[s][a] = encode(next_qs);
        }
    }
    return pa;
}

// ── θ_i : greedy par période ──────────────────────────────────────────────────

std::pair<double, Vec1d> SSPProblem::solve_theta(const Vec1d& cost) const
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

// ── ζ_i : DP sur l'automate produit ──────────────────────────────────────────

std::pair<double, Vec1d> SSPProblem::solve_zeta(int i, const Vec1d& cost) const
{
    const ProductAut& pa = product_auts_[i];
    int S = total_states_;
    constexpr double NEG_INF = -1e30;

    Vec1d dp(S, NEG_INF);
    dp[0] = 0.0;   // état initial : (0,...,0) → indice 0

    // Tableaux de reconstruction : best_action[t*S + s] et prev_state[t*S + s]
    std::vector<int16_t> best_action(n_ * S, -1);
    std::vector<int32_t> prev_state(n_ * S, -1);

    Vec1d dp_new(S);

    for (int t = 0; t < n_; ++t) {
        std::fill(dp_new.begin(), dp_new.end(), NEG_INF);
        int16_t* ba = best_action.data() + t * S;
        int32_t* ps = prev_state.data()  + t * S;

        for (int s = 0; s < S; ++s) {
            if (dp[s] < NEG_INF / 2) continue;   // état non atteignable
            const Vec1i& tr = pa.trans[s];
            for (int a = 0; a < m_; ++a) {
                int s_next = tr[a];
                if (s_next < 0) continue;
                double v = dp[s] + cost[t * m_ + a];
                if (v > dp_new[s_next]) {
                    dp_new[s_next] = v;
                    ba[s_next]     = static_cast<int16_t>(a);
                    ps[s_next]     = static_cast<int32_t>(s);
                }
            }
        }
        std::swap(dp, dp_new);
    }

    // Meilleur état final acceptant
    double best_val   = NEG_INF;
    int    best_state = -1;
    for (int s = 0; s < S; ++s) {
        if (pa.acc[s] && dp[s] > best_val) {
            best_val   = dp[s];
            best_state = s;
        }
    }
    if (best_state < 0)
        return solve_theta(cost);   // aucune séquence faisable — fallback

    // Reconstruction de la séquence optimale par remontée
    std::vector<int> actions(n_);
    int s = best_state;
    for (int t = n_ - 1; t >= 0; --t) {
        actions[t] = best_action[t * S + s];
        s          = prev_state[t * S + s];
    }

    Vec1d x(n_ * m_, 0.0);
    for (int t = 0; t < n_; ++t)
        x[t * m_ + actions[t]] = 1.0;

    return {best_val, std::move(x)};
}
