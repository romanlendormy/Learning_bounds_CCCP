/*
 * ssp_cld2_problem.cpp — SSPCLD2Problem : v1CLD2, ζ_{i,j} décomposé exactement
 * par employé (E) et par période (T) via tables de valeurs avant/arrière.
 *
 * dp_forward(e,cost)  : F[t][q]  = meilleure valeur pour atteindre l'état q
 *                        à la frontière t en partant de l'état initial.
 * dp_backward(e,cost) : V[t][q]  = meilleure valeur pour aller de l'état q à
 *                        la frontière t jusqu'à un état acceptant à T.
 * work_val_at(e,j,·)  : max_{(q,W,q')} F[j][q] + cost[e,j,W] + V[j+1][q']
 *                        — identité exacte de décomposition avant/arrière
 *                        d'un plus long chemin dans un DAG : donne, terme à
 *                        terme, la même valeur que
 *                        SSPCLDProblem::dp_one_employee_forced_at(e,j,cost),
 *                        sans refaire de DP complète par période forcée.
 *
 * La sélection des employés forcés (tri des delta croissants) est inchangée
 * par rapport à SSPCLDProblem::solve_zeta — déjà exacte, déjà décomposée
 * par employé (problème de cardinalité sur un matroïde uniforme).
 */
#include "ssp_cld2_problem.hpp"
#include <algorithm>
#include <numeric>

static constexpr double NEG_INF_CLD2 = -1e30;

// ── Constructeur (identique à SSPCLDProblem) ──────────────────────────────────

SSPCLD2Problem::SSPCLD2Problem(
    const std::vector<std::tuple<int,int,int>>& transitions,
    const std::vector<int>&  accepting,
    int                      initial,
    const Vec1d&             salary,
    const Vec2i&             demand,
    Vec1d                    p,
    double                   epsilon,
    int                      E
)
    : E_(E),
      T_(static_cast<int>(salary.size())),
      N_(static_cast<int>(p.size())),
      epsilon_(epsilon),
      p_(std::move(p)),
      initial_(initial),
      demand_(demand)
{
    Q_ = initial + 1;
    for (auto& [s, sym, d] : transitions)
        Q_ = std::max(Q_, std::max(s, d) + 1);
    for (int q : accepting)
        Q_ = std::max(Q_, q + 1);

    transitions_ = transitions;
    std::sort(transitions_.begin(), transitions_.end());

    accepting_sorted_ = accepting;
    std::sort(accepting_sorted_.begin(), accepting_sorted_.end());

    adj_.assign(Q_, {});
    for (auto& [src, sym, dst] : transitions_)
        adj_[src].push_back({sym, dst});
    for (auto& lst : adj_)
        std::sort(lst.begin(), lst.end());

    c_.assign(E_ * T_ * SIGMA_SIZE, 0.0);
    for (int e = 0; e < E_; ++e)
        for (int t = 0; t < T_; ++t)
            c_[e * T_ * SIGMA_SIZE + t * SIGMA_SIZE + SIGMA_W] = -salary[t];
}

// ── DP pour un employé e (sans contrainte sur sigma) — recopie exacte de ────
// ── SSPCLDProblem::dp_one_employee (utilisée par solve_theta et free_val) ───

std::pair<double, std::vector<int>>
SSPCLD2Problem::dp_one_employee(int e, const Vec1d& cost) const
{
    const int base = e * T_ * SIGMA_SIZE;

    Vec1d dp(Q_, NEG_INF_CLD2);
    dp[initial_] = 0.0;

    std::vector<int> bp_prev(T_ * Q_, -1);
    std::vector<int> bp_sym (T_ * Q_, -1);
    Vec1d dp_new(Q_);

    for (int t = 0; t < T_; ++t) {
        std::fill(dp_new.begin(), dp_new.end(), NEG_INF_CLD2);
        for (int q = 0; q < Q_; ++q) {
            if (dp[q] <= NEG_INF_CLD2 / 2) continue;
            for (auto [sigma, q_next] : adj_[q]) {
                double v = dp[q] + cost[base + t * SIGMA_SIZE + sigma];
                if (v > dp_new[q_next]) {
                    dp_new[q_next]           = v;
                    bp_prev[t * Q_ + q_next] = q;
                    bp_sym [t * Q_ + q_next] = sigma;
                }
            }
        }
        std::swap(dp, dp_new);
    }

    double best_val = NEG_INF_CLD2;
    int    best_q   = -1;
    for (int q : accepting_sorted_)
        if (dp[q] > best_val) { best_val = dp[q]; best_q = q; }

    // Fallback si le DFA est vide pour ce T (même comportement que SSPCLDProblem)
    if (best_q < 0) {
        std::vector<int> actions(T_);
        double val = 0.0;
        for (int t = 0; t < T_; ++t) {
            int    best_s = 0;
            double best_c = cost[base + t * SIGMA_SIZE];
            for (int s = 1; s < SIGMA_SIZE; ++s) {
                double cv = cost[base + t * SIGMA_SIZE + s];
                if (cv > best_c) { best_c = cv; best_s = s; }
            }
            actions[t] = best_s;
            val       += best_c;
        }
        return {val, std::move(actions)};
    }

    std::vector<int> actions(T_);
    int q = best_q;
    for (int t = T_ - 1; t >= 0; --t) {
        actions[t] = bp_sym [t * Q_ + q];
        q          = bp_prev[t * Q_ + q];
    }
    return {best_val, std::move(actions)};
}

// ── Table avant F[t][q] : meilleure valeur de l'initial jusqu'à (t,q) ────────

SSPCLD2Problem::Forward
SSPCLD2Problem::dp_forward(int e, const Vec1d& cost) const
{
    const int base = e * T_ * SIGMA_SIZE;

    Forward fwd;
    fwd.F.assign(T_ + 1, Vec1d(Q_, NEG_INF_CLD2));
    fwd.bp_prev.assign(T_ * Q_, -1);
    fwd.bp_sym .assign(T_ * Q_, -1);
    fwd.F[0][initial_] = 0.0;

    for (int t = 0; t < T_; ++t) {
        for (int q = 0; q < Q_; ++q) {
            if (fwd.F[t][q] <= NEG_INF_CLD2 / 2) continue;
            for (auto [sigma, q_next] : adj_[q]) {
                double v = fwd.F[t][q] + cost[base + t * SIGMA_SIZE + sigma];
                if (v > fwd.F[t + 1][q_next]) {
                    fwd.F[t + 1][q_next]        = v;
                    fwd.bp_prev[t * Q_ + q_next] = q;
                    fwd.bp_sym [t * Q_ + q_next] = sigma;
                }
            }
        }
    }
    return fwd;
}

// ── Table arrière V[t][q] : meilleure valeur de (t,q) jusqu'à un acceptant ──

SSPCLD2Problem::Backward
SSPCLD2Problem::dp_backward(int e, const Vec1d& cost) const
{
    const int base = e * T_ * SIGMA_SIZE;

    Backward bwd;
    bwd.V.assign(T_ + 1, Vec1d(Q_, NEG_INF_CLD2));
    bwd.bp_sym .assign(T_ * Q_, -1);
    bwd.bp_next.assign(T_ * Q_, -1);
    for (int q : accepting_sorted_)
        bwd.V[T_][q] = 0.0;

    for (int t = T_ - 1; t >= 0; --t) {
        for (int q = 0; q < Q_; ++q) {
            double best_val = NEG_INF_CLD2;
            int    best_sym = -1, best_next = -1;
            for (auto [sigma, q_next] : adj_[q]) {
                if (bwd.V[t + 1][q_next] <= NEG_INF_CLD2 / 2) continue;
                double v = cost[base + t * SIGMA_SIZE + sigma] + bwd.V[t + 1][q_next];
                if (v > best_val) { best_val = v; best_sym = sigma; best_next = q_next; }
            }
            bwd.V[t][q]              = best_val;
            bwd.bp_sym [t * Q_ + q]  = best_sym;
            bwd.bp_next[t * Q_ + q]  = best_next;
        }
    }
    return bwd;
}

// ── work_val_e(j) : sous-problème atomique T=1, E=1 ──────────────────────────

std::pair<double, std::vector<int>>
SSPCLD2Problem::work_val_at(int e, int j, const Vec1d& cost,
                             const Forward& fwd, const Backward& bwd) const
{
    const int base = e * T_ * SIGMA_SIZE;

    double best_val = NEG_INF_CLD2;
    int    best_src = -1, best_dst = -1;
    for (auto& [src, sym, dst] : transitions_) {
        if (sym != SIGMA_W) continue;
        if (fwd.F[j][src]     <= NEG_INF_CLD2 / 2) continue;
        if (bwd.V[j + 1][dst] <= NEG_INF_CLD2 / 2) continue;
        double v = fwd.F[j][src] + cost[base + j * SIGMA_SIZE + SIGMA_W] + bwd.V[j + 1][dst];
        if (v > best_val) { best_val = v; best_src = src; best_dst = dst; }
    }
    if (best_src < 0) return {NEG_INF_CLD2, {}};  // infaisable pour cet employé

    std::vector<int> actions(T_);
    actions[j] = SIGMA_W;

    int q = best_src;
    for (int t = j - 1; t >= 0; --t) {
        actions[t] = fwd.bp_sym [t * Q_ + q];
        q          = fwd.bp_prev[t * Q_ + q];
    }
    q = best_dst;
    for (int t = j + 1; t < T_; ++t) {
        actions[t] = bwd.bp_sym [t * Q_ + q];
        q          = bwd.bp_next[t * Q_ + q];
    }
    return {best_val, std::move(actions)};
}

// ── solve_theta : identique à SSPCLDProblem (j ignoré) ───────────────────────

std::pair<double, Vec1d>
SSPCLD2Problem::solve_theta(int /*j*/, const Vec1d& cost) const
{
    Vec1d  x(n_vars(), 0.0);
    double total = 0.0;
    for (int e = 0; e < E_; ++e) {
        auto [val, actions] = dp_one_employee(e, cost);
        total += val;
        for (int t = 0; t < T_; ++t)
            x[e * T_ * SIGMA_SIZE + t * SIGMA_SIZE + actions[t]] = 1.0;
    }
    return {total, std::move(x)};
}

// ── solve_zeta : glouton sur delta_e(j), work_val_e(j) via F/V précalculés ──

std::pair<double, Vec1d>
SSPCLD2Problem::solve_zeta(int i, int j, const Vec1d& cost) const
{
    const int d_it = demand_[i][j];

    if (d_it <= 0) return solve_theta(j, cost);
    if (d_it > E_) return {NEG_INF_CLD2, Vec1d(n_vars(), 0.0)};

    std::vector<double>           free_val(E_), work_val(E_);
    std::vector<std::vector<int>> free_actions(E_), work_actions(E_);

    for (int e = 0; e < E_; ++e) {
        {
            auto [fv, fa] = dp_one_employee(e, cost);
            free_val[e]     = fv;
            free_actions[e] = std::move(fa);
        }
        {
            Forward  fwd = dp_forward(e, cost);
            Backward bwd = dp_backward(e, cost);
            auto [wv, wa] = work_val_at(e, j, cost, fwd, bwd);
            work_val[e]     = wv;
            work_actions[e] = std::move(wa);
        }
    }

    // Employés capables de travailler à t=j selon le DFA
    std::vector<int> feasible;
    feasible.reserve(E_);
    for (int e = 0; e < E_; ++e)
        if (work_val[e] > NEG_INF_CLD2 / 2)
            feasible.push_back(e);

    if (static_cast<int>(feasible.size()) < d_it)
        return {NEG_INF_CLD2, Vec1d(n_vars(), 0.0)};

    // Trier par delta = free_val − work_val croissant (pénalité minimale en premier)
    std::sort(feasible.begin(), feasible.end(), [&](int a, int b) {
        return (free_val[a] - work_val[a]) < (free_val[b] - work_val[b]);
    });

    std::vector<bool> forced(E_, false);
    for (int k = 0; k < d_it; ++k)
        forced[feasible[k]] = true;

    Vec1d  x(n_vars(), 0.0);
    double total = 0.0;
    for (int e = 0; e < E_; ++e) {
        if (forced[e]) {
            total += work_val[e];
            for (int t = 0; t < T_; ++t)
                x[e * T_ * SIGMA_SIZE + t * SIGMA_SIZE + work_actions[e][t]] = 1.0;
        } else {
            total += free_val[e];
            for (int t = 0; t < T_; ++t)
                x[e * T_ * SIGMA_SIZE + t * SIGMA_SIZE + free_actions[e][t]] = 1.0;
        }
    }
    return {total, std::move(x)};
}
