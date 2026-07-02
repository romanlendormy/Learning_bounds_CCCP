/*
 * ssp_cld_problem.cpp — SSPCLDProblem : oracles v1CLD pour le SSP shift scheduling.
 *
 * Décomposition sur la période t : m = T contraintes, une par période.
 *
 * solve_theta(j, cost)
 *   Identique à SSPShiftProblem::solve_theta (j ignoré).
 *   E sous-problèmes indépendants résolus par DP topologique sur le DAG déroulé.
 *
 * solve_zeta(i, t, cost)
 *   Sous-problème : max cost^T x  s.t.  DFA(x[e,·]) ∀e,  Σ_e x[e,t,W] ≥ d[t,i].
 *   Algorithme greedy (optimal) :
 *     1. free_val[e]  = DP sans contrainte sur sigma à t.
 *     2. work_val[e]  = DP avec W forcé à t  (infaisable → NEG_INF).
 *     3. delta[e]     = free_val[e] − work_val[e] ≥ 0.
 *     4. Trier par delta croissant, forcer les d[t,i] premiers.
 *   Preuve d'optimalité : total(S) = Σ free_val − Σ_{e∈S} delta[e],
 *   le problème se réduit à minimiser Σ_{e∈S} delta[e] sur les ensembles S de
 *   taille d_it parmi les employés faisables → tri glouton exact.
 *   Complexité : O(E · T · Q · 4)  vs. CP-SAT (NP-difficile quand T > 1).
 */
#include "ssp_cld_problem.hpp"
#include <algorithm>
#include <numeric>

static constexpr double NEG_INF_CLD = -1e30;

// ── Constructeur ──────────────────────────────────────────────────────────────

SSPCLDProblem::SSPCLDProblem(
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

    // c[e,t,W] = −salary[t], autres = 0
    c_.assign(E_ * T_ * SIGMA_SIZE, 0.0);
    for (int e = 0; e < E_; ++e)
        for (int t = 0; t < T_; ++t)
            c_[e * T_ * SIGMA_SIZE + t * SIGMA_SIZE + SIGMA_W] = -salary[t];
}

// ── DP pour un employé e (sans contrainte sur sigma) ─────────────────────────

std::pair<double, std::vector<int>>
SSPCLDProblem::dp_one_employee(int e, const Vec1d& cost, const Vec1i* mask_row) const
{
    const int base = e * T_ * SIGMA_SIZE;

    Vec1d dp(Q_, NEG_INF_CLD);
    dp[initial_] = 0.0;

    std::vector<int> bp_prev(T_ * Q_, -1);
    std::vector<int> bp_sym (T_ * Q_, -1);
    Vec1d dp_new(Q_);

    for (int t = 0; t < T_; ++t) {
        std::fill(dp_new.begin(), dp_new.end(), NEG_INF_CLD);
        const int forced = (mask_row && (*mask_row)[t] >= 0) ? (*mask_row)[t] : -1;
        for (int q = 0; q < Q_; ++q) {
            if (dp[q] <= NEG_INF_CLD / 2) continue;
            for (auto [sigma, q_next] : adj_[q]) {
                if (forced >= 0 && sigma != forced) continue;  // période masquée : symbole imposé
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

    double best_val = NEG_INF_CLD;
    int    best_q   = -1;
    for (int q : accepting_sorted_)
        if (dp[q] > best_val) { best_val = dp[q]; best_q = q; }

    // Fallback si le DFA est vide pour ce T (même comportement que SSPShiftProblem)
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

// ── DP pour un employé e avec W forcé à la période t_force ───────────────────
// Retourne {NEG_INF, {}} si aucun chemin DFA de longueur T avec W à t_force
// n'existe (employé incapable de travailler à t_force selon l'automate).

std::pair<double, std::vector<int>>
SSPCLDProblem::dp_one_employee_forced_at(int e, int t_force, const Vec1d& cost,
                                          const Vec1i* mask_row) const
{
    const int base = e * T_ * SIGMA_SIZE;

    Vec1d dp(Q_, NEG_INF_CLD);
    dp[initial_] = 0.0;

    std::vector<int> bp_prev(T_ * Q_, -1);
    std::vector<int> bp_sym (T_ * Q_, -1);
    Vec1d dp_new(Q_);

    for (int t = 0; t < T_; ++t) {
        std::fill(dp_new.begin(), dp_new.end(), NEG_INF_CLD);
        // t_force : W obligatoire. Sinon, respecte le masque éventuel (résiduel B&B).
        const int forced = (t == t_force)
            ? SIGMA_W
            : ((mask_row && (*mask_row)[t] >= 0) ? (*mask_row)[t] : -1);
        for (int q = 0; q < Q_; ++q) {
            if (dp[q] <= NEG_INF_CLD / 2) continue;
            for (auto [sigma, q_next] : adj_[q]) {
                if (forced >= 0 && sigma != forced) continue;
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

    double best_val = NEG_INF_CLD;
    int    best_q   = -1;
    for (int q : accepting_sorted_)
        if (dp[q] > best_val) { best_val = dp[q]; best_q = q; }

    if (best_q < 0) return {NEG_INF_CLD, {}};  // infaisable pour cet employé

    std::vector<int> actions(T_);
    int q = best_q;
    for (int t = T_ - 1; t >= 0; --t) {
        actions[t] = bp_sym [t * Q_ + q];
        q          = bp_prev[t * Q_ + q];
    }
    return {best_val, std::move(actions)};
}

// ── solve_theta : E DPs indépendants sur le DAG (j ignoré) ───────────────────

std::pair<double, Vec1d>
SSPCLDProblem::solve_theta(int j, const Vec1d& cost) const
{
    return solve_theta_masked(j, cost, nullptr);
}

std::pair<double, Vec1d>
SSPCLDProblem::solve_theta_masked(int /*j*/, const Vec1d& cost, const Vec2i* mask) const
{
    Vec1d  x(n_vars(), 0.0);
    double total = 0.0;
    for (int e = 0; e < E_; ++e) {
        const Vec1i* mask_row = mask ? &(*mask)[e] : nullptr;
        auto [val, actions] = dp_one_employee(e, cost, mask_row);
        total += val;
        for (int t = 0; t < T_; ++t)
            x[e * T_ * SIGMA_SIZE + t * SIGMA_SIZE + actions[t]] = 1.0;
    }
    return {total, std::move(x)};
}

// ── solve_zeta : greedy sur la contrainte de couverture à la période j ────────

std::pair<double, Vec1d>
SSPCLDProblem::solve_zeta(int i, int j, const Vec1d& cost) const
{
    return solve_zeta_masked(i, j, cost, nullptr);
}

// Généralisation masquée : un employé déjà fixé à la période j (mask[e][j] >= 0,
// par une décision de branchement B&B) n'est plus un candidat au tri glouton —
// sa contribution est déterministe (via son propre masque complet) et son
// indicateur W éventuel réduit d'autant la demande résiduelle à satisfaire par
// les employés encore libres à j. mask == nullptr reproduit exactement l'algorithme
// d'origine (tous les employés candidats, aucune couverture déjà acquise).
std::pair<double, Vec1d>
SSPCLDProblem::solve_zeta_masked(int i, int j, const Vec1d& cost, const Vec2i* mask) const
{
    const int d_it = demand_[i][j];

    // Couverture déjà acquise par les employés fixés à j (mask[e][j] == SIGMA_W)
    int already_covered = 0;
    if (mask)
        for (int e = 0; e < E_; ++e)
            if ((*mask)[e][j] == SIGMA_W) ++already_covered;

    const int d_eff = std::max(0, d_it - already_covered);
    if (d_eff <= 0) return solve_theta_masked(j, cost, mask);

    std::vector<double>           free_val(E_), work_val(E_, NEG_INF_CLD);
    std::vector<std::vector<int>> free_actions(E_), work_actions(E_);

    for (int e = 0; e < E_; ++e) {
        const Vec1i* mask_row = mask ? &(*mask)[e] : nullptr;
        {
            auto [fv, fa] = dp_one_employee(e, cost, mask_row);
            free_val[e]     = fv;
            free_actions[e] = std::move(fa);
        }
        const bool already_fixed_at_j = mask && (*mask)[e][j] >= 0;
        if (!already_fixed_at_j) {
            auto [wv, wa] = dp_one_employee_forced_at(e, j, cost, mask_row);
            work_val[e]     = wv;
            work_actions[e] = std::move(wa);
        }
        // Sinon : e est déjà décidé à j, il ne peut pas être candidat à la
        // sélection gloutonne (work_val[e] reste à NEG_INF_CLD, jamais choisi).
    }

    // Employés encore libres à j et capables d'y travailler selon le DFA
    std::vector<int> feasible;
    feasible.reserve(E_);
    for (int e = 0; e < E_; ++e) {
        const bool already_fixed_at_j = mask && (*mask)[e][j] >= 0;
        if (!already_fixed_at_j && work_val[e] > NEG_INF_CLD / 2)
            feasible.push_back(e);
    }

    if (static_cast<int>(feasible.size()) < d_eff)
        return {NEG_INF_CLD, Vec1d(n_vars(), 0.0)};

    // Trier par delta = free_val − work_val croissant (pénalité minimale en premier)
    std::sort(feasible.begin(), feasible.end(), [&](int a, int b) {
        return (free_val[a] - work_val[a]) < (free_val[b] - work_val[b]);
    });

    std::vector<bool> forced(E_, false);
    for (int k = 0; k < d_eff; ++k)
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
