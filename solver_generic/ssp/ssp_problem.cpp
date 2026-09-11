/*
 * ssp_problem.cpp — SSPShiftProblem : oracles SLD pour le SSP multi-employés.
 *
 * Chaîne mathématique
 * -------------------
 * SLD dualise la contrainte de quantile (un multiplicateur λ_i par scénario).
 * Le Lagrangien L1(λ) = Σ_i p_i θ_i − min_y Σ p_i δ_i y_i  s.c.  Σ p_i y_i ≥ 1−ε.
 *
 * θ_i(λ) = max_{x ∈ BASE} cost_i^T x      — domaine de base (automate, aucune couverture)
 * ζ_i(λ) = max_{x ∈ X^i}  cost_i^T x      — domaine scénario i (automate + couverture d[:,i])
 * δ_i    = max(0, θ_i − ζ_i) ≥ 0          — coût de faisabilité (garanti ≥ 0 car X^i ⊆ BASE)
 *
 * Pourquoi θ décompose par employé :
 *   La contrainte "regular(x[e,·], DFA) ∀e" est séparable par e.
 *   L'objectif cost^T x = Σ_e Σ_t Σ_σ cost[e,t,σ] x[e,t,σ] est aussi séparable.
 *   → E sous-problèmes indépendants, chacun résolu par DP topologique sur le DAG.
 *
 * Pourquoi ζ doit être exact :
 *   La couverture Σ_e x[e,t,W] ≥ d[t,i] couple les E employés → NP-difficile (E>1).
 *   Une solution approchée rendrait la borne SLD invalide (L1 ne serait plus une
 *   borne supérieure) et casserait la correction d'un B&B downstream.
 *   On utilise donc OR-Tools CP-SAT qui garantit l'optimalité.
 *
 * Poids négatifs :
 *   Le vecteur cost = c + λ_i/p_i − Λ_sum peut avoir des entrées négatives.
 *   La DP topologique (ordre par couche temporelle t=0..T-1) gère nativement
 *   les poids négatifs — aucun besoin de Bellman-Ford.
 */

#include "ssp_problem.hpp"
#include "ortools/sat/cp_model.h"
#include "ortools/sat/sat_parameters.pb.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <iostream>

namespace sat = operations_research::sat;
using operations_research::Domain;

static constexpr double NEG_INF      = -1e30;
static constexpr int64_t CPSAT_SCALE = 1'000'000;  // 6 décimales de précision

// ── Constructeur ──────────────────────────────────────────────────────────────

SSPShiftProblem::SSPShiftProblem(
    const std::vector<std::tuple<int,int,int>>& transitions,
    const std::vector<int>&  accepting,
    int                      initial,
    const Vec1d&             salary,
    const Vec2i&             demand,
    Vec1d                    p,
    double                   epsilon,
    int E
)
    : E_(E),
      T_(static_cast<int>(salary.size())),
      N_(static_cast<int>(p.size())),
      epsilon_(epsilon),
      p_(std::move(p)),
      initial_(initial),
      demand_(demand)
{
    // Calcul de Q = nombre d'états = max index + 1
    Q_ = initial + 1;
    for (auto& [s, sym, d] : transitions)
        Q_ = std::max(Q_, std::max(s, d) + 1);
    for (int q : accepting)
        Q_ = std::max(Q_, q + 1);

    // Tri des transitions pour le déterminisme
    transitions_ = transitions;
    std::sort(transitions_.begin(), transitions_.end());

    // États acceptants triés (CP-SAT AddAutomaton prend Span<const int>)
    accepting_sorted_ = accepting;
    std::sort(accepting_sorted_.begin(), accepting_sorted_.end());

    // Liste d'adjacence par état source
    adj_.assign(Q_, {});
    for (auto& [src, sym, dst] : transitions_)
        adj_[src].push_back({sym, dst});
    for (auto& lst : adj_)
        std::sort(lst.begin(), lst.end());  // ordre (sym, dst) pour déterminisme

    // Coût de base : c[e,t,W] = −salary[t], autres = 0
    // (maximiser −salary équivaut à minimiser salary)
    c_.assign(E_ * T_ * SSP_SIGMA_SIZE, 0.0);
    for (int e = 0; e < E_; ++e)
        for (int t = 0; t < T_; ++t)
            c_[e * T_ * SSP_SIGMA_SIZE + t * SSP_SIGMA_SIZE + SSP_SIGMA_W] = -salary[t];
}

// ── DP pour un employé e ──────────────────────────────────────────────────────
//
// Calcule max_{x[e,·] ∈ DFA} cost_{e}^T x[e,·] par DP topologique.
// Le DAG est déroulé sur T couches temporelles (t=0..T-1).
// Complexité : O(T · Q · |transitions_par_état|) ≤ O(T · Q · 4).
//
// Poids négatifs supportés nativement : on n'utilise pas de DP min-chemin
// sur graphe général (Bellman-Ford non nécessaire car le DAG est acyclique).
//
// Retourne (valeur_optimale, actions[T]) où actions[t] = σ choisi à l'étape t.

std::pair<double, std::vector<int>>
SSPShiftProblem::dp_one_employee(int e, const Vec1d& cost, const Vec1i* mask_row) const
{
    const int base = e * T_ * SSP_SIGMA_SIZE;

    // dp[q] = meilleure valeur pour atteindre l'état q après les t symboles lus
    Vec1d dp(Q_, NEG_INF);
    dp[initial_] = 0.0;

    // Tableaux de backtracking
    // bp_prev[t * Q_ + q] = état précédent   (−1 si non atteint)
    // bp_sym[t * Q_ + q]  = symbole pris     (−1 si non atteint)
    std::vector<int> bp_prev(T_ * Q_, -1);
    std::vector<int> bp_sym (T_ * Q_, -1);

    Vec1d dp_new(Q_);

    for (int t = 0; t < T_; ++t) {
        std::fill(dp_new.begin(), dp_new.end(), NEG_INF);
        const int forced = (mask_row && (*mask_row)[t] >= 0) ? (*mask_row)[t] : -1;

        for (int q = 0; q < Q_; ++q) {
            if (dp[q] <= NEG_INF / 2) continue;  // état non atteignable

            for (auto [sigma, q_next] : adj_[q]) {
                if (forced >= 0 && sigma != forced) continue;  // période masquée : symbole imposé
                double v = dp[q] + cost[base + t * SSP_SIGMA_SIZE + sigma];
                if (v > dp_new[q_next]) {
                    dp_new[q_next]                  = v;
                    bp_prev[t * Q_ + q_next] = q;
                    bp_sym [t * Q_ + q_next] = sigma;
                }
            }
        }

        std::swap(dp, dp_new);
    }

    // Meilleur état acceptant après T étapes
    double best_val = NEG_INF;
    int    best_q   = -1;
    for (int q : accepting_sorted_) {
        if (dp[q] > best_val) {
            best_val = dp[q];
            best_q   = q;
        }
    }

    // Aucun état acceptant atteint : ce masque (mask_row) n'admet AUCUNE
    // séquence valide de longueur T_ pour cet employé — sous-problème
    // réellement infaisable (pas juste "difficile"). Renvoyer NEG_INF
    // (plutôt qu'ignorer l'automate/le masque comme avant) : toute autre
    // valeur romprait l'exactitude requise par la borne SLD, cf.
    // debug_verify_oracle. Le vecteur d'actions reste de taille T_ (valeurs
    // arbitraires, jamais consommées puisque la valeur -inf exclut cet
    // employé de toute sélection gloutonne en aval).
    if (best_q < 0) {
        return {NEG_INF, std::vector<int>(T_, 0)};
    }

    // Reconstruction par remontée du backtracking
    std::vector<int> actions(T_);
    int q = best_q;
    for (int t = T_ - 1; t >= 0; --t) {
        actions[t] = bp_sym [t * Q_ + q];
        q          = bp_prev[t * Q_ + q];
    }

    return {best_val, std::move(actions)};
}

// ── solve_theta : E DPs indépendants sur le DAG déroulé ──────────────────────
//
// Domaine de base = { x : regular(x[e,·], DFA) ∀e } — aucune contrainte de couverture.
// Les E sous-problèmes sont indépendants car le seul lien entre employés viendrait
// de la couverture, qui n'est PAS dans le domaine de base.

std::pair<double, Vec1d> SSPShiftProblem::solve_theta(const Vec1d& cost) const
{
    return solve_theta_masked(cost, nullptr);
}

std::pair<double, Vec1d>
SSPShiftProblem::solve_theta_masked(const Vec1d& cost, const Vec2i* mask) const
{
    Vec1d  x(n_vars(), 0.0);
    double total = 0.0;

    for (int e = 0; e < E_; ++e) {
        const Vec1i* mask_row = mask ? &(*mask)[e] : nullptr;
        auto [val, actions] = dp_one_employee(e, cost, mask_row);
        total += val;
        for (int t = 0; t < T_; ++t)
            x[e * T_ * SSP_SIGMA_SIZE + t * SSP_SIGMA_SIZE + actions[t]] = 1.0;
    }

    return {total, std::move(x)};
}

// ── solve_zeta : CP-SAT multi-employés avec couverture ────────────────────────
//
// Domaine du scénario i = base ∩ { Σ_e x[e,t,W] ≥ d[t,i] ∀t }.
// La couverture couple les E employés → problème NP-difficile → CP-SAT exact.
//
// Modèle CP-SAT :
//   shift[e][t] ∈ {0,1,2,3}         — activité de l'employé e à la période t
//   AddAutomaton(shift[e][·], DFA)   — contrainte de quart légal (par employé)
//   work[e][t] ∈ {0,1}              — indicateur shift[e][t]==3 (W)
//   Σ_e work[e][t] ≥ d[t,i]         — couverture de la demande
//   Objectif : max Σ_{e,t} cost_et[shift[e][t]]  (via AddElement, pas de BoolVar)

std::pair<double, Vec1d> SSPShiftProblem::solve_zeta(int i, const Vec1d& cost) const
{
    return solve_zeta_masked(i, cost, nullptr);
}

std::pair<double, Vec1d>
SSPShiftProblem::solve_zeta_masked(int i, const Vec1d& cost, const Vec2i* mask) const
{
    // Infaisabilité structurelle immédiate (demande > nombre d'employés)
    for (int t = 0; t < T_; ++t) {
        if (demand_[i][t] > E_)
            return {NEG_INF, Vec1d(n_vars(), 0.0)};
    }

    sat::CpModelBuilder model;

    // ── Variables : shift[e][t] ∈ {0,1,2,3} (ou fixé si masqué) ──────────────
    std::vector<std::vector<sat::IntVar>> shift(
        E_, std::vector<sat::IntVar>(T_));
    for (int e = 0; e < E_; ++e)
        for (int t = 0; t < T_; ++t) {
            int forced = (mask && (*mask)[e][t] >= 0) ? (*mask)[e][t] : -1;
            shift[e][t] = model.NewIntVar(
                forced >= 0 ? Domain(forced, forced) : Domain(0, SSP_SIGMA_SIZE - 1));
        }

    // ── Contraintes d'automate : AddAutomaton par employé ─────────────────────
    // AddAutomaton(vars, starting_state: int, final_states: Span<const int>)
    // AddTransition(tail: int, head: int, label: int64_t)  ordre = (src, dst, sym)
    for (int e = 0; e < E_; ++e) {
        std::vector<sat::IntVar> vars(T_);
        for (int t = 0; t < T_; ++t) vars[t] = shift[e][t];

        auto aut_c = model.AddAutomaton(vars, initial_, accepting_sorted_);
        for (auto& [src, sym, dst] : transitions_)
            aut_c.AddTransition(src, dst, static_cast<int64_t>(sym));
    }

    // ── Indicateurs de travail et contrainte de couverture ────────────────────
    std::vector<std::vector<sat::BoolVar>> work(
        E_, std::vector<sat::BoolVar>(T_));
    for (int e = 0; e < E_; ++e) {
        for (int t = 0; t < T_; ++t) {
            work[e][t] = model.NewBoolVar();
            // work[e][t] == 1  ↔  shift[e][t] == W
            model.AddEquality(shift[e][t], SSP_SIGMA_W).OnlyEnforceIf(work[e][t]);
            model.AddNotEqual(shift[e][t], SSP_SIGMA_W).OnlyEnforceIf(work[e][t].Not());
        }
    }

    for (int t = 0; t < T_; ++t) {
        if (demand_[i][t] <= 0) continue;
        std::vector<sat::BoolVar> work_at_t;
        work_at_t.reserve(E_);
        for (int e = 0; e < E_; ++e)
            work_at_t.push_back(work[e][t]);
        model.AddLinearConstraint(
            sat::LinearExpr::Sum(work_at_t),
            Domain(static_cast<int64_t>(demand_[i][t]),
                   static_cast<int64_t>(E_)));
    }

    // ── Objectif : max Σ_{e,t} cost_et[shift[e][t]] ─────────────────────────
    // AddElement(index, values, target) copie 'values' immédiatement dans le proto.
    // On réutilise cost_row à chaque itération (safe car copie immédiate).
    sat::LinearExpr obj;
    std::vector<int64_t> cost_row(SSP_SIGMA_SIZE);

    for (int e = 0; e < E_; ++e) {
        for (int t = 0; t < T_; ++t) {
            const int base = e * T_ * SSP_SIGMA_SIZE + t * SSP_SIGMA_SIZE;
            for (int s = 0; s < SSP_SIGMA_SIZE; ++s)
                cost_row[s] = static_cast<int64_t>(
                    std::llround(cost[base + s] * CPSAT_SCALE));

            int64_t lo = *std::min_element(cost_row.begin(), cost_row.end());
            int64_t hi = *std::max_element(cost_row.begin(), cost_row.end());

            sat::IntVar obj_et = model.NewIntVar(Domain(lo, hi));
            model.AddElement(shift[e][t],
                             absl::Span<const int64_t>(cost_row),
                             obj_et);
            obj += sat::LinearExpr(obj_et);
        }
    }
    model.Maximize(obj);

    // ── Résolution mono-thread, silencieuse ───────────────────────────────────
    sat::SatParameters params;
    params.set_log_search_progress(false);
    params.set_num_workers(1);
    const sat::CpSolverResponse resp =
        sat::SolveWithParameters(model.Build(), params);

    // ── Reconstruction de la solution ─────────────────────────────────────────
    if (resp.status() != sat::CpSolverStatus::OPTIMAL &&
        resp.status() != sat::CpSolverStatus::FEASIBLE) {
        // Scénario infaisable (Σ_e max_work < d[t,i] pour un t, ou DFA incompatible)
        return {NEG_INF, Vec1d(n_vars(), 0.0)};
    }

    Vec1d  x(n_vars(), 0.0);
    double val = 0.0;
    for (int e = 0; e < E_; ++e) {
        for (int t = 0; t < T_; ++t) {
            int sigma = static_cast<int>(
                sat::SolutionIntegerValue(resp, shift[e][t]));
            int idx = e * T_ * SSP_SIGMA_SIZE + t * SSP_SIGMA_SIZE + sigma;
            x[idx] = 1.0;
            val   += cost[idx];  // valeur exacte (non scalée)
        }
    }

    return {val, std::move(x)};
}
