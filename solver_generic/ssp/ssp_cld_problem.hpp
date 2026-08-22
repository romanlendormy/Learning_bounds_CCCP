#pragma once
/*
 * SSPCLDProblem — V1CLDProblem pour le SSP shift scheduling.
 *
 * Décomposition CLD sur les périodes t ∈ {0,...,T-1} : m = T contraintes.
 * La contrainte j=t du scénario i est la couverture à la période t :
 *   Σ_e x[e,t,W] ≥ d[t,i]
 *
 * solve_theta(j, cost) : E DPs indépendants sur le DAG déroulé (j ignoré).
 * solve_zeta(i, t, cost) : greedy polynomial O(E·T·Q) :
 *   - Pour chaque employé e : free_val (DP libre) et work_val (DP avec W forcé à t)
 *   - delta[e] = free_val[e] − work_val[e] ≥ 0
 *   - Trier par delta croissant, forcer les d[t,i] premiers à travailler.
 *   Optimalité : total(S) = Σ free_val − Σ_{e∈S} delta[e], minimiser Σ delta.
 */
#include "v1cld_base.hpp"
#include <tuple>
#include <vector>

class SSPCLDProblem : public V1CLDProblem {
public:
    /*
     * Mêmes paramètres que SSPShiftProblem.
     * transitions : (src, sym_int, dst), sym_int ∈ {0=O,1=P,2=R,3=W}
     * salary      : (T,) coûts ≥ 0
     * demand      : (N, T) demande par scénario et période
     * p           : (N,) probabilités des scénarios
     */
    SSPCLDProblem(
        const std::vector<std::tuple<int,int,int>>& transitions,
        const std::vector<int>&  accepting,
        int                      initial,
        const Vec1d&             salary,
        const Vec2i&             demand,
        Vec1d                    p,
        double                   epsilon,
        int                      E
    );

    int           n_vars()        const override { return E_ * T_ * SIGMA_SIZE; }
    int           N_scen()        const override { return N_; }
    int           n_constraints() const override { return T_; }
    double        epsilon()       const override { return epsilon_; }
    const Vec1d&  p()             const override { return p_; }
    const Vec1d&  c()             const override { return c_; }

    std::pair<double, Vec1d> solve_theta(int j, const Vec1d& cost) const override;
    std::pair<double, Vec1d> solve_zeta(int i, int j, const Vec1d& cost) const override;

protected:
    // Accès protégé : nécessaire à SSPCLDProblemBnB (ssp_cld_problem_bnb.hpp) qui
    // surcharge solve_theta/solve_zeta avec un masque de variables fixées (résiduel B&B).
    static constexpr int SIGMA_SIZE = 4;  // {O=0, P=1, R=2, W=3}
    static constexpr int SIGMA_W    = 3;

    int    E_, T_, N_, Q_;
    double epsilon_;
    Vec1d  c_, p_;

    int              initial_;
    std::vector<int> accepting_sorted_;
    std::vector<std::tuple<int,int,int>> transitions_;
    std::vector<std::vector<std::pair<int,int>>> adj_;  // adj_[q] = [(sym, q_next)]
    Vec2i  demand_;  // (N, T)

    // Court-circuit (2026-08-20) : quand une séquence complète de T symboles
    // est déjà entièrement déterminée (employé fixé partout par le
    // branchement B&B), il n'existe qu'UN SEUL chemin possible dans le DFA --
    // pas besoin de la DP sur les Q_ états (qui scanne O(T·Q_) même quand un
    // seul état reste jamais atteignable). On suit directement seq, en O(T ×
    // degré sortant) -- identité exacte avec ce que la DP renverrait dans ce
    // cas précis (même valeur, même séquence), pas une approximation.
    std::pair<double, std::vector<int>>
    walk_fixed_sequence(int e, const std::vector<int>& seq, const Vec1d& cost) const;

    // DP sur le DAG déroulé pour un employé e.
    // mask_row (taille T, optionnel) : mask_row[t] ∈ {-1, 0..3}, -1 = libre.
    // mask_row == nullptr → comportement inchangé (aucune contrainte sur sigma).
    std::pair<double, std::vector<int>>
    dp_one_employee(int e, const Vec1d& cost, const Vec1i* mask_row = nullptr) const;

    // DP pour un employé e avec sigma = W forcé à la période t_force, en plus
    // du masque éventuel sur les autres périodes (mask_row[t_force] doit être
    // -1 : t_force est par construction une période encore libre).
    // Retourne {NEG_INF, {}} si aucun chemin valide n'existe (infaisable pour cet employé).
    std::pair<double, std::vector<int>>
    dp_one_employee_forced_at(int e, int t_force, const Vec1d& cost,
                               const Vec1i* mask_row = nullptr) const;

    // ── Variantes masquées de solve_theta/solve_zeta ─────────────────────────
    // mask (E×T, optionnel) : mask[e][t] ∈ {-1, 0..3}. mask == nullptr →
    // comportement rigoureusement identique à solve_theta(j,cost)/solve_zeta(i,j,cost).
    // Utilisées par SSPCLDProblemBnB pour le sous-problème résiduel d'un nœud B&B.
    std::pair<double, Vec1d>
    solve_theta_masked(int j, const Vec1d& cost, const Vec2i* mask = nullptr) const;
    std::pair<double, Vec1d>
    solve_zeta_masked(int i, int j, const Vec1d& cost, const Vec2i* mask = nullptr) const;
};
