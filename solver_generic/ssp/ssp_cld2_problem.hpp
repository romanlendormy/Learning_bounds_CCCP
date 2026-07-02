#pragma once
/*
 * SSPCLD2Problem — v1CLD2 pour le SSP shift scheduling.
 *
 * Même décomposition externe que SSPCLDProblem (scénario i, période j, m = T),
 * mêmes θ_{i,j} (E DP indépendantes, inchangées). La différence porte
 * uniquement sur la décomposition INTERNE de ζ_{i,j} : au lieu de relancer,
 * pour chaque employé, une DP complète sur les T périodes avec W forcé à j
 * (dp_one_employee_forced_at), on précalcule une fois par employé deux tables
 * de valeurs — avant (F) et arrière (V) — puis on dérive work_val_e(j) en
 * O(|transitions_W|) par un simple recollement F[j][q] + coût + V[j+1][q'].
 *
 * C'est une identité exacte de décomposition avant/arrière d'un plus long
 * chemin dans un DAG (aucune approximation, aucune itération) : les valeurs
 * obtenues sont rigoureusement identiques à celles de SSPCLDProblem. Ce qui
 * change, c'est que la décision « employé e, période j » devient un
 * sous-problème atomique (T=1, E=1) résolu par lookup sur F/V précalculés,
 * au lieu d'une DP dédiée par période forcée.
 */
#include "v1cld_base.hpp"
#include <tuple>
#include <vector>

class SSPCLD2Problem : public V1CLDProblem {
public:
    SSPCLD2Problem(
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

private:
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

    // DP sur le DAG déroulé pour un employé e, sans contrainte sur sigma —
    // identique à SSPCLDProblem::dp_one_employee (recopiée : pas de
    // dépendance fragile à un membre privé d'une autre classe).
    std::pair<double, std::vector<int>>
    dp_one_employee(int e, const Vec1d& cost) const;

    // Table avant F[t][q] (t=0..T) + backpointers (prédécesseur, symbole)
    // pour la période t=0..T-1.
    struct Forward {
        std::vector<Vec1d> F;         // (T+1) x Q
        std::vector<int>   bp_prev;   // T x Q
        std::vector<int>   bp_sym;    // T x Q
    };
    Forward dp_forward(int e, const Vec1d& cost) const;

    // Table arrière V[t][q] (t=0..T) + backpointers (symbole, successeur)
    // pour la période t=0..T-1.
    struct Backward {
        std::vector<Vec1d> V;         // (T+1) x Q
        std::vector<int>   bp_sym;    // T x Q
        std::vector<int>   bp_next;   // T x Q
    };
    Backward dp_backward(int e, const Vec1d& cost) const;

    // work_val_e(j) : sous-problème atomique T=1, E=1 — argmax sur les arcs
    // W à la période j, recollant F[j][·] et V[j+1][·]. Retourne
    // {NEG_INF, {}} si aucun arc W à j n'est atteignable/faisable.
    std::pair<double, std::vector<int>>
    work_val_at(int e, int j, const Vec1d& cost,
                const Forward& fwd, const Backward& bwd) const;
};
