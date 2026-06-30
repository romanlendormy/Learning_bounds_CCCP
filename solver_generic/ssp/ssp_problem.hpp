#pragma once
/*
 * ssp_problem.hpp — V1LDProblem pour le SSP shift scheduling multi-employés.
 *
 * Problème : E employés partagent un DFA unique de quarts légaux.
 * Scénarios : demande d[t,i] (entier ≥ 0) requise à la période t pour le scénario i.
 * Objectif  : min Σ_e Σ_t salary[t] · 1[x[e,t]=W]  →  max Σ_e Σ_t (-salary[t]) · x[e,t,W]
 *
 * Représentation des variables (one-hot sur Σ={O,P,R,W}) :
 *   n_vars = E × T × 4
 *   idx(e,t,σ) = e*T*4 + t*4 + σ   (σ : O=0, P=1, R=2, W=3)
 *   x[e,t,σ] ∈ {0,1} : indicateur "employé e fait σ à la période t"
 *
 * Coût de base : c[e,t,W] = −salary[t], c[e,t,autre] = 0.
 *
 * Graphe en couches (DAG) déroulé construit UNE FOIS à l'init :
 *   T+1 couches, Q états par couche, arcs (t,q)→(t+1,q') étiquetés σ.
 *   adj_[q] = [(σ, q')] — liste d'adjacence triée, partagée par tous les appels.
 *
 * solve_theta(cost) :
 *   Domaine de base = { x : regular(x[e,·], DFA) ∀e } (aucune couverture).
 *   Décompose en E DPs indépendants sur le DAG. O(E·T·Q·4).
 *
 * solve_zeta(i, cost) :
 *   Domaine du scénario i = base ∩ { Σ_e x[e,t,W] ≥ d[t,i] ∀t }.
 *   Exact via OR-Tools CP-SAT (AddAutomaton × E + couverture). NP-hard exact.
 *   Validité de la borne v1LD exige l'optimalité exacte.
 */

#include "v1ld_base.hpp"
#include <tuple>
#include <vector>

// Indices alphabétiques (O=0, P=1, R=2, W=3 — ordre lexicographique trié)
static constexpr int SSP_SIGMA_SIZE = 4;
static constexpr int SSP_SIGMA_W    = 3;  // indice de W (work)

class SSPShiftProblem : public V1LDProblem {
public:
    /*
     * Constructeur.
     *
     * transitions : liste de tuples (src, sym_int, dst)
     *               sym_int ∈ {0=O, 1=P, 2=R, 3=W}
     *               transitions absentes = blocage (DFA partiel)
     * accepting   : états acceptants du DFA (non triés à l'entrée)
     * initial     : état initial du DFA
     * salary      : (T,) coûts salariaux ≥ 0
     * demand      : (N, T) demande par scénario et période  demand[i][t]
     * p           : (N,) probabilités des scénarios (Σ=1)
     * epsilon     : niveau de risque
     * E           : nombre d'employés
     */
    SSPShiftProblem(
        const std::vector<std::tuple<int,int,int>>& transitions,
        const std::vector<int>&  accepting,
        int                      initial,
        const Vec1d&             salary,
        const Vec2i&             demand,   // (N, T)
        Vec1d                    p,
        double                   epsilon,
        int E
    );

    // ── Interface V1LDProblem ─────────────────────────────────────────────────
    int           n_vars()  const override { return E_ * T_ * SSP_SIGMA_SIZE; }
    int           N_scen()  const override { return N_; }
    double        epsilon() const override { return epsilon_; }
    const Vec1d&  p()       const override { return p_; }
    const Vec1d&  c()       const override { return c_; }  // c[e,t,W]=-salary[t]

    std::pair<double, Vec1d> solve_theta(const Vec1d& cost) const override;
    std::pair<double, Vec1d> solve_zeta(int i, const Vec1d& cost) const override;

private:
    int    E_, T_, N_, Q_;
    double epsilon_;
    Vec1d  c_, p_;

    // DFA (invariant — construit une seule fois dans le constructeur)
    int              initial_;
    std::vector<int> accepting_sorted_;           // trié pour le déterminisme

    // Transitions (src, sym, dst), triées pour le déterminisme
    std::vector<std::tuple<int,int,int>> transitions_;

    // Adjacence par état source : adj_[q] = [(sym, q_next)] trié
    std::vector<std::vector<std::pair<int,int>>> adj_;

    // Demande : demand_[i][t]
    Vec2i demand_;   // (N, T)

    // ── DP interne pour un employé e ─────────────────────────────────────────
    // Retourne (valeur optimale, séquence d'actions [T])
    std::pair<double, std::vector<int>>
    dp_one_employee(int e, const Vec1d& cost) const;
};
