#pragma once
#include "types.hpp"
#include <utility>

struct EvalResult {
    double  L1_val;
    Vec2d   g;           // g[i][k] = x_i*[k] - x̄[k]
    Vec1d   x_bar;       // x̄ = Σ p_i x_i*
    Vec2d   x_scenarios; // x_scenarios[i] = x_i*
    Vec1d   z_star;      // z* du knapsack
};

// Évalue L1(λ) et son sous-gradient sans mise à jour des multiplicateurs.
EvalResult evaluate_L1(
    const Vec1d& c,
    const Vec1d& p,
    const Vec3d& A,
    const Vec2d& b,
    double       epsilon,
    const Vec2d& lam
);

SolverResult solve_impl(
    const Vec1d& c,
    const Vec1d& p,
    const Vec3d& A,
    const Vec2d& b,
    double       epsilon,
    int          T            = 500,
    bool         decomp       = true,
    bool         verbose      = false,
    double       time_limit_s = -1.0,
    int          log_interval = 1,
    double       rel_tol      = -1.0,
    int          tol_window   = 500,
    const Vec2d& lambda_init  = {}
);

// Calcul de v1LD par descente de coordonnées avec recherche ternaire exacte (multi-départ).
//
// Principe : pour chaque coordonnée λ_{i,k}, on maximise L1 en 1D par recherche
// ternaire (applicable car L1 est concave). Un seul cycle de coordonnées peut rester
// bloqué dans un plat non-lisse ; le multi-départ explore l'espace λ.
//
// L'algorithme lance n_restarts initialisations aléatoires + optionnellement lambda_init,
// applique T_outer cycles de descente de coordonnées à chaque départ, et retourne
// le meilleur résultat observé (= borne inférieure certifiée sur le vrai v1LD).
SolverResult solve_ahmed(
    const Vec1d& c,
    const Vec1d& p,
    const Vec3d& A,
    const Vec2d& b,
    double       epsilon,
    int          T_outer        = 20,    // cycles de coordonnées par départ
    int          T_search       = 60,    // itérations de recherche ternaire par coord.
    double       search_radius  = 50.0,  // plage de recherche autour du centre courant
    int          n_restarts     = 5,     // nb de départs aléatoires supplémentaires
    double       init_scale     = 1.0,   // écart-type de l'initialisation aléatoire
    bool         verbose        = false,
    double       time_limit_s   = -1.0,
    const Vec2d& lambda_init    = {},    // départ supplémentaire fourni (optionnel)
    unsigned int seed           = 42
);
