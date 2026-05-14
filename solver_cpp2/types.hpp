#pragma once
#include <vector>

using Vec1d = std::vector<double>;
using Vec2d = std::vector<Vec1d>;
using Vec3d = std::vector<Vec2d>;

struct SolverResult {
    Vec2d  lambda_star;
    Vec3d  mu_star;
    double v1LD_star;
    Vec1d  convergence;      // L1[t] à chaque point enregistré
    Vec1d  times_s;          // temps écoulé (s) à chaque point enregistré
    Vec1d  gnorm_history;    // ||g_λ||₂ à chaque point enregistré
    Vec1d  z_changes;        // nb composantes z* changées depuis le point précédent
    double solve_time_s;
    int    iterations_run;

    // Snapshots aux 5 dernières itérations multiples de 5 (de la plus ancienne à la plus récente)
    Vec3d  g_history;            // [snapshot][i][k] = x_i*[k] - x̄_outer[k]
    Vec2d  xbar_outer_history;   // [snapshot][k]    = Σ p_i x_i*[k]
    Vec2d  zstar_history;        // [snapshot][i]    = z*[i]
    Vec1d  iter_history;         // numéros d'itération correspondants
};
