#pragma once
/*
 * mkp_cld_problem_dp.hpp — Variante DP de MKPCLDProblem.
 *
 * solve_zeta utilise un DP O(n × W) avec buffer pré-alloué :
 * dp_buf_ est alloué une fois dans le constructeur (taille maximale sur
 * tous les (i,j)), puis réutilisé à chaque appel.  Évite les (ni+1)
 * allocations heap par appel du vector<vector<double>> naïf.
 */

#include "mkp_cld_problem.hpp"

class MKPCLDProblemDP : public MKPCLDProblem {
public:
    MKPCLDProblemDP(Vec1d c, Vec1d p, Vec3d A, Vec2d b, double epsilon);

    std::pair<double, Vec1d> solve_zeta(int i, int j, const Vec1d& cost) const override;

private:
    // Buffer plat (n+1) × (W_max+1), partagé entre tous les appels.
    // mutable car solve_zeta est const mais écrit dans le buffer.
    mutable std::vector<double> dp_buf_;
    int W_max_;   // max sur tous les b_[i][j] (arrondi)
};
