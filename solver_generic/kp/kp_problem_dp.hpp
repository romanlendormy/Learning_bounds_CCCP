#pragma once
/*
 * kp_problem_dp.hpp — Variante DP de KPProblem.
 *
 * solve_zeta utilise un DP O(n × B) avec buffer pré-alloué (dp_buf_, taille
 * maximale sur tous les scénarios, alloué une fois dans le constructeur) au
 * lieu du B&B — cf. kp_problem.hpp pour la discussion du choix par défaut.
 *
 * Poids/capacité arrondis à l'entier le plus proche : exact seulement si les
 * données sont (proches d')entières. Sur des instances Pisinger+Song avec
 * perturbation continue, ζ devient une approximation — à valider par
 * benchmark avant de préférer cette variante au B&B par défaut.
 */

#include "kp_problem.hpp"

class KPProblemDP : public KPProblem {
public:
    KPProblemDP(Vec1d c, Vec1d p, Vec2d A, Vec1d b, double epsilon);

    std::pair<double, Vec1d> solve_zeta(int i, const Vec1d& cost) const override;

private:
    // Buffer plat (n+1) × (B_max+1), partagé entre tous les appels.
    mutable std::vector<double> dp_buf_;
    int B_max_;   // max sur tous les b_[i] (arrondi)
};
