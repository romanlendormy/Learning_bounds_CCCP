#pragma once
/*
 * mkp_problem_bnb.hpp — Variante B&B de MKPProblem pour benchmarks comparatifs.
 *
 * Hérite de MKPProblem (θ identique), surcharge uniquement solve_zeta avec
 * le Branch-and-Bound exact original (Dantzig, relaxation LP par colonne).
 * Utilisé pour comparer convergence / qualité de borne vs CP-SAT.
 */

#include "mkp_problem.hpp"

class MKPProblemBnB : public MKPProblem {
public:
    using MKPProblem::MKPProblem;   // même constructeur

    std::pair<double, Vec1d> solve_zeta(int i, const Vec1d& cost) const override;
};
