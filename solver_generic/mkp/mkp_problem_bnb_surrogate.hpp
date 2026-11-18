#pragma once
/*
 * mkp_problem_bnb_surrogate.hpp — Variante B&B de MKPProblem (option "A" de
 * l'exploration accélération ζ) : même squelette de B&B récursif que
 * MKPProblemBnB (mkp_problem_bnb.cpp), mais la borne de branchement est une
 * relaxation SURROGATE (agrégation pondérée des m contraintes en une seule)
 * au lieu du minimum de m relaxations séparées — cf. mkp_surrogate_bound.hpp
 * pour la justification mathématique complète.
 *
 * θ identique (hérité de MKPProblem, inchangé). ζ_i retourne EXACTEMENT la
 * même valeur optimale que MKPProblemBnB — seule la vitesse de la preuve
 * d'optimalité change (moins de nœuds explorés grâce à une borne plus serrée).
 */

#include "mkp_problem.hpp"

class MKPProblemBnBSurrogate : public MKPProblem {
public:
    using MKPProblem::MKPProblem;   // même constructeur

    std::pair<double, Vec1d> solve_zeta(int i, const Vec1d& cost) const override;
};
