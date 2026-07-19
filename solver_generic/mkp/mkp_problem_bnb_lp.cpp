/*
 * mkp_problem_bnb_lp.cpp — solve_zeta par Branch-and-Bound exact, borne =
 * relaxation LP jointe (GLOP), warm-startée d'un nœud à l'autre.
 *
 * cf. mkp_problem_bnb_lp.hpp pour le contexte. Un seul objet LinearProgram +
 * LPSolver est construit à la racine (variables actives, bornes [0,1]) et
 * réutilisé pour TOUS les nœuds de CET appel solve_zeta : à chaque décision,
 * on modifie juste les bornes de LA variable concernée ([1,1] ou [0,0]) et on
 * rappelle Solve() sur le MÊME objet — pas de reconstruction de modèle
 * (contrairement à CP-SAT/Gecode, cf. discussion de session).
 */

#include "mkp_problem_bnb_lp.hpp"
#include "mkp_surrogate_bound.hpp"

#include "ortools/glop/lp_solver.h"
#include "ortools/lp_data/lp_data.h"
#include "ortools/lp_data/lp_types.h"

#include <vector>

using operations_research::glop::ColIndex;
using operations_research::glop::kInfinity;
using operations_research::glop::LinearProgram;
using operations_research::glop::LPSolver;
using operations_research::glop::ProblemStatus;
using operations_research::glop::RowIndex;

namespace {

struct MkpBnBLP {
    int na = 0, m = 0;
    double best_val = 0.0;
    std::vector<int8_t> x_bits, best_x_bits;
    LinearProgram lp;
    LPSolver solver;
    std::vector<ColIndex> cols;   // cols[p] = colonne GLOP de la position p (ordre de branchement)

    // Appelé une fois par nœud : résout le LP courant (bornes déjà à jour pour
    // les positions < idx), élague si la borne ne peut battre best_val, sinon
    // branche sur cols[idx] (essaie val=1 en premier).
    void run(int idx) {
        ProblemStatus st = solver.Solve(lp);
        if (st != ProblemStatus::OPTIMAL) return;   // infaisable (ou statut anormal) -> élagué
        double ub = solver.GetObjectiveValue();
        if (ub <= best_val + 1e-9) return;          // ne peut pas améliorer

        if (idx == na) {
            // Toutes les variables sont fixées : ub est la valeur EXACTE de
            // cette feuille (pas juste une borne), et elle bat déjà best_val.
            best_val = ub;
            best_x_bits = x_bits;
            return;
        }

        ColIndex col = cols[idx];

        lp.SetVariableBounds(col, 1.0, 1.0);
        x_bits[idx] = 1;
        run(idx + 1);
        x_bits[idx] = 0;

        lp.SetVariableBounds(col, 0.0, 0.0);
        run(idx + 1);

        lp.SetVariableBounds(col, 0.0, 1.0);   // restaure pour le retour au parent
    }
};

}  // namespace

std::pair<double, Vec1d> MKPProblemBnBLP::solve_zeta(int i, const Vec1d& cost) const
{
    Vec1d  x_out(n_, 0.0);
    double val_free = 0.0;
    const Vec2d& Ai = A_[i];
    const Vec1d& bi = b_[i];

    std::vector<int> active;
    active.reserve(n_);
    for (int k = 0; k < n_; ++k) {
        if (cost[k] <= 0.0) continue;
        bool free_item = true;
        for (int j = 0; j < m_; ++j)
            if (Ai[j][k] >= 1e-9) { free_item = false; break; }
        if (free_item) { x_out[k] = 1.0; val_free += cost[k]; }
        else           { active.push_back(k); }
    }
    if (active.empty()) return {val_free, std::move(x_out)};

    // Ordre de branchement : réutilise le tri surrogate (ratio décroissant)
    // pour comparer à ordre égal avec MKPProblemBnBSurrogate — seule la
    // borne (LP jointe vs surrogate) change.
    MkpSurrogatePrep prep = prepare_mkp_surrogate(active, cost, Ai, bi, m_);
    int na = static_cast<int>(prep.active_sorted.size());

    MkpBnBLP bb;
    bb.na = na; bb.m = m_;
    bb.x_bits.assign(na, 0); bb.best_x_bits.assign(na, 0);

    bb.cols.resize(na);
    for (int p = 0; p < na; ++p) bb.cols[p] = bb.lp.CreateNewVariable();
    bb.lp.SetMaximizationProblem(true);
    for (int p = 0; p < na; ++p) {
        int k = prep.active_sorted[p];
        bb.lp.SetVariableBounds(bb.cols[p], 0.0, 1.0);
        bb.lp.SetObjectiveCoefficient(bb.cols[p], cost[k]);
    }
    for (int j = 0; j < m_; ++j) {
        RowIndex row = bb.lp.CreateNewConstraint();
        bb.lp.SetConstraintBounds(row, -kInfinity, bi[j]);
        for (int p = 0; p < na; ++p) {
            int k = prep.active_sorted[p];
            bb.lp.SetCoefficient(row, bb.cols[p], Ai[j][k]);
        }
    }

    bb.run(0);

    for (int p = 0; p < na; ++p)
        if (bb.best_x_bits[p]) x_out[prep.active_sorted[p]] = 1.0;

    return {val_free + bb.best_val, std::move(x_out)};
}
