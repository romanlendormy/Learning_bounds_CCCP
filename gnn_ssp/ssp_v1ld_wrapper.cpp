/*
 * ssp_v1ld_wrapper.cpp — API extern "C" autour de SSPProblemBnB (variante
 * MASQUÉE de SSPShiftProblem, solver_generic/ssp/ssp_problem_bnb.hpp), pour
 * appel depuis Python via ctypes.
 *
 * IMPORTANT : on utilise SSPProblemBnB, pas SSPShiftProblem directement —
 * un nœud B&B a des variables x[e,t] déjà fixées par le branchement, et
 * SSPShiftProblem::solve_theta/solve_zeta (non masqués) les ignoreraient
 * complètement (domaine = TOUTES les variables libres), donnant une borne
 * invalide pour ce nœud. SSPProblemBnB restreint le domaine via mask (E×T,
 * -1=libre sinon symbole imposé), cf. ssp_problem_bnb.hpp.
 *
 * Miroir de gnn/bnb_solver.py (MKP) : même esprit (wrapper C plat, compilé à
 * la volée en .so), mais ici solve_zeta dépend d'OR-Tools CP-SAT, donc le
 * lien est plus lourd que le B&B maison du MKP — cf. bnb_solver_ssp.py pour
 * la commande de compilation complète.
 */
#include "ssp_problem_bnb.hpp"
#include <tuple>
#include <vector>

extern "C" {

struct SSPHandle {
    SSPProblemBnB* prob;
    int n_vars;
};

void* ssp_v1ld_create(
    int n_trans, const int* trans_flat,   // (src,sym,dst) x n_trans, aplati
    int n_acc,   const int* accepting,
    int initial,
    int T, const double* salary,
    int N, const int* demand_flat,        // demand[i][t], aplati row-major
    const double* p,
    double epsilon,
    int E,
    const int* mask_flat)                 // (E*T,) -1=libre sinon symbole 0..3, aplati e*T+t
{
    std::vector<std::tuple<int,int,int>> transitions(n_trans);
    for (int k = 0; k < n_trans; ++k)
        transitions[k] = std::make_tuple(
            trans_flat[3*k], trans_flat[3*k+1], trans_flat[3*k+2]);

    std::vector<int> acc(accepting, accepting + n_acc);
    Vec1d salary_v(salary, salary + T);

    Vec2i demand(N, Vec1i(T));
    for (int i = 0; i < N; ++i)
        for (int t = 0; t < T; ++t)
            demand[i][t] = demand_flat[i * T + t];

    Vec1d p_v(p, p + N);

    Vec2i mask(E, Vec1i(T));
    for (int e = 0; e < E; ++e)
        for (int t = 0; t < T; ++t)
            mask[e][t] = mask_flat[e * T + t];

    auto* h = new SSPHandle();
    h->prob = new SSPProblemBnB(transitions, acc, initial,
                                 salary_v, demand, p_v, epsilon, E, mask);
    h->n_vars = E * T * 4;
    return static_cast<void*>(h);
}

void ssp_v1ld_destroy(void* handle) {
    auto* h = static_cast<SSPHandle*>(handle);
    delete h->prob;
    delete h;
}

int ssp_v1ld_n_vars(void* handle) {
    return static_cast<SSPHandle*>(handle)->n_vars;
}

// cost : (n_vars,) — x_out : (n_vars,) rempli en sortie. Retourne la valeur optimale.
double ssp_v1ld_solve_theta(void* handle, const double* cost, double* x_out) {
    auto* h = static_cast<SSPHandle*>(handle);
    Vec1d c(cost, cost + h->n_vars);
    auto result = h->prob->solve_theta(c);
    const Vec1d& x = result.second;
    for (int k = 0; k < h->n_vars; ++k) x_out[k] = x[k];
    return result.first;
}

double ssp_v1ld_solve_zeta(void* handle, int i, const double* cost, double* x_out) {
    auto* h = static_cast<SSPHandle*>(handle);
    Vec1d c(cost, cost + h->n_vars);
    auto result = h->prob->solve_zeta(i, c);
    const Vec1d& x = result.second;
    for (int k = 0; k < h->n_vars; ++k) x_out[k] = x[k];
    return result.first;
}

} // extern "C"
