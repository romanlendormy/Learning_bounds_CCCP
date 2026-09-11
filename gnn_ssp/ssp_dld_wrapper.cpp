/*
 * ssp_dld_wrapper.cpp — API extern "C" autour de SSPDLDProblemBnB
 * (solver_generic/ssp/ssp_dld_problem_bnb.hpp), miroir de ssp_sld_wrapper.cpp.
 *
 * Différence avec SLD : solve_theta/solve_zeta prennent un indice de période
 * j en plus (la contrainte de couverture dualisée), et ζ est un glouton
 * O(E·T·Q·4) au lieu d'un CP-SAT — beaucoup moins cher par appel, mais il y a
 * N_z*T appels par évaluation de la borne au lieu de N_z (cf. dld_solver.cpp).
 */
#include "ssp_dld_problem_bnb.hpp"
#include <tuple>
#include <vector>

extern "C" {

struct SSPDLDHandle {
    SSPDLDProblemBnB* prob;
    int n_vars;
};

void* ssp_dld_create(
    int n_trans, const int* trans_flat,
    int n_acc,   const int* accepting,
    int initial,
    int T, const double* salary,
    int N, const int* demand_flat,
    const double* p,
    double epsilon,
    int E,
    const int* mask_flat)
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

    auto* h = new SSPDLDHandle();
    h->prob = new SSPDLDProblemBnB(transitions, acc, initial,
                                    salary_v, demand, p_v, epsilon, E, mask);
    h->n_vars = E * T * 4;
    return static_cast<void*>(h);
}

void ssp_dld_destroy(void* handle) {
    auto* h = static_cast<SSPDLDHandle*>(handle);
    delete h->prob;
    delete h;
}

int ssp_dld_n_vars(void* handle) {
    return static_cast<SSPDLDHandle*>(handle)->n_vars;
}

double ssp_dld_solve_theta(void* handle, int j, const double* cost, double* x_out) {
    auto* h = static_cast<SSPDLDHandle*>(handle);
    Vec1d c(cost, cost + h->n_vars);
    auto result = h->prob->solve_theta(j, c);
    const Vec1d& x = result.second;
    for (int k = 0; k < h->n_vars; ++k) x_out[k] = x[k];
    return result.first;
}

double ssp_dld_solve_zeta(void* handle, int i, int j, const double* cost, double* x_out) {
    auto* h = static_cast<SSPDLDHandle*>(handle);
    Vec1d c(cost, cost + h->n_vars);
    auto result = h->prob->solve_zeta(i, j, c);
    const Vec1d& x = result.second;
    for (int k = 0; k < h->n_vars; ++k) x_out[k] = x[k];
    return result.first;
}

} // extern "C"
