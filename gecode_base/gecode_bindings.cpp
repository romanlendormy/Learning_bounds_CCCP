/*
 * gecode_bindings.cpp — Bindings pybind11 pour gecode_cpp.
 *
 * Expose :
 *   solve_mkp(c, p, A, b, epsilon, time_limit=300.0)  → dict
 *   solve_ssp(E, T, m, Q, N, epsilon,
 *             work_indicator, salary, delta, q0, accepting, demand,
 *             time_limit=300.0, break_symmetry=True)   → dict
 *   run_ssp_tests()                                    → bool
 *
 * Utilisation Python :
 *   import gecode_cpp as gc
 *   res = gc.solve_mkp(c, p, A, b, epsilon=0.1, time_limit=120.0)
 *   print(res["objective"], res["status"])
 *
 *   res = gc.solve_ssp(E, T, m, Q, N, 0.1,
 *                      work_indicator, salary, delta, q0, accepting, demand)
 */

#include "gecode_solver.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;

// ── Helpers numpy → Vec ───────────────────────────────────────────────────────

static Vec1d to_vec1d(py::array_t<double> arr) {
    auto buf = arr.unchecked<1>();
    Vec1d v(buf.shape(0));
    for (int i = 0; i < (int)v.size(); ++i) v[i] = buf(i);
    return v;
}

static Vec2d to_vec2d(py::array_t<double> arr) {
    auto buf = arr.unchecked<2>();
    int r = buf.shape(0), c = buf.shape(1);
    Vec2d v(r, Vec1d(c));
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            v[i][j] = buf(i, j);
    return v;
}

static Vec3d to_vec3d(py::array_t<double> arr) {
    auto buf = arr.unchecked<3>();
    int a = buf.shape(0), b = buf.shape(1), c = buf.shape(2);
    Vec3d v(a, Vec2d(b, Vec1d(c)));
    for (int i = 0; i < a; ++i)
        for (int j = 0; j < b; ++j)
            for (int k = 0; k < c; ++k)
                v[i][j][k] = buf(i, j, k);
    return v;
}

// delta SSP : (Q, m) int32 — table de transition
static std::vector<std::vector<int>> to_delta2d(py::array_t<int32_t> arr) {
    auto buf = arr.unchecked<2>();
    int Q = buf.shape(0), m = buf.shape(1);
    std::vector<std::vector<int>> d(Q, std::vector<int>(m));
    for (int q = 0; q < Q; ++q)
        for (int a = 0; a < m; ++a)
            d[q][a] = buf(q, a);
    return d;
}

// demand SSP : (N, T) int32
static std::vector<std::vector<int>> to_demand(py::array_t<int32_t> arr) {
    auto buf = arr.unchecked<2>();
    int N = buf.shape(0), T = buf.shape(1);
    std::vector<std::vector<int>> d(N, std::vector<int>(T));
    for (int i = 0; i < N; ++i)
        for (int t = 0; t < T; ++t)
            d[i][t] = buf(i, t);
    return d;
}

// ── Convertir GecodeResult → py::dict ─────────────────────────────────────────

static py::dict to_dict(const GecodeResult& r) {
    py::dict d;
    d["objective"]               = r.objective;
    d["optimal"]                 = r.optimal;
    d["feasible"]                = r.feasible;
    d["solve_time_s"]            = r.solve_time_s;
    d["cold_start_time_s"]       = r.cold_start_time_s;
    d["status"]                  = r.status;
    d["nodes"]                   = r.nodes;
    d["fails"]                   = r.fails;
    d["depth"]                   = r.depth;
    d["incumbent_improvements"]  = r.incumbent_improvements;
    return d;
}

// ── Module ────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(gecode_cpp, m) {
    m.doc() = "Solveur CP exact Gecode 6 — CCMKP et CCSSP one-shot";

    m.def("solve_mkp",
        [](py::array_t<double> c_arr,
           py::array_t<double> p_arr,
           py::array_t<double> A_arr,
           py::array_t<double> b_arr,
           double epsilon,
           double time_limit) -> py::dict
        {
            auto c_buf = c_arr.unchecked<1>();
            auto p_buf = p_arr.unchecked<1>();
            auto A_buf = A_arr.unchecked<3>();
            auto b_buf = b_arr.unchecked<2>();
            int n = c_buf.shape(0);
            int N = p_buf.shape(0);
            int M = A_buf.shape(1);
            if (A_buf.shape(0) != N || A_buf.shape(2) != n)
                throw std::invalid_argument("solve_mkp: A shape mismatch");
            if (b_buf.shape(0) != N || b_buf.shape(1) != M)
                throw std::invalid_argument("solve_mkp: b shape mismatch");
            return to_dict(solve_gecode_mkp(
                n, M, N, epsilon,
                to_vec1d(c_arr), to_vec1d(p_arr),
                to_vec3d(A_arr), to_vec2d(b_arr),
                time_limit));
        },
        py::arg("c"), py::arg("p"), py::arg("A"), py::arg("b"),
        py::arg("epsilon"),
        py::arg("time_limit") = 300.0,
        R"doc(
Résout le CCMKP chance-contraint en one-shot via Gecode (B&B, formulation Big-M).

Args:
    c          : (n,)      coûts déterministes (float64)
    p          : (N,)      probabilités des scénarios (float64)
    A          : (N, m, n) matrices de contraintes (float64)
    b          : (N, m)    membres droits (float64)
    epsilon    : niveau de risque
    time_limit : limite de temps en secondes (défaut 300s)

Retourne un dict :
    objective, feasible, optimal, solve_time_s, status
)doc");

    m.def("solve_ssp",
        [](int                  E,
           int                  T,
           int                  m,
           int                  Q,
           int                  N,
           double               epsilon,
           py::array_t<int32_t> work_indicator_arr,
           py::array_t<double>  salary_arr,
           py::array_t<int32_t> delta_arr,
           int                  q0,
           std::vector<int>     accepting,
           py::array_t<int32_t> demand_arr,
           double               time_limit,
           bool                 break_symmetry) -> py::dict
        {
            auto wi_buf = work_indicator_arr.unchecked<1>();
            if (wi_buf.shape(0) != m)
                throw std::invalid_argument("solve_ssp: work_indicator size != m");
            std::vector<int> work_indicator(m);
            for (int a = 0; a < m; ++a) work_indicator[a] = wi_buf(a);

            auto sal_buf = salary_arr.unchecked<1>();
            if (sal_buf.shape(0) != T)
                throw std::invalid_argument("solve_ssp: salary size != T");
            Vec1d salary(T);
            for (int t = 0; t < T; ++t) salary[t] = sal_buf(t);

            auto delta_buf = delta_arr.unchecked<2>();
            if (delta_buf.shape(0) != Q || delta_buf.shape(1) != m)
                throw std::invalid_argument("solve_ssp: delta shape != (Q, m)");

            auto dem_buf = demand_arr.unchecked<2>();
            if (dem_buf.shape(0) != N || dem_buf.shape(1) != T)
                throw std::invalid_argument("solve_ssp: demand shape != (N, T)");

            return to_dict(solve_gecode_ssp(
                E, T, m, Q, N, epsilon,
                work_indicator, salary,
                to_delta2d(delta_arr), q0, accepting,
                to_demand(demand_arr),
                time_limit, break_symmetry));
        },
        py::arg("E"), py::arg("T"), py::arg("m"), py::arg("Q"), py::arg("N"),
        py::arg("epsilon"),
        py::arg("work_indicator"), py::arg("salary"),
        py::arg("delta"), py::arg("q0"), py::arg("accepting"), py::arg("demand"),
        py::arg("time_limit")      = 300.0,
        py::arg("break_symmetry")  = true,
        R"doc(
Résout le CCSSP couverture multi-employés en one-shot via Gecode (B&B, BAB).

Args:
    E              : nombre d'employés
    T              : nombre de périodes
    m              : taille de l'alphabet
    Q              : nombre d'états du DFA
    N              : nombre de scénarios
    epsilon        : niveau de risque (chance constraint)
    work_indicator : (m,) int32 — 1 si le symbole est "travail", 0 sinon
    salary         : (T,) float64 — coût salarial par période
    delta          : (Q, m) int32 — table de transition (-1 = morte)
    q0             : état initial du DFA
    accepting      : list[int] — états acceptants
    demand         : (N, T) int32 — demande[scénario][période]
    time_limit     : limite de temps en secondes (défaut 300s)
    break_symmetry : brisure lex sur les lignes employés (défaut True)

Retourne un dict :
    objective, feasible, optimal, solve_time_s, cold_start_time_s,
    status, nodes, fails, depth, incumbent_improvements
)doc");

    m.def("run_ssp_tests", &run_ssp_tests,
        "Exécute les tests de régression SSP (E=3,T=8,N=20,ε=0.1). Retourne True si tous OK.");
}
