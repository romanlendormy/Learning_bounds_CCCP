/*
 * gecode_bindings.cpp — Bindings pybind11 pour gecode_cpp.
 *
 * Expose :
 *   solve_mkp(c, p, A, b, epsilon, time_limit=300.0)  → dict
 *   solve_ssp(profit, p, epsilon, delta, accepting, time_limit=300.0) → dict
 *
 * Utilisation Python :
 *   import gecode_cpp as gc
 *   res = gc.solve_mkp(c, p, A, b, epsilon=0.1, time_limit=120.0)
 *   print(res["objective"], res["status"])
 *
 *   res = gc.solve_ssp(profit, p, epsilon, delta, accepting, time_limit=300.0)
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

// delta : (N, k, Q, m) int32
static std::vector<std::vector<std::vector<std::vector<int>>>>
to_delta(py::array_t<int32_t> arr) {
    auto buf = arr.unchecked<4>();
    int N = buf.shape(0), k = buf.shape(1), Q = buf.shape(2), m = buf.shape(3);
    std::vector<std::vector<std::vector<std::vector<int>>>> d(
        N, std::vector<std::vector<std::vector<int>>>(
            k, std::vector<std::vector<int>>(
                Q, std::vector<int>(m))));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < k; ++j)
            for (int q = 0; q < Q; ++q)
                for (int a = 0; a < m; ++a)
                    d[i][j][q][a] = buf(i, j, q, a);
    return d;
}

// accepting : (N, k, Q) uint8
static std::vector<std::vector<std::vector<int>>>
to_accepting(py::array_t<uint8_t> arr) {
    auto buf = arr.unchecked<3>();
    int N = buf.shape(0), k = buf.shape(1), Q = buf.shape(2);
    std::vector<std::vector<std::vector<int>>> ac(
        N, std::vector<std::vector<int>>(k, std::vector<int>(Q)));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < k; ++j)
            for (int q = 0; q < Q; ++q)
                ac[i][j][q] = (buf(i, j, q) != 0) ? 1 : 0;
    return ac;
}

// ── Convertir GecodeResult → py::dict ─────────────────────────────────────────

static py::dict to_dict(const GecodeResult& r) {
    py::dict d;
    d["objective"]          = r.objective;
    d["optimal"]            = r.optimal;
    d["feasible"]           = r.feasible;
    d["solve_time_s"]       = r.solve_time_s;
    d["cold_start_time_s"]  = r.cold_start_time_s;
    d["status"]             = r.status;
    d["nodes"]              = r.nodes;
    d["fails"]              = r.fails;
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
        [](py::array_t<double>  profit_arr,
           py::array_t<double>  p_arr,
           double               epsilon,
           py::array_t<int32_t> delta_arr,
           py::array_t<uint8_t> acc_arr,
           double               time_limit) -> py::dict
        {
            auto prof_buf  = profit_arr.unchecked<2>();
            auto p_buf     = p_arr.unchecked<1>();
            auto delta_buf = delta_arr.unchecked<4>();

            int n = prof_buf.shape(0), mv = prof_buf.shape(1);
            int N = p_buf.shape(0);
            int k = delta_buf.shape(1), Q = delta_buf.shape(2), m2 = delta_buf.shape(3);
            if (mv != m2)
                throw std::invalid_argument("solve_ssp: profit.shape[1] != delta.shape[3]");

            // profit aplati (n*m,)
            Vec1d profit_flat(n * mv);
            for (int t = 0; t < n; ++t)
                for (int a = 0; a < mv; ++a)
                    profit_flat[t * mv + a] = prof_buf(t, a);

            return to_dict(solve_gecode_ssp(
                n, mv, k, Q, N, epsilon,
                profit_flat, to_vec1d(p_arr),
                to_delta(delta_arr), to_accepting(acc_arr),
                time_limit));
        },
        py::arg("profit"), py::arg("p"), py::arg("epsilon"),
        py::arg("delta"), py::arg("accepting"),
        py::arg("time_limit") = 300.0,
        R"doc(
Résout le CCSSP chance-contraint en one-shot via Gecode (B&B, automate produit).

Args:
    profit    : (n, m)       matrice de profits (float64)
    p         : (N,)         probabilités des scénarios (float64)
    epsilon   : niveau de risque
    delta     : (N, k, Q, m) int32 — tables de transition (-1 = mort)
    accepting : (N, k, Q)    uint8 — états acceptants
    time_limit: limite de temps en secondes (défaut 300s)

Retourne un dict :
    objective, feasible, optimal, solve_time_s, status
)doc");
}
