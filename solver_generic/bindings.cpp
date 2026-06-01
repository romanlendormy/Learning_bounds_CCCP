/*
 * bindings.cpp — Bindings pybind11 pour v1ld_generic_cpp.
 *
 * Expose :
 *   MKPProblem(c, p, A, b, epsilon)
 *   SSPProblem(profit, p, epsilon, delta, accepting)
 *     delta     : (N, k, Q, m) int32
 *     accepting : (N, k, Q)    uint8  (0/1)
 *   solve_v1ld(problem, T, log_interval, verbose, time_limit_s,
 *              rel_tol, tol_window, v_lower, lambda_init)
 *
 * Utilisation Python :
 *   import v1ld_generic_cpp as cpp
 *   prob = cpp.MKPProblem(c, p, A, b, epsilon)
 *   res  = cpp.solve_v1ld(prob, T=500)
 */

#include "v1ld_base.hpp"
#include "v1ld_solver.hpp"
#include "v1cld_base.hpp"
#include "v1cld_solver.hpp"
#include "mkp_problem.hpp"
#include "mkp_problem_bnb.hpp"
#include "mkp_problem_lp.hpp"
#include "mkp_cld_problem.hpp"
#include "ssp_problem.hpp"
#include "ssp_cld_problem.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// ── Helpers de conversion numpy → C++ ────────────────────────────────────────

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

// ── Module ────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(v1ld_generic_cpp, m) {
    m.doc() = "Solveur v1LD générique — MKP et SSP, C++17";

    // ── V1LDProblem (base abstraite — non instanciable depuis Python) ─────────
    py::class_<V1LDProblem>(m, "V1LDProblem")
        .def("n_vars",  &V1LDProblem::n_vars)
        .def("N_scen",  &V1LDProblem::N_scen)
        .def("epsilon", &V1LDProblem::epsilon);

    // ── MKPProblem ────────────────────────────────────────────────────────────
    py::class_<MKPProblem, V1LDProblem>(m, "MKPProblem",
        R"doc(
Problème v1LD pour le MKP chance-contraint.

Args:
    c       : (n,)      coûts déterministes
    p       : (N,)      probabilités des scénarios
    A       : (N, m, n) matrices de contraintes
    b       : (N, m)    membres droits
    epsilon : niveau de risque ε
)doc")
        .def(py::init([](
            py::array_t<double> c_arr,
            py::array_t<double> p_arr,
            py::array_t<double> A_arr,
            py::array_t<double> b_arr,
            double epsilon)
        {
            return new MKPProblem(
                to_vec1d(c_arr), to_vec1d(p_arr),
                to_vec3d(A_arr), to_vec2d(b_arr),
                epsilon
            );
        }),
        py::arg("c"), py::arg("p"), py::arg("A"), py::arg("b"),
        py::arg("epsilon"));

    // ── SSPProblem ────────────────────────────────────────────────────────────
    py::class_<SSPProblem, V1LDProblem>(m, "SSPProblem",
        R"doc(
Problème v1LD pour le CCSSP à contraintes automates.

Args:
    profit    : (n, m)      matrice de profits déterministe (row=période, col=activité)
    p         : (N,)        probabilités des scénarios
    epsilon   : niveau de risque ε
    delta     : (N, k, Q, m) int32 — tables de transition (-1 = transition morte)
    accepting : (N, k, Q)   uint8  — états acceptants (0/1)
)doc")
        .def(py::init([](
            py::array_t<double>  profit_arr,
            py::array_t<double>  p_arr,
            double               epsilon,
            py::array_t<int32_t> delta_arr,
            py::array_t<uint8_t> acc_arr)
        {
            auto profit_buf = profit_arr.unchecked<2>();
            auto p_buf      = p_arr.unchecked<1>();
            auto delta_buf  = delta_arr.unchecked<4>();
            auto acc_buf    = acc_arr.unchecked<3>();

            int n = (int)profit_buf.shape(0);
            int mv = (int)profit_buf.shape(1);
            int N = (int)p_buf.shape(0);
            int k = (int)delta_buf.shape(1);
            int Q = (int)delta_buf.shape(2);

            Vec1d profit(n * mv);
            for (int t = 0; t < n; ++t)
                for (int a = 0; a < mv; ++a)
                    profit[t * mv + a] = profit_buf(t, a);

            Vec1d p(N);
            for (int i = 0; i < N; ++i) p[i] = p_buf(i);

            std::vector<SSPScenario> scenarios(N);
            for (int i = 0; i < N; ++i) {
                scenarios[i].automata.resize(k);
                for (int j = 0; j < k; ++j) {
                    scenarios[i].automata[j].delta.assign(Q, Vec1i(mv));
                    scenarios[i].automata[j].accepting.resize(Q);
                    for (int q = 0; q < Q; ++q) {
                        scenarios[i].automata[j].accepting[q] = (acc_buf(i,j,q) != 0);
                        for (int a = 0; a < mv; ++a)
                            scenarios[i].automata[j].delta[q][a] = delta_buf(i,j,q,a);
                    }
                }
            }

            return new SSPProblem(n, mv, k, Q,
                                  std::move(profit), std::move(p), epsilon,
                                  std::move(scenarios));
        }),
        py::arg("profit"), py::arg("p"), py::arg("epsilon"),
        py::arg("delta"), py::arg("accepting"));

    // ── MKPProblemBnB (variante B&B pour benchmark) ───────────────────────────
    py::class_<MKPProblemBnB, MKPProblem>(m, "MKPProblemBnB",
        "Variante B&B de MKPProblem — solve_zeta par Branch-and-Bound exact.")
        .def(py::init([](
            py::array_t<double> c_arr,
            py::array_t<double> p_arr,
            py::array_t<double> A_arr,
            py::array_t<double> b_arr,
            double epsilon)
        {
            return new MKPProblemBnB(
                to_vec1d(c_arr), to_vec1d(p_arr),
                to_vec3d(A_arr), to_vec2d(b_arr),
                epsilon
            );
        }),
        py::arg("c"), py::arg("p"), py::arg("A"), py::arg("b"),
        py::arg("epsilon"));

    // ── MKPProblemLP (variante LP-relax pour benchmark) ──────────────────────
    py::class_<MKPProblemLP, MKPProblem>(m, "MKPProblemLP",
        "Variante LP-relax de MKPProblem — solve_zeta par LP fractionnaire 1D (min sur j).")
        .def(py::init([](
            py::array_t<double> c_arr,
            py::array_t<double> p_arr,
            py::array_t<double> A_arr,
            py::array_t<double> b_arr,
            double epsilon)
        {
            return new MKPProblemLP(
                to_vec1d(c_arr), to_vec1d(p_arr),
                to_vec3d(A_arr), to_vec2d(b_arr),
                epsilon
            );
        }),
        py::arg("c"), py::arg("p"), py::arg("A"), py::arg("b"),
        py::arg("epsilon"));

    // ── MKPCLDProblem ─────────────────────────────────────────────────────────
    py::class_<V1CLDProblem>(m, "V1CLDProblem")
        .def("n_vars",         &V1CLDProblem::n_vars)
        .def("N_scen",         &V1CLDProblem::N_scen)
        .def("n_constraints",  &V1CLDProblem::n_constraints)
        .def("epsilon",        &V1CLDProblem::epsilon);

    py::class_<MKPCLDProblem, V1CLDProblem>(m, "MKPCLDProblem",
        R"doc(
Problème v1CLD pour le MKP chance-contraint (décomposition per-contrainte).

Args:
    c       : (n,)      coûts déterministes
    p       : (N,)      probabilités des scénarios
    A       : (N, m, n) matrices de contraintes
    b       : (N, m)    membres droits
    epsilon : niveau de risque ε
)doc")
        .def(py::init([](
            py::array_t<double> c_arr,
            py::array_t<double> p_arr,
            py::array_t<double> A_arr,
            py::array_t<double> b_arr,
            double epsilon)
        {
            return new MKPCLDProblem(
                to_vec1d(c_arr), to_vec1d(p_arr),
                to_vec3d(A_arr), to_vec2d(b_arr),
                epsilon
            );
        }),
        py::arg("c"), py::arg("p"), py::arg("A"), py::arg("b"),
        py::arg("epsilon"));

    // ── solve_v1cld ───────────────────────────────────────────────────────────
    m.def("solve_v1cld",
        [](V1CLDProblem& problem,
           int    T,
           int    log_interval,
           bool   verbose,
           double time_limit_s,
           double rel_tol,
           int    tol_window,
           double v_lower,
           py::object lambda_init_obj) -> py::dict
        {
            Vec3d lambda_init;
            if (!lambda_init_obj.is_none()) {
                auto arr = lambda_init_obj.cast<py::array_t<double>>();
                auto buf = arr.unchecked<3>();
                int a = buf.shape(0), b = buf.shape(1), c = buf.shape(2);
                lambda_init.assign(a, Vec2d(b, Vec1d(c)));
                for (int i = 0; i < a; ++i)
                    for (int j = 0; j < b; ++j)
                        for (int k = 0; k < c; ++k)
                            lambda_init[i][j][k] = buf(i, j, k);
            }

            V1CLDResult res = solve_v1cld(problem, T, log_interval, verbose,
                                          time_limit_s, rel_tol, tol_window,
                                          v_lower, lambda_init);

            py::dict d;
            d["v1CLD_star"]     = res.v1CLD_star;
            d["convergence"]    = res.convergence;
            d["times_s"]        = res.times_s;
            d["gnorm_history"]  = res.gnorm_history;
            d["z_changes"]      = res.z_changes;
            d["solve_time_s"]   = res.solve_time_s;
            d["iterations_run"] = res.iterations_run;
            d["lambda_star"]    = res.lambda_star;
            return d;
        },
        py::arg("problem"),
        py::arg("T")            = 500,
        py::arg("log_interval") = 1,
        py::arg("verbose")      = false,
        py::arg("time_limit_s") = -1.0,
        py::arg("rel_tol")      = -1.0,
        py::arg("tol_window")   = 500,
        py::arg("v_lower")      = -1e18,
        py::arg("lambda_init")  = py::none(),
        "Solveur v1CLD générique. Retourne v1CLD_star, convergence, times_s, "
        "gnorm_history, z_changes, solve_time_s, iterations_run, lambda_star.");

    // ── SSPCLDProblem ─────────────────────────────────────────────────────────
    py::class_<SSPCLDProblem, V1CLDProblem>(m, "SSPCLDProblem",
        R"doc(
Problème v1CLD pour le CCSSP à contraintes automates (décomposition par automate).

n_constraints() = k — un sous-problème ζ_{i,j} par automate j du scénario i.

Args:
    profit    : (n, m)       matrice de profits déterministe
    p         : (N,)         probabilités des scénarios
    epsilon   : niveau de risque ε
    delta     : (N, k, Q, m) int32 — tables de transition (-1 = mort)
    accepting : (N, k, Q)    uint8  — états acceptants (0/1)
)doc")
        .def(py::init([](
            py::array_t<double>  profit_arr,
            py::array_t<double>  p_arr,
            double               epsilon,
            py::array_t<int32_t> delta_arr,
            py::array_t<uint8_t> acc_arr)
        {
            auto profit_buf = profit_arr.unchecked<2>();
            auto p_buf      = p_arr.unchecked<1>();
            auto delta_buf  = delta_arr.unchecked<4>();
            auto acc_buf    = acc_arr.unchecked<3>();

            int n  = (int)profit_buf.shape(0);
            int mv = (int)profit_buf.shape(1);
            int N  = (int)p_buf.shape(0);
            int k  = (int)delta_buf.shape(1);
            int Q  = (int)delta_buf.shape(2);

            Vec1d profit(n * mv);
            for (int t = 0; t < n; ++t)
                for (int a = 0; a < mv; ++a)
                    profit[t * mv + a] = profit_buf(t, a);

            Vec1d p(N);
            for (int i = 0; i < N; ++i) p[i] = p_buf(i);

            std::vector<SSPScenario> scenarios(N);
            for (int i = 0; i < N; ++i) {
                scenarios[i].automata.resize(k);
                for (int j = 0; j < k; ++j) {
                    scenarios[i].automata[j].delta.assign(Q, Vec1i(mv));
                    scenarios[i].automata[j].accepting.resize(Q);
                    for (int q = 0; q < Q; ++q) {
                        scenarios[i].automata[j].accepting[q] = (acc_buf(i,j,q) != 0);
                        for (int a = 0; a < mv; ++a)
                            scenarios[i].automata[j].delta[q][a] = delta_buf(i,j,q,a);
                    }
                }
            }

            return new SSPCLDProblem(n, mv, k, Q,
                                     std::move(profit), std::move(p), epsilon,
                                     std::move(scenarios));
        }),
        py::arg("profit"), py::arg("p"), py::arg("epsilon"),
        py::arg("delta"), py::arg("accepting"));

    // ── solve_v1ld ────────────────────────────────────────────────────────────
    m.def("solve_v1ld",
        [](V1LDProblem& problem,
           int    T,
           int    log_interval,
           bool   verbose,
           double time_limit_s,
           double rel_tol,
           int    tol_window,
           double v_lower,
           py::object lambda_init_obj) -> py::dict
        {
            Vec2d lambda_init;
            if (!lambda_init_obj.is_none()) {
                auto arr = lambda_init_obj.cast<py::array_t<double>>();
                auto buf = arr.unchecked<2>();
                int rr = buf.shape(0), cc = buf.shape(1);
                lambda_init.assign(rr, Vec1d(cc));
                for (int i = 0; i < rr; ++i)
                    for (int j = 0; j < cc; ++j)
                        lambda_init[i][j] = buf(i, j);
            }

            V1LDResult res = solve_v1ld(problem, T, log_interval, verbose,
                                        time_limit_s, rel_tol, tol_window,
                                        v_lower, lambda_init);

            py::dict d;
            d["v1LD_star"]      = res.v1LD_star;
            d["convergence"]    = res.convergence;
            d["times_s"]        = res.times_s;
            d["gnorm_history"]  = res.gnorm_history;
            d["z_changes"]      = res.z_changes;
            d["solve_time_s"]   = res.solve_time_s;
            d["iterations_run"] = res.iterations_run;
            // lambda_star : Vec2d → liste de listes
            d["lambda_star"]    = res.lambda_star;
            return d;
        },
        py::arg("problem"),
        py::arg("T")            = 500,
        py::arg("log_interval") = 1,
        py::arg("verbose")      = false,
        py::arg("time_limit_s") = -1.0,
        py::arg("rel_tol")      = -1.0,
        py::arg("tol_window")   = 500,
        py::arg("v_lower")      = -1e18,
        py::arg("lambda_init")  = py::none(),
        R"doc(
Solveur v1LD générique par descente de sous-gradient.

Accepte tout objet V1LDProblem (MKPProblem ou SSPProblem).
Retourne un dict : v1LD_star, convergence, times_s, gnorm_history,
                   z_changes, solve_time_s, iterations_run, lambda_star.
)doc");
}
