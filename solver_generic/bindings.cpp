/*
 * bindings.cpp — Bindings pybind11 pour v1ld_generic_cpp.
 *
 * Expose :
 *   MKPProblem(c, p, A, b, epsilon)
 *   SSPShiftProblem(transitions, accepting, initial, salary, demand, p, epsilon, E)
 *     transitions : (n_trans, 3) int32  — (src, sym_int, dst), sym_int∈{0=O,1=P,2=R,3=W}
 *     accepting   : (n_acc,)    int32   — états acceptants
 *     initial     : int                 — état initial
 *     salary      : (T,)        float64 — coûts salariaux
 *     demand      : (N, T)      int32   — demande[scénario][période]
 *     p           : (N,)        float64 — probabilités
 *     epsilon     : float
 *     E           : int                 — nombre d'employés
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
#include "mkp_cld_problem.hpp"
#include "mkp_cld_problem_dp.hpp"
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

    // ── SSPShiftProblem ───────────────────────────────────────────────────────
    py::class_<SSPShiftProblem, V1LDProblem>(m, "SSPShiftProblem",
        R"doc(
Problème v1LD pour le SSP shift scheduling multi-employés.

E employés partagent un DFA unique de quarts légaux.
Scénarios de demande d[t,k] : couverture requise à la période t.
Objectif : min Σ_e Σ_t salary[t] · 1[x[e,t]=W]  (max −salary en interne).

solve_theta : E DPs indépendants sur le DFA déroulé.  O(E·T·Q·4).
solve_zeta  : CP-SAT exact (AddAutomaton × E + couverture). NP-hard exact.

Args:
    transitions : (n_trans, 3) int32  — [(src, sym_int, dst)]
                  sym_int ∈ {0=O, 1=P, 2=R, 3=W}
    accepting   : (n_acc,)    int32   — états acceptants
    initial     : int                 — état initial
    salary      : (T,)        float64 — coûts salariaux (positifs)
    demand      : (N, T)      int32   — demand[scénario][période]
    p           : (N,)        float64 — probabilités (Σ=1)
    epsilon     : float               — niveau de risque
    E           : int                 — nombre d'employés
)doc")
        .def(py::init([](
            py::array_t<int32_t> trans_arr,
            py::array_t<int32_t> acc_arr,
            int                  initial,
            py::array_t<double>  salary_arr,
            py::array_t<int32_t> demand_arr,
            py::array_t<double>  p_arr,
            double               epsilon,
            int                  E)
        {
            // transitions : (n_trans, 3) — (src, sym, dst)
            auto tb = trans_arr.unchecked<2>();
            int  nt = (int)tb.shape(0);
            std::vector<std::tuple<int,int,int>> transitions(nt);
            for (int i = 0; i < nt; ++i)
                transitions[i] = {tb(i,0), tb(i,1), tb(i,2)};

            // accepting : (n_acc,)
            auto ab = acc_arr.unchecked<1>();
            std::vector<int> accepting((int)ab.shape(0));
            for (int i = 0; i < (int)accepting.size(); ++i)
                accepting[i] = ab(i);

            // salary : (T,)
            auto sb = salary_arr.unchecked<1>();
            int T = (int)sb.shape(0);
            Vec1d salary(T);
            for (int t = 0; t < T; ++t) salary[t] = sb(t);

            // demand : (N, T)
            auto db = demand_arr.unchecked<2>();
            int N = (int)db.shape(0);
            Vec2i demand(N, Vec1i(T));
            for (int i = 0; i < N; ++i)
                for (int t = 0; t < T; ++t)
                    demand[i][t] = db(i, t);

            // p : (N,)
            auto pb = p_arr.unchecked<1>();
            Vec1d p(N);
            for (int i = 0; i < N; ++i) p[i] = pb(i);

            return new SSPShiftProblem(transitions, accepting, initial,
                                       salary, demand, std::move(p), epsilon, E);
        }),
        py::arg("transitions"), py::arg("accepting"), py::arg("initial"),
        py::arg("salary"), py::arg("demand"), py::arg("p"),
        py::arg("epsilon"), py::arg("E"));

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

    // ── MKPCLDProblem (ζ par B&B 1D — défaut) ────────────────────────────────
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

    // ── MKPCLDProblemDP (ζ par DP O(n×W)) ────────────────────────────────────
    py::class_<MKPCLDProblemDP, MKPCLDProblem>(m, "MKPCLDProblemDP",
        R"doc(
Variante DP de MKPCLDProblem — solve_zeta par DP O(n × W) au lieu du B&B.
Même interface que MKPCLDProblem, utilisable avec solve_v1cld.

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
            return new MKPCLDProblemDP(
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

    // ── SSPCLDProblem : TODO (v1CLD hors-scope) ──────────────────────────────
    // La décomposition v1CLD pour le SSP n'est pas encore implémentée.
    // Voir ssp/ssp_cld_problem.hpp pour les TODO.

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
