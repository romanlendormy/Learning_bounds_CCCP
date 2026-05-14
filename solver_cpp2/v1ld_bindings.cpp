/*
 * v1ld_bindings.cpp — Bindings pybind11 pour v1ld_cpp2.
 */

#include "v1ld_core.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(v1ld_cpp2, m) {
    m.doc() = "v1LD solver — C++17, sous-problèmes par DP, pas sous-gradient 1/√t";

    // ── solve : boucle sous-gradient complète ────────────────────────────────
    m.def("solve",
        [](
            py::array_t<double> c_arr,
            py::array_t<double> p_arr,
            py::array_t<double> A_arr,
            py::array_t<double> b_arr,
            double epsilon,
            int    T,
            bool   decomp,
            bool   verbose,
            double time_limit_s,
            int    log_interval,
            double rel_tol,
            int    tol_window,
            py::object lambda_init_obj
        ) -> py::dict {
            auto c_buf = c_arr.unchecked<1>();
            auto p_buf = p_arr.unchecked<1>();
            auto A_buf = A_arr.unchecked<3>();
            auto b_buf = b_arr.unchecked<2>();

            int n = static_cast<int>(c_buf.shape(0));
            int N = static_cast<int>(p_buf.shape(0));
            int m = static_cast<int>(b_buf.shape(1));

            Vec1d c(n), p(N);
            for (int k = 0; k < n; ++k) c[k] = c_buf(k);
            for (int i = 0; i < N; ++i) p[i] = p_buf(i);

            Vec3d A(N, Vec2d(m, Vec1d(n)));
            Vec2d b(N, Vec1d(m));
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < m; ++j) {
                    b[i][j] = b_buf(i, j);
                    for (int k = 0; k < n; ++k)
                        A[i][j][k] = A_buf(i, j, k);
                }

            Vec2d lam_init;
            if (!lambda_init_obj.is_none()) {
                auto lam_arr = lambda_init_obj.cast<py::array_t<double>>();
                auto lam_buf = lam_arr.unchecked<2>();
                lam_init.assign(N, Vec1d(n));
                for (int i = 0; i < N; ++i)
                    for (int k = 0; k < n; ++k)
                        lam_init[i][k] = lam_buf(i, k);
            }

            SolverResult res = solve_impl(
                c, p, A, b, epsilon,
                T, decomp, verbose,
                time_limit_s, log_interval, rel_tol, tol_window,
                lam_init
            );

            py::dict d;
            d["v1LD_star"]             = res.v1LD_star;
            d["convergence"]           = res.convergence;
            d["times_s"]               = res.times_s;
            d["gnorm_history"]         = res.gnorm_history;
            d["z_changes"]             = res.z_changes;
            d["solve_time_s"]          = res.solve_time_s;
            d["iterations_run"]        = res.iterations_run;
            d["lambda_star"]           = res.lambda_star;
            if (decomp) d["mu_star"]   = res.mu_star;
            d["g_history"]             = res.g_history;           // list[snapshot] de (N×n)
            d["xbar_outer_history"]    = res.xbar_outer_history;  // list[snapshot] de (n,)
            d["zstar_history"]         = res.zstar_history;       // list[snapshot] de (N,)
            d["iter_history"]          = res.iter_history;        // list[snapshot] des numéros d'itération
            return d;
        },
        py::arg("c"),
        py::arg("p"),
        py::arg("A"),
        py::arg("b"),
        py::arg("epsilon"),
        py::arg("T")            = 500,
        py::arg("decomp")       = true,
        py::arg("verbose")      = false,
        py::arg("time_limit_s") = -1.0,
        py::arg("log_interval") = 1,
        py::arg("rel_tol")      = -1.0,
        py::arg("tol_window")   = 500,
        py::arg("lambda_init")  = py::none()
    );

    // ── solve_ahmed : descente de coordonnées multi-départ + recherche ternaire
    m.def("solve_ahmed",
        [](
            py::array_t<double> c_arr,
            py::array_t<double> p_arr,
            py::array_t<double> A_arr,
            py::array_t<double> b_arr,
            double              epsilon,
            int                 T_outer,
            int                 T_search,
            double              search_radius,
            int                 n_restarts,
            double              init_scale,
            bool                verbose,
            double              time_limit_s,
            py::object          lambda_init_obj,
            unsigned int        seed
        ) -> py::dict {
            auto c_buf = c_arr.unchecked<1>();
            auto p_buf = p_arr.unchecked<1>();
            auto A_buf = A_arr.unchecked<3>();
            auto b_buf = b_arr.unchecked<2>();

            int n = static_cast<int>(c_buf.shape(0));
            int N = static_cast<int>(p_buf.shape(0));
            int m = static_cast<int>(b_buf.shape(1));

            Vec1d c(n), p(N);
            for (int k = 0; k < n; ++k) c[k] = c_buf(k);
            for (int i = 0; i < N; ++i) p[i] = p_buf(i);

            Vec3d A(N, Vec2d(m, Vec1d(n)));
            Vec2d b(N, Vec1d(m));
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < m; ++j) {
                    b[i][j] = b_buf(i, j);
                    for (int k = 0; k < n; ++k)
                        A[i][j][k] = A_buf(i, j, k);
                }

            Vec2d lam_init;
            if (!lambda_init_obj.is_none()) {
                auto lam_arr = lambda_init_obj.cast<py::array_t<double>>();
                auto lam_buf = lam_arr.unchecked<2>();
                lam_init.assign(N, Vec1d(n));
                for (int i = 0; i < N; ++i)
                    for (int k = 0; k < n; ++k)
                        lam_init[i][k] = lam_buf(i, k);
            }

            SolverResult res = solve_ahmed(
                c, p, A, b, epsilon,
                T_outer, T_search, search_radius,
                n_restarts, init_scale,
                verbose, time_limit_s, lam_init, seed
            );

            py::dict d;
            d["v1LD_star"]      = res.v1LD_star;
            d["convergence"]    = res.convergence;
            d["times_s"]        = res.times_s;
            d["solve_time_s"]   = res.solve_time_s;
            d["iterations_run"] = res.iterations_run;
            d["lambda_star"]    = res.lambda_star;
            return d;
        },
        py::arg("c"),
        py::arg("p"),
        py::arg("A"),
        py::arg("b"),
        py::arg("epsilon"),
        py::arg("T_outer")       = 20,
        py::arg("T_search")      = 60,
        py::arg("search_radius") = 50.0,
        py::arg("n_restarts")    = 5,
        py::arg("init_scale")    = 1.0,
        py::arg("verbose")       = false,
        py::arg("time_limit_s")  = -1.0,
        py::arg("lambda_init")   = py::none(),
        py::arg("seed")          = 42u
    );

    // ── evaluate : calcule L1(λ) et g(λ) à λ fixé, sans mise à jour ─────────
    m.def("evaluate",
        [](
            py::array_t<double> c_arr,
            py::array_t<double> p_arr,
            py::array_t<double> A_arr,
            py::array_t<double> b_arr,
            double              epsilon,
            py::array_t<double> lam_arr
        ) -> py::dict {
            auto c_buf   = c_arr.unchecked<1>();
            auto p_buf   = p_arr.unchecked<1>();
            auto A_buf   = A_arr.unchecked<3>();
            auto b_buf   = b_arr.unchecked<2>();
            auto lam_buf = lam_arr.unchecked<2>();

            int n = static_cast<int>(c_buf.shape(0));
            int N = static_cast<int>(p_buf.shape(0));
            int m = static_cast<int>(b_buf.shape(1));

            Vec1d c(n), p(N);
            for (int k = 0; k < n; ++k) c[k] = c_buf(k);
            for (int i = 0; i < N; ++i) p[i] = p_buf(i);

            Vec3d A(N, Vec2d(m, Vec1d(n)));
            Vec2d b(N, Vec1d(m));
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < m; ++j) {
                    b[i][j] = b_buf(i, j);
                    for (int k = 0; k < n; ++k)
                        A[i][j][k] = A_buf(i, j, k);
                }

            Vec2d lam(N, Vec1d(n));
            for (int i = 0; i < N; ++i)
                for (int k = 0; k < n; ++k)
                    lam[i][k] = lam_buf(i, k);

            auto res = evaluate_L1(c, p, A, b, epsilon, lam);

            py::dict d;
            d["L1"]          = res.L1_val;
            d["g"]           = res.g;
            d["x_bar"]       = res.x_bar;
            d["x_scenarios"] = res.x_scenarios;
            d["z_star"]      = res.z_star;
            return d;
        },
        py::arg("c"),
        py::arg("p"),
        py::arg("A"),
        py::arg("b"),
        py::arg("epsilon"),
        py::arg("lam")
    );
}
