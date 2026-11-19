// presolve_bind.cpp — binding ctypes : presolve exact certifié (C++) et ancien presolve, pour la validation
// et le recalcul des drapeaux de scénarios (collect/recompute_presolve_flags.py). Cible CMake presolve_exact.
#include "ccpo_sld.hpp"

static std::vector<std::vector<double>> to_mat(int n, const double* S) {
    std::vector<std::vector<double>> M(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) M[i][j] = S[i * n + j];
    return M;
}

extern "C" {
// out = {lower, primal, iterations, optimal}
void ccpo_box_min_certified_c(int n, double Q, double U, const double* S, double* out) {
    const auto r = ccpo_presolve_exact::box_min_certified(n, Q, U, to_mat(n, S));
    out[0] = r.lower; out[1] = r.primal; out[2] = r.iterations; out[3] = r.optimal ? 1.0 : 0.0;
}
double ccpo_box_min_legacy_c(int n, double Q, double U, const double* S) {
    return presolve_min_variance_box(n, Q, U, to_mat(n, S));
}
}
