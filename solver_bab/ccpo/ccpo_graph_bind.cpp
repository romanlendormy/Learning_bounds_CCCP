/*
 * ccpo_graph_bind.cpp — Interface C pour ccpo_graph_cpp.hpp, meme patron que
 * ccpo_lagrangian_bind.cpp (extern "C", pointeurs bruts, ctypes). Sert a
 * valider numeriquement ccpo_graph_cpp.hpp contre gnn_ccpo/ccpo_graph.py
 * (cf. validate_ccpo_graph_cpp.py) AVANT de l'utiliser dans le futur driver
 * B&B+GNN batche.
 *
 * Compilation : cible CMake ccpo_graph (solver_bab/CMakeLists.txt).
 */

#include "ccpo_graph_cpp.hpp"

#include <cstdint>

extern "C" {

void ccpo_build_node_features_c(
    int n_free, int N,
    const double* cost_free,
    const double* Sigma_free_flat,
    const double* B_res,
    const double* R_ext_flat,
    const uint8_t* infeasible,
    int Q_res, int U,
    double c_scale, double sigma_scale,
    float* feats_out,
    int* eligible_out)
{
    ccpo_graph_cpp::build_node_features_cpp(
        n_free, N, cost_free, Sigma_free_flat, B_res, R_ext_flat, infeasible,
        Q_res, U, c_scale, sigma_scale, feats_out, eligible_out);
}

void ccpo_build_context_features_c(
    int n_free, int N, int k_res, int Q_res, int U,
    const double* B_res, const uint8_t* infeasible,
    int depth, int n_total, double frac_fixed_active, double cost_fixed,
    double c_scale, double sigma_scale, double epsilon,
    float* ctx_out)
{
    ccpo_graph_cpp::build_context_features_cpp(
        n_free, N, k_res, Q_res, U, B_res, infeasible,
        depth, n_total, frac_fixed_active, cost_fixed,
        c_scale, sigma_scale, epsilon, ctx_out);
}

void ccpo_build_node_features_v2_c(
    int n_free, int N, const double* cost_free, const double* Sigma_free_flat,
    const double* B_res, const double* R_ext_flat, const uint8_t* infeasible,
    int Q_res, int U, double c_scale, double sigma_scale,
    float* feats_out, int* eligible_out)
{
    ccpo_graph_cpp::build_node_features_v2_cpp(
        n_free, N, cost_free, Sigma_free_flat, B_res, R_ext_flat, infeasible,
        Q_res, U, c_scale, sigma_scale, feats_out, eligible_out);
}

void ccpo_build_context_features_v_c(
    int n_free, int N, int k_res, int Q_res, int U,
    const double* B_res, const uint8_t* infeasible,
    int depth, int n_total, double frac_fixed_active, double cost_fixed,
    double c_scale, double sigma_scale, double epsilon,
    float* ctx_out, int feat_version)
{
    ccpo_graph_cpp::build_context_features_cpp(
        n_free, N, k_res, Q_res, U, B_res, infeasible,
        depth, n_total, frac_fixed_active, cost_fixed,
        c_scale, sigma_scale, epsilon, ctx_out, feat_version);
}

void ccpo_build_node_features_v3_c(
    int n_free, int N, const double* cost_free, const double* Sigma_free_flat,
    const double* B_res, const double* R_ext_flat, const uint8_t* infeasible,
    int Q_res, int U, double c_scale, double B_orig,
    float* feats_out, int* eligible_out)
{
    ccpo_graph_cpp::build_node_features_v3_cpp(
        n_free, N, cost_free, Sigma_free_flat, B_res, R_ext_flat, infeasible,
        Q_res, U, c_scale, B_orig, feats_out, eligible_out);
}

void ccpo_build_context_features_v3_c(
    int n_free, int N, int k_res, int Q_res, int U,
    const double* B_res, const uint8_t* infeasible,
    int depth, int n_total, double frac_fixed_active, double cost_fixed,
    double c_scale, double sigma_scale, double epsilon,
    float* ctx_out, int feat_version, double B_orig)
{
    ccpo_graph_cpp::build_context_features_cpp(
        n_free, N, k_res, Q_res, U, B_res, infeasible,
        depth, n_total, frac_fixed_active, cost_fixed,
        c_scale, sigma_scale, epsilon, ctx_out, feat_version, B_orig);
}

void ccpo_build_edge_weights_mkp_c(int n_free, int n_elig, float* w_out) {
    ccpo_graph_cpp::build_edge_weights_mkp_cpp(n_free, n_elig, w_out);
}

int64_t ccpo_count_edges_c(int n_free, int n_elig) {
    return ccpo_graph_cpp::count_edges_cpp(n_free, n_elig);
}

void ccpo_build_edges_c(int n_free, int n_elig, int64_t* edge_index_out) {
    ccpo_graph_cpp::build_edges_cpp(n_free, n_elig, edge_index_out);
}

}  // extern "C"
