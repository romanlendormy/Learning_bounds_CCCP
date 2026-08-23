/*
 * gnn_v1cld_sd_edit_native.cpp — cf. gnn_v1cld_sd_edit_native.hpp pour le
 * contexte (adaptateur mono-nœud autour de GNNV1CLDSdEditNativeBatched).
 */
#include "gnn_v1cld_sd_edit_native.hpp"

GNNV1CLDSdEditNative::GNNV1CLDSdEditNative(
    const std::string& checkpoint_path, const std::string& device,
    int n, int m, int N, double epsilon,
    const Vec1d& c, const Vec1d& p, const Vec3d& A, const Vec2d& b,
    bool use_surrogate_zeta)
    : batched_(std::make_unique<GNNV1CLDSdEditNativeBatched>(
          checkpoint_path, device, n, m, N, epsilon, c, p, A, b, use_surrogate_zeta))
{}

GNNV1CLDSdEditNative::~GNNV1CLDSdEditNative() = default;

double GNNV1CLDSdEditNative::query(const std::vector<int8_t>& x_mask, const Vec2d& b_res,
                                    double curr_cost, int depth, GNNCallTiming& timing)
{
    const std::vector<std::vector<int8_t>> x_masks{x_mask};
    const std::vector<double> curr_costs{curr_cost};
    const std::vector<Vec2d> b_res_list{b_res};
    std::vector<double> ub_relative_out;
    GNNBatchCallTiming batch_timing;

    batched_->query_batch(x_masks, curr_costs, b_res_list, depth, ub_relative_out, batch_timing);

    // GNNCallTiming distingue graph_features/graph_edges (backend V1LD
    // one-shot, cf. gnn_bound_backend.hpp) ; GNNBatchCallTiming ne les
    // décompose pas (cf. gnn_bound_backend_batch.hpp) — repliés dans
    // t_graph_build_s plutôt que perdus.
    timing.t_graph_build_s    = batch_timing.t_graph_build_s;
    timing.t_graph_features_s = 0.0;
    timing.t_graph_edges_s    = 0.0;
    timing.t_forward_s        = batch_timing.t_forward_s;
    timing.t_bound_compute_s  = batch_timing.t_bound_compute_s;

    return ub_relative_out.at(0);
}
