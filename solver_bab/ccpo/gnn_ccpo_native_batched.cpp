/*
 * gnn_ccpo_native_batched.cpp — cf. gnn_ccpo_native_batched.hpp.
 * Layout des nœuds : s*n_free + i = (s-ième scénario ÉLIGIBLE, actif i)
 * (cf. ccpo_gnn_scriptable_batched.py). λ = h_out * c_scale / N, lignes des
 * scénarios infaisables à 0 (cf. CCPOGNN.predict_lambda).
 */
#include "gnn_ccpo_native_batched.hpp"

#include "ccpo_graph_cpp.hpp"

#include <torch/script.h>
#include <torch/torch.h>
#include <torch/cuda.h>

#include <chrono>
#include <iostream>
#include <cstring>
#include <map>
#include <stdexcept>
#include <utility>

namespace {
using Clock = std::chrono::steady_clock;
double since(const Clock::time_point& t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}
}  // namespace

struct CCPOGNNNativeBatched::Impl {
    torch::jit::script::Module module;
    torch::Device device{torch::kCPU};
    // Cache d'arêtes par (n_free, n_elig) : la topologie ne dépend que de ces deux entiers.
    std::map<std::pair<int, int>, torch::Tensor> edge_cache;
    // Cache de vecteurs de poids d'arêtes (tous à 1) par taille B*E.
    std::map<int64_t, torch::Tensor> ones_cache;
    // Poids façon CCMKP (edge_mode=2) d'UN graphe, par (n_free, n_elig) ; dupliqués B fois à la volée.
    std::map<std::pair<int, int>, torch::Tensor> w_cache;
    // Lus sur le module scripté AVANT freeze (attributs absents des anciens exports -> 1 / 0).
    int feat_version = 1;
    int edge_mode = 0;
};

CCPOGNNNativeBatched::CCPOGNNNativeBatched(const std::string& scripted_path, int torch_threads,
                                           const std::string& device)
    : impl_(new Impl)
{
    torch::set_num_threads(torch_threads);
    if (device == "cuda") {
        if (!torch::cuda::is_available())
            throw std::invalid_argument("CCPOGNNNativeBatched: device=cuda demandé mais CUDA indisponible");
        impl_->device = torch::Device(torch::kCUDA);
    } else if (device != "cpu") {
        throw std::invalid_argument("CCPOGNNNativeBatched: device inconnu (\"" + device + "\"), attendu cpu|cuda");
    }
    impl_->module = torch::jit::load(scripted_path, impl_->device);
    if (impl_->module.hasattr("feat_version")) impl_->feat_version = static_cast<int>(impl_->module.attr("feat_version").toInt());
    if (impl_->module.hasattr("edge_mode")) impl_->edge_mode = static_cast<int>(impl_->module.attr("edge_mode").toInt());
    if (impl_->feat_version < 1 || impl_->feat_version > 3)
        throw std::invalid_argument("CCPOGNNNativeBatched: feat_version non supporte");
    if (impl_->edge_mode != ccpo_graph_cpp::EDGE_NONE && impl_->edge_mode != ccpo_graph_cpp::EDGE_MKP)
        throw std::invalid_argument("CCPOGNNNativeBatched: edge_mode non supporte en C++ (attendu 0=none ou 2=mkp)");
    std::cerr << "[CCPOGNNNativeBatched] " << scripted_path << " : feat_version=" << impl_->feat_version
              << " edge_mode=" << impl_->edge_mode << "\n";
    impl_->module.eval();
    impl_->module = torch::jit::freeze(impl_->module);
    impl_->module = torch::jit::optimize_for_inference(impl_->module);
}

CCPOGNNNativeBatched::~CCPOGNNNativeBatched() = default;

void CCPOGNNNativeBatched::query_batch(
    const CCPOInstanceCtx& ctx, const std::vector<const CCPOQuery*>& batch,
    std::vector<std::vector<std::vector<double>>>& lambda_out)
{
    namespace cg = ccpo_graph_cpp;
    torch::NoGradGuard no_grad;

    const int B = static_cast<int>(batch.size());
    if (B == 0) { lambda_out.clear(); return; }
    const int n_free = batch[0]->res.n_free;
    const int N = ctx.N;
    for (const CCPOQuery* q : batch)
        if (q->res.n_free != n_free)
            throw std::runtime_error("CCPOGNNNativeBatched : n_free non uniforme dans le batch");

    // ── Graphe : features par membre ──
    const auto tg0 = Clock::now();
    std::vector<uint8_t> infeasible_u8(N);
    for (int j = 0; j < N; ++j) infeasible_u8[j] = (*ctx.infeasible)[j] ? 1 : 0;
    int n_elig = 0;
    std::vector<int> eligible;
    for (int j = 0; j < N; ++j) if (!infeasible_u8[j]) { eligible.push_back(j); ++n_elig; }
    const int64_t V = static_cast<int64_t>(n_elig) * n_free;

    const int node_dim = cg::node_dim_for(impl_->feat_version);
    torch::Tensor X = torch::empty({static_cast<int64_t>(B) * V, node_dim}, torch::kFloat32);
    torch::Tensor C = torch::empty({B, cg::CONTEXT_DIM}, torch::kFloat32);
    float* Xp = X.data_ptr<float>();
    float* Cp = C.data_ptr<float>();

    std::vector<double> sigma_flat(static_cast<size_t>(N) * n_free * n_free);
    std::vector<double> rext_flat(static_cast<size_t>(N) * n_free);
    std::vector<int> elig_out(n_elig);
    for (int b = 0; b < B; ++b) {
        const CCPOQuery& q = *batch[b];
        for (int j = 0; j < N; ++j) {
            for (int a = 0; a < n_free; ++a) {
                std::memcpy(&sigma_flat[(static_cast<size_t>(j) * n_free + a) * n_free],
                            q.res.Sigma_free[j][a].data(), sizeof(double) * n_free);
                rext_flat[static_cast<size_t>(j) * n_free + a] = q.res.R_ext[j][a];
            }
        }
        cg::build_node_features_any_cpp(
            impl_->feat_version, n_free, N, q.res.cost_free.data(), sigma_flat.data(), q.b_res.data(),
            rext_flat.data(), infeasible_u8.data(), q.res.Q_res, ctx.U,
            ctx.c_scale, ctx.sigma_scale, Xp + static_cast<size_t>(b) * V * node_dim,
            elig_out.data(), ctx.B);
        cg::build_context_features_cpp(
            n_free, N, q.res.k_res, q.res.Q_res, ctx.U, q.b_res.data(), infeasible_u8.data(),
            q.depth, ctx.n, q.frac_fixed_active, q.cost_fixed,
            ctx.c_scale, ctx.sigma_scale, ctx.epsilon, Cp + static_cast<size_t>(b) * cg::CONTEXT_DIM,
            impl_->feat_version, ctx.B);
    }

    // ── Arêtes : topologie mise en cache, dupliquée B fois en bloc-diagonal ──
    const auto key = std::make_pair(n_free, n_elig);
    auto it = impl_->edge_cache.find(key);
    if (it == impl_->edge_cache.end()) {
        const int64_t E = cg::count_edges_cpp(n_free, n_elig);
        torch::Tensor ei = torch::empty({2, E}, torch::kInt64);
        cg::build_edges_cpp(n_free, n_elig, ei.data_ptr<int64_t>());
        it = impl_->edge_cache.emplace(key, ei.to(impl_->device)).first;
    }
    const torch::Tensor& ei = it->second;
    const int64_t E = ei.size(1);
    torch::Tensor offsets = (torch::arange(B, torch::TensorOptions().dtype(torch::kInt64).device(impl_->device)) * V).view({B, 1, 1});
    torch::Tensor edge_index = (ei.unsqueeze(0).expand({B, 2, E}) + offsets)
                                   .permute({1, 0, 2}).reshape({2, B * E}).contiguous();
    torch::Tensor edge_weight;
    if (impl_->edge_mode == cg::EDGE_MKP) {
        auto wit = impl_->w_cache.find(key);
        if (wit == impl_->w_cache.end()) {
            torch::Tensor w = torch::empty({E}, torch::kFloat32);
            cg::build_edge_weights_mkp_cpp(n_free, n_elig, w.data_ptr<float>());
            wit = impl_->w_cache.emplace(key, w.to(impl_->device)).first;
        }
        edge_weight = wit->second.unsqueeze(0).expand({B, E}).reshape({B * E}).contiguous();
    } else {
        auto oit = impl_->ones_cache.find(B * E);
        if (oit == impl_->ones_cache.end())
            oit = impl_->ones_cache.emplace(B * E, torch::ones({B * E}, torch::TensorOptions().dtype(torch::kFloat32).device(impl_->device))).first;
        edge_weight = oit->second;
    }
    t_graph_s += since(tg0);

    // ── Forward : UN appel pour tout le batch ──
    const auto tf0 = Clock::now();
    torch::Tensor X_dev = X, C_dev = C;
    if (impl_->device.is_cuda()) { X_dev = X.to(impl_->device); C_dev = C.to(impl_->device); }
    std::vector<torch::jit::IValue> inputs{
        X_dev, edge_index, edge_weight, C_dev,
        static_cast<int64_t>(n_free), static_cast<int64_t>(n_elig), static_cast<int64_t>(B)};
    torch::Tensor h = impl_->module.forward(inputs).toTensor();
    if (impl_->device.is_cuda()) h = h.to(torch::kCPU);  // .to(kCPU) synchronise le flux CUDA
    h = h.contiguous();
    t_forward_s += since(tf0);

    // ── λ = h * c_scale / N, rangé dans (N, n_free), zéros aux lignes infaisables ──
    const auto tp0 = Clock::now();
    const float* hp = h.data_ptr<float>();
    lambda_out.assign(B, std::vector<std::vector<double>>(N, std::vector<double>(n_free, 0.0)));
    const double scale = ctx.c_scale / static_cast<double>(N);
    for (int b = 0; b < B; ++b)
        for (int s = 0; s < n_elig; ++s) {
            const int j = eligible[s];
            for (int i = 0; i < n_free; ++i)
                lambda_out[b][j][i] = static_cast<double>(hp[static_cast<size_t>(b) * V + static_cast<size_t>(s) * n_free + i]) * scale;
        }
    t_post_s += since(tp0);
    ++n_queries; n_members += B;
}
