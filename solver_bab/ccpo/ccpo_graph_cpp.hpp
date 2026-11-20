#pragma once
/*
 * ccpo_graph_cpp.hpp — Port C++ EXACT de gnn/ccpo/ccpo_graph.py (construction
 * du graphe d'entree du GNN CCPO), pour etre appele a CHAQUE noeud du futur
 * driver B&B+GNN batche SANS repasser par Python (miroir du pattern MKP :
 * gnn_sd_edit_native_batched.cpp::compute_active_x_invariant/build_features_cpp
 * /build_edges_sd_edit_cpp).
 *
 * DOIT rester synchronise ligne a ligne avec ccpo_graph.py -- toute
 * divergence casserait silencieusement la validite du graphe presente au
 * GNN (le GNN continuerait a produire UN lambda, potentiellement de
 * mauvaise qualite, mais JAMAIS invalide au sens de la borne -- cf. dualite
 * faible -- donc ce genre de bug ne se verrait PAS a l'exactitude de la
 * borne, seulement a sa qualite -- raison de plus pour valider
 * numeriquement contre ccpo_graph.py AVANT integration au B&B, cf.
 * validate_ccpo_graph_cpp.py).
 *
 * Convention de layout (DOIT matcher ccpo_gnn_scriptable_batched.py) : pour
 * un graphe, noeud d'indice s*n_free + i correspond au (s-ieme scenario
 * ELIGIBLE, actif libre i) -- scenario-majeur, actif-mineur.
 *
 * Pas de cache ici (contrairement a MKP) : build_node_features_cpp depend de
 * cost_free/Sigma_free/B_res/R_ext, qui changent a CHAQUE noeud (pas
 * invariant par n_free seul, contrairement a la topologie build_edges_cpp,
 * qui elle ne depend que de (n_free, n_elig) et EST mise en cache par
 * l'appelant (cf. futur driver, pattern Impl::edge_cache de MKP).
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace ccpo_graph_cpp {

constexpr int NODE_DIM = 10;
constexpr int CONTEXT_DIM = 11;
constexpr int NODE_DIM_V2 = 26;  // cf. ccpo_graph.py::build_node_features_v2
constexpr int NODE_DIM_V3 = 28;  // cf. ccpo_graph.py::build_node_features_v3 (normalisations CCMKP)

// Modes de ponderation des aretes -- DOIVENT matcher ccpo_graph.py::EDGE_MODES et
// l'attribut edge_mode du module scripte. EDGE_SIGMA (ablation D) n'est PAS porte en C++.
enum EdgeMode : int { EDGE_NONE = 0, EDGE_SIGMA = 1, EDGE_MKP = 2 };

inline int node_dim_for(int feat_version) {
    return feat_version == 3 ? NODE_DIM_V3 : feat_version == 2 ? NODE_DIM_V2 : NODE_DIM;
}

// ── dantzig_like_rank — rang decroissant du ratio cost/risk, normalise [0,1] ──
inline std::vector<double> dantzig_like_rank(
    const std::vector<double>& cost_free, const std::vector<double>& risk_proxy)
{
    const int n = static_cast<int>(cost_free.size());
    std::vector<double> ratio(n);
    for (int i = 0; i < n; ++i) ratio[i] = cost_free[i] / std::max(risk_proxy[i], 1e-9);

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return ratio[a] > ratio[b]; });  // argsort(-ratio)

    std::vector<double> rank(n);
    const double denom = static_cast<double>(std::max(1, n - 1));
    for (int k = 0; k < n; ++k) rank[order[k]] = static_cast<double>(k) / denom;
    return rank;
}

/*
 * build_node_features_cpp — remplit `feats` (n_elig*n_free, NODE_DIM),
 * row-major (deja alloue par l'appelant, taille = n_elig*n_free*NODE_DIM).
 *
 * Sigma_free : (N, n_free, n_free) dense, applati row-major --
 * Sigma_free[j][a][b] == Sigma_free_flat[(j*n_free + a)*n_free + b].
 * cost_free : (n_free,). B_res, R_ext(N,n_free) : idem. infeasible : (N,) 0/1.
 * eligible_out : (n_elig,) -- indices j (0..N-1) des scenarios eligibles,
 * DANS L'ORDRE ou ils ont ete places dans `feats` (meme convention que
 * ccpo_graph.build_graph::eligible/scenario_id/asset_id).
 */
inline void build_node_features_cpp(
    int n_free, int N,
    const double* cost_free,        // (n_free,)
    const double* Sigma_free_flat,  // (N, n_free, n_free)
    const double* B_res,            // (N,)
    const double* R_ext_flat,       // (N, n_free)
    const uint8_t* infeasible,      // (N,)
    int Q_res, int U,
    double c_scale, double sigma_scale,
    float* feats_out,               // (n_elig*n_free, NODE_DIM) -- deja alloue
    int* eligible_out)              // (n_elig,) -- deja alloue
{
    std::vector<int> eligible;
    for (int j = 0; j < N; ++j) if (!infeasible[j]) eligible.push_back(j);
    const int n_elig = static_cast<int>(eligible.size());
    for (int s = 0; s < n_elig; ++s) eligible_out[s] = eligible[s];

    auto Sigma = [&](int j, int a, int b) -> double {
        return Sigma_free_flat[(static_cast<size_t>(j) * n_free + a) * n_free + b];
    };

    // diag[j][i], ligne/corr -- calcules sur TOUS les N scenarios (cf. ccpo_graph.py :
    // inst.Sigma_free a toujours la forme (N,n_free,n_free), diag=np.diagonal(...)
    // porte sur les N lignes, meme pour les scenarios presolve-infaisibles).
    std::vector<std::vector<double>> diag(N, std::vector<double>(n_free));
    std::vector<std::vector<double>> corr_mean(N, std::vector<double>(n_free));
    std::vector<std::vector<double>> corr_std(N, std::vector<double>(n_free));
    const double denom = static_cast<double>(std::max(1, n_free - 1));

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < n_free; ++i) {
            const double d = Sigma(j, i, i);
            diag[j][i] = d;
            double row_sum = 0.0, row_sq_sum = 0.0;
            for (int k = 0; k < n_free; ++k) {
                const double v = Sigma(j, i, k);
                row_sum += v;
                row_sq_sum += v * v;
            }
            row_sum -= d;
            row_sq_sum -= d * d;
            const double cm = row_sum / denom;
            const double cvar = std::max(row_sq_sum / denom - cm * cm, 0.0);
            corr_mean[j][i] = cm;
            corr_std[j][i] = std::sqrt(cvar);
        }
    }

    std::vector<double> diag_mean_over_scen(n_free, 0.0), diag_std_over_scen(n_free, 0.0);
    for (int i = 0; i < n_free; ++i) {
        double sum = 0.0;
        for (int j = 0; j < N; ++j) sum += diag[j][i];
        const double mean = sum / N;
        double var = 0.0;
        for (int j = 0; j < N; ++j) var += (diag[j][i] - mean) * (diag[j][i] - mean);
        diag_mean_over_scen[i] = mean;
        diag_std_over_scen[i] = std::sqrt(var / N);  // np.std, population (ddof=0), comme numpy par defaut
    }

    std::vector<double> cost_free_v(cost_free, cost_free + n_free);
    std::vector<double> risk_proxy(n_free);
    for (int i = 0; i < n_free; ++i) risk_proxy[i] = diag_mean_over_scen[i] + 1e-9;
    std::vector<double> dantzig = dantzig_like_rank(cost_free_v, risk_proxy);

    int row = 0;
    for (int s = 0; s < n_elig; ++s) {
        const int j = eligible[s];
        for (int i = 0; i < n_free; ++i) {
            float* out = feats_out + static_cast<size_t>(row + i) * NODE_DIM;
            out[0] = static_cast<float>(cost_free[i] / c_scale);
            out[1] = static_cast<float>(diag[j][i] / sigma_scale);
            out[2] = static_cast<float>(diag[j][i] / std::max(B_res[j], 1e-9));
            out[3] = static_cast<float>(corr_mean[j][i] / sigma_scale);
            out[4] = static_cast<float>(corr_std[j][i] / sigma_scale);
            out[5] = static_cast<float>(dantzig[i]);
            out[6] = static_cast<float>(diag_mean_over_scen[i] / sigma_scale);
            out[7] = static_cast<float>(diag_std_over_scen[i] / sigma_scale);
            out[8] = static_cast<float>(B_res[j] / std::max(sigma_scale * Q_res * Q_res, 1e-9));
            out[9] = static_cast<float>(R_ext_flat[static_cast<size_t>(j) * n_free + i] / std::max(sigma_scale * U, 1e-9));
        }
        row += n_free;
    }
}

/*
 * Rang decroissant normalise dans [0,1], egalites departagees par indice croissant
 * (std::stable_sort == np.argsort(-x, kind="stable")) -- utilise par les features v2.
 */
inline void rank_desc_stable(const std::vector<double>& x, std::vector<double>& rank) {
    const int n = static_cast<int>(x.size());
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return x[a] > x[b]; });
    rank.assign(n, 0.0);
    const double denom = static_cast<double>(std::max(1, n - 1));
    for (int k = 0; k < n; ++k) rank[order[k]] = static_cast<double>(k) / denom;
}

// Moyenne et ecart-type population (ddof=0), DEUX PASSES -- meme formule que _pop_std (Python).
inline void mean_pop_std(const std::vector<double>& v, double& mean, double& sd) {
    const size_t n = v.size();
    if (n == 0) { mean = 0.0; sd = 0.0; return; }
    double s = 0.0;
    for (double x : v) s += x;
    mean = s / static_cast<double>(n);
    double q = 0.0;
    for (double x : v) q += (x - mean) * (x - mean);
    sd = std::sqrt(q / static_cast<double>(n));
}

/*
 * build_node_features_v2_cpp — port EXACT de ccpo_graph.py::build_node_features_v2
 * (NODE_DIM_V2 = 26 colonnes, cf. la docstring Python pour la liste). Memes
 * arguments et meme layout que build_node_features_cpp ; feats_out est
 * (n_elig*n_free, NODE_DIM_V2). Statistiques sur les scenarios ELIGIBLES seulement.
 */
inline void build_node_features_v2_cpp(
    int n_free, int N,
    const double* cost_free, const double* Sigma_free_flat, const double* B_res,
    const double* R_ext_flat, const uint8_t* infeasible,
    int Q_res, int U, double c_scale, double sigma_scale,
    float* feats_out, int* eligible_out)
{
    std::vector<int> eligible;
    for (int j = 0; j < N; ++j) if (!infeasible[j]) eligible.push_back(j);
    const int E = static_cast<int>(eligible.size());
    for (int s = 0; s < E; ++s) eligible_out[s] = eligible[s];
    if (E == 0) return;
    const int n = n_free;
    const double sig = sigma_scale;
    const double rx_den = std::max(sig * U, 1e-9);
    const double qm = static_cast<double>(std::max(std::min(U, Q_res), 0));
    const double qm2 = qm * qm;  // q_max^2, cf. docstring Python (r_ji, saturation)
    auto Sigma = [&](int j, int a, int b) -> double {
        return Sigma_free_flat[(static_cast<size_t>(j) * n + a) * n + b];
    };

    // Grandeurs par (s, i), s = indice dans eligible.
    std::vector<double> d(static_cast<size_t>(E) * n), cm(d.size()), cs(d.size()), r(d.size()), rx(d.size());
    std::vector<double> off_mean(E, 0.0), off_std(E, 0.0), Bv(E);
    const double denom = static_cast<double>(std::max(1, n - 1));
    for (int s = 0; s < E; ++s) {
        const int j = eligible[s];
        Bv[s] = B_res[j];
        const double bden = std::max(B_res[j], 1e-9);
        std::vector<double> offv;
        offv.reserve(static_cast<size_t>(n) * (n - 1 > 0 ? n - 1 : 0));
        for (int i = 0; i < n; ++i) {
            const double di = Sigma(j, i, i);
            double rs = 0.0, rq = 0.0;
            for (int k = 0; k < n; ++k) { const double v = Sigma(j, i, k); rs += v; rq += v * v; }
            rs -= di; rq -= di * di;
            const double m = rs / denom;
            const size_t id = static_cast<size_t>(s) * n + i;
            d[id] = di;
            cm[id] = m;
            cs[id] = std::sqrt(std::max(rq / denom - m * m, 0.0));
            r[id] = std::min(qm2 * di / bden, 10.0);
            rx[id] = R_ext_flat[static_cast<size_t>(j) * n + i] / rx_den;
        }
        if (n > 1) {
            for (int a = 0; a < n; ++a)
                for (int b = 0; b < n; ++b)
                    if (a != b) offv.push_back(Sigma(j, a, b));  // ordre ligne-majeur, comme S[:, off]
            mean_pop_std(offv, off_mean[s], off_std[s]);
        }
    }

    // Statistiques par actif sur les scenarios eligibles : (moyenne, ecart-type) de d, cm, cs, r, rx.
    const std::vector<double>* srcs[5] = {&d, &cm, &cs, &r, &rx};
    std::vector<double> pa_mean(5 * static_cast<size_t>(n)), pa_std(5 * static_cast<size_t>(n));
    std::vector<double> col(E);
    for (int q = 0; q < 5; ++q)
        for (int i = 0; i < n; ++i) {
            for (int s = 0; s < E; ++s) col[s] = (*srcs[q])[static_cast<size_t>(s) * n + i];
            mean_pop_std(col, pa_mean[q * n + i], pa_std[q * n + i]);
        }

    std::vector<double> ratio(n), dantzig;
    for (int i = 0; i < n; ++i) ratio[i] = cost_free[i] / std::max(pa_mean[0 * n + i] + 1e-9, 1e-9);
    rank_desc_stable(ratio, dantzig);

    const double bt_den = std::max(sig * Q_res * Q_res, 1e-9);
    std::vector<double> rowv(n), rank_s;
    for (int s = 0; s < E; ++s) {
        // Statistiques par scenario sur les actifs.
        double r_min = std::numeric_limits<double>::infinity(), r_mean, r_std, d_mean, d_std;
        int n_sat = 0;
        for (int i = 0; i < n; ++i) {
            const size_t id = static_cast<size_t>(s) * n + i;
            r_min = std::min(r_min, r[id]);
            rowv[i] = r[id];
            if (qm2 * d[id] > Bv[s]) ++n_sat;
        }
        mean_pop_std(rowv, r_mean, r_std);
        for (int i = 0; i < n; ++i) rowv[i] = d[static_cast<size_t>(s) * n + i];
        mean_pop_std(rowv, d_mean, d_std);
        for (int i = 0; i < n; ++i) ratio[i] = cost_free[i] / std::max(d[static_cast<size_t>(s) * n + i], 1e-9);
        rank_desc_stable(ratio, rank_s);

        for (int i = 0; i < n; ++i) {
            const size_t id = static_cast<size_t>(s) * n + i;
            float* o = feats_out + id * NODE_DIM_V2;
            o[0] = static_cast<float>(cost_free[i] / c_scale);
            o[1] = static_cast<float>(d[id] / sig);
            o[2] = static_cast<float>(r[id]);
            o[3] = static_cast<float>(cm[id] / sig);
            o[4] = static_cast<float>(cs[id] / sig);
            o[5] = static_cast<float>(dantzig[i]);
            o[6] = static_cast<float>(pa_mean[0 * n + i] / sig);
            o[7] = static_cast<float>(pa_std[0 * n + i] / sig);
            o[8] = static_cast<float>(Bv[s] / bt_den);
            o[9] = static_cast<float>(rx[id]);
            o[10] = static_cast<float>(r_min);
            o[11] = static_cast<float>(r_mean);
            o[12] = static_cast<float>(r_std);
            o[13] = static_cast<float>(static_cast<double>(n_sat) / n);
            o[14] = static_cast<float>(off_mean[s] / sig);
            o[15] = static_cast<float>(off_std[s] / sig);
            o[16] = static_cast<float>(d_std / sig);
            o[17] = static_cast<float>(pa_mean[1 * n + i] / sig);
            o[18] = static_cast<float>(pa_std[1 * n + i] / sig);
            o[19] = static_cast<float>(pa_mean[2 * n + i] / sig);
            o[20] = static_cast<float>(pa_std[2 * n + i] / sig);
            o[21] = static_cast<float>(pa_mean[3 * n + i]);
            o[22] = static_cast<float>(pa_std[3 * n + i]);
            o[23] = static_cast<float>(pa_mean[4 * n + i]);
            o[24] = static_cast<float>(pa_std[4 * n + i]);
            o[25] = static_cast<float>(rank_s[i]);
        }
    }
}

/*
 * build_node_features_v3_cpp — port EXACT de ccpo_graph.py::build_node_features_v3
 * (NODE_DIM_V3 = 28, cf. la docstring Python). Comme v2 mais unite de risque =
 * budget D'ORIGINE B_orig (a = U^2 d / B_orig, analogue de A/b de CCMKP), plus
 * f_eff = log1p((c/c_scale)/a) (col 26) et cap_sat = 1[B_res/B_orig < 0.1] (col 27).
 */
inline void build_node_features_v3_cpp(
    int n_free, int N,
    const double* cost_free, const double* Sigma_free_flat, const double* B_res,
    const double* R_ext_flat, const uint8_t* infeasible,
    int Q_res, int U, double c_scale, double B_orig,
    float* feats_out, int* eligible_out)
{
    std::vector<int> eligible;
    for (int j = 0; j < N; ++j) if (!infeasible[j]) eligible.push_back(j);
    const int E = static_cast<int>(eligible.size());
    for (int s = 0; s < E; ++s) eligible_out[s] = eligible[s];
    if (E == 0) return;
    const int n = n_free;
    const double B0 = std::max(B_orig, 1e-9);
    const double u2b = static_cast<double>(U) * static_cast<double>(U) / B0;
    const double ub = static_cast<double>(U) / B0;
    const double qm = static_cast<double>(std::max(std::min(U, Q_res), 0));
    const double qm2 = qm * qm;
    auto Sigma = [&](int j, int a, int b) -> double {
        return Sigma_free_flat[(static_cast<size_t>(j) * n + a) * n + b];
    };

    std::vector<double> d(static_cast<size_t>(E) * n), a(d.size()), acm(d.size()), acs(d.size()),
        r(d.size()), rx(d.size());
    std::vector<double> off_mean(E, 0.0), off_std(E, 0.0), Bv(E), cap(E);
    const double denom = static_cast<double>(std::max(1, n - 1));
    for (int s = 0; s < E; ++s) {
        const int j = eligible[s];
        Bv[s] = B_res[j];
        cap[s] = B_res[j] / B0;
        const double bden = std::max(B_res[j], 1e-9);
        std::vector<double> offv;
        offv.reserve(static_cast<size_t>(n) * (n - 1 > 0 ? n - 1 : 0));
        for (int i = 0; i < n; ++i) {
            const double di = Sigma(j, i, i);
            double rs = 0.0, rq = 0.0;
            for (int k = 0; k < n; ++k) { const double v = Sigma(j, i, k); rs += v; rq += v * v; }
            rs -= di; rq -= di * di;
            const double m = rs / denom;
            const size_t id = static_cast<size_t>(s) * n + i;
            d[id] = di;
            a[id] = di * u2b;
            acm[id] = m * u2b;
            acs[id] = std::sqrt(std::max(rq / denom - m * m, 0.0)) * u2b;
            r[id] = std::min(qm2 * di / bden, 10.0);
            rx[id] = R_ext_flat[static_cast<size_t>(j) * n + i] * ub;
        }
        if (n > 1) {
            for (int p = 0; p < n; ++p)
                for (int q = 0; q < n; ++q)
                    if (p != q) offv.push_back(Sigma(j, p, q));
            mean_pop_std(offv, off_mean[s], off_std[s]);
        }
    }

    // Statistiques par actif sur les scenarios eligibles : (moyenne, ecart-type) de a, acm, acs, r, rx.
    const std::vector<double>* srcs[5] = {&a, &acm, &acs, &r, &rx};
    std::vector<double> pa_mean(5 * static_cast<size_t>(n)), pa_std(5 * static_cast<size_t>(n));
    std::vector<double> col(E), d_mean(n);
    for (int q = 0; q < 5; ++q)
        for (int i = 0; i < n; ++i) {
            for (int s = 0; s < E; ++s) col[s] = (*srcs[q])[static_cast<size_t>(s) * n + i];
            mean_pop_std(col, pa_mean[q * n + i], pa_std[q * n + i]);
        }
    for (int i = 0; i < n; ++i) {
        double acc = 0.0;
        for (int s = 0; s < E; ++s) acc += d[static_cast<size_t>(s) * n + i];
        d_mean[i] = acc / E;
    }

    std::vector<double> ratio(n), dantzig;
    for (int i = 0; i < n; ++i) ratio[i] = cost_free[i] / std::max(d_mean[i] + 1e-9, 1e-9);
    rank_desc_stable(ratio, dantzig);

    std::vector<double> rowv(n), rank_s;
    for (int s = 0; s < E; ++s) {
        double r_min = std::numeric_limits<double>::infinity(), r_mean, r_std, a_mean, a_std;
        int n_sat = 0;
        for (int i = 0; i < n; ++i) {
            const size_t id = static_cast<size_t>(s) * n + i;
            r_min = std::min(r_min, r[id]);
            rowv[i] = r[id];
            if (qm2 * d[id] > Bv[s]) ++n_sat;
        }
        mean_pop_std(rowv, r_mean, r_std);
        for (int i = 0; i < n; ++i) rowv[i] = a[static_cast<size_t>(s) * n + i];
        mean_pop_std(rowv, a_mean, a_std);
        for (int i = 0; i < n; ++i) ratio[i] = cost_free[i] / std::max(d[static_cast<size_t>(s) * n + i], 1e-9);
        rank_desc_stable(ratio, rank_s);

        for (int i = 0; i < n; ++i) {
            const size_t id = static_cast<size_t>(s) * n + i;
            const double cn = cost_free[i] / c_scale;
            float* o = feats_out + id * NODE_DIM_V3;
            o[0] = static_cast<float>(cn);
            o[1] = static_cast<float>(a[id]);
            o[2] = static_cast<float>(r[id]);
            o[3] = static_cast<float>(acm[id]);
            o[4] = static_cast<float>(acs[id]);
            o[5] = static_cast<float>(dantzig[i]);
            o[6] = static_cast<float>(pa_mean[0 * n + i]);
            o[7] = static_cast<float>(pa_std[0 * n + i]);
            o[8] = static_cast<float>(cap[s]);
            o[9] = static_cast<float>(rx[id]);
            o[10] = static_cast<float>(r_min);
            o[11] = static_cast<float>(r_mean);
            o[12] = static_cast<float>(r_std);
            o[13] = static_cast<float>(static_cast<double>(n_sat) / n);
            o[14] = static_cast<float>(off_mean[s] * u2b);
            o[15] = static_cast<float>(off_std[s] * u2b);
            o[16] = static_cast<float>(a_std);
            o[17] = static_cast<float>(pa_mean[1 * n + i]);
            o[18] = static_cast<float>(pa_std[1 * n + i]);
            o[19] = static_cast<float>(pa_mean[2 * n + i]);
            o[20] = static_cast<float>(pa_std[2 * n + i]);
            o[21] = static_cast<float>(pa_mean[3 * n + i]);
            o[22] = static_cast<float>(pa_std[3 * n + i]);
            o[23] = static_cast<float>(pa_mean[4 * n + i]);
            o[24] = static_cast<float>(pa_std[4 * n + i]);
            o[25] = static_cast<float>(rank_s[i]);
            o[26] = static_cast<float>(std::log1p(cn / std::max(a[id], 1e-9)));
            o[27] = cap[s] < 0.1 ? 1.0f : 0.0f;
        }
    }
}

/*
 * build_node_features_any_cpp — aiguillage par version (1 = build_node_features_cpp,
 * 2 = build_node_features_v2_cpp, 3 = build_node_features_v3_cpp, qui exige B_orig > 0).
 * feats_out : (n_elig*n_free, node_dim_for(feat_version)).
 */
inline void build_node_features_any_cpp(
    int feat_version, int n_free, int N,
    const double* cost_free, const double* Sigma_free_flat, const double* B_res,
    const double* R_ext_flat, const uint8_t* infeasible,
    int Q_res, int U, double c_scale, double sigma_scale,
    float* feats_out, int* eligible_out, double B_orig = 0.0)
{
    if (feat_version == 3)
        build_node_features_v3_cpp(n_free, N, cost_free, Sigma_free_flat, B_res, R_ext_flat,
                                   infeasible, Q_res, U, c_scale, B_orig, feats_out, eligible_out);
    else if (feat_version == 2)
        build_node_features_v2_cpp(n_free, N, cost_free, Sigma_free_flat, B_res, R_ext_flat,
                                   infeasible, Q_res, U, c_scale, sigma_scale, feats_out, eligible_out);
    else
        build_node_features_cpp(n_free, N, cost_free, Sigma_free_flat, B_res, R_ext_flat,
                                infeasible, Q_res, U, c_scale, sigma_scale, feats_out, eligible_out);
}

/*
 * build_context_features_cpp — remplit ctx_out (CONTEXT_DIM,).
 */
inline void build_context_features_cpp(
    int n_free, int N, int k_res, int Q_res, int U,
    const double* B_res, const uint8_t* infeasible,
    int depth, int n_total, double frac_fixed_active, double cost_fixed,
    double c_scale, double sigma_scale, double epsilon,
    float* ctx_out,  // (CONTEXT_DIM,)
    int feat_version = 1,  // v2 : ctx[5] = n_elig/N (v1 : 0, reserve)
    double B_orig = 0.0)   // v3 seulement : ctx[2] = min_E B_res/B_orig, ctx[7-8] sur n_elig
{
    int n_eligible = 0;
    double tightest = std::numeric_limits<double>::infinity();
    const double t_den = feat_version >= 3 ? std::max(B_orig, 1e-9)
                                           : std::max(sigma_scale * Q_res * Q_res, 1e-9);
    for (int j = 0; j < N; ++j) {
        if (infeasible[j]) continue;
        ++n_eligible;
        const double v = B_res[j] / t_den;
        tightest = std::min(tightest, v);
    }
    if (n_eligible == 0) tightest = 0.0;
    const int n_scen = feat_version >= 3 ? n_eligible : N;

    ctx_out[0] = static_cast<float>(depth / static_cast<double>(std::max(1, n_total)));
    ctx_out[1] = static_cast<float>(frac_fixed_active);
    ctx_out[2] = static_cast<float>(tightest);
    ctx_out[3] = static_cast<float>(cost_fixed / c_scale);
    ctx_out[4] = static_cast<float>(epsilon);
    ctx_out[5] = feat_version >= 2
        ? static_cast<float>(n_eligible / static_cast<double>(std::max(1, N))) : 0.0f;
    ctx_out[6] = static_cast<float>(std::log1p(static_cast<double>(n_free)));
    ctx_out[7] = static_cast<float>(std::log1p(static_cast<double>(n_scen)));
    ctx_out[8] = static_cast<float>(n_scen / static_cast<double>(std::max(1, n_free)));
    ctx_out[9] = static_cast<float>(k_res / static_cast<double>(std::max(1, n_free)));
    ctx_out[10] = static_cast<float>(Q_res / static_cast<double>(std::max(1, n_free * U)));
}

/*
 * count_edges_cpp — nombre d'aretes que produira build_edges_cpp, pour que
 * l'appelant puisse allouer edge_index_out AVANT d'appeler build_edges_cpp
 * (meme pattern que n_elig pour build_node_features_cpp).
 */
inline int64_t count_edges_cpp(int n_free, int n_elig) {
    if (n_elig == 0) return 0;
    const int64_t clique_per_scen = static_cast<int64_t>(n_free) * (n_free - 1);  // a!=b, deja bidirectionnel
    int64_t total = clique_per_scen * n_elig;
    if (n_elig > 1) total += 2LL * n_free * (n_elig - 1);  // etoile : anchor->others + others->anchor
    return total;
}

/*
 * build_edges_cpp — remplit edge_index_out (2, n_edges) -- layout "deux
 * lignes contigues" (src[0..E-1] puis dst[0..E-1]), comme un tenseur (2,E)
 * row-major -- pret a etre utilise tel quel par ccpo_gnn_scriptable_batched.
 * n_edges DOIT valoir count_edges_cpp(n_free, n_elig) (alloue par l'appelant).
 */
inline void build_edges_cpp(int n_free, int n_elig, int64_t* edge_index_out) {
    if (n_elig == 0) return;
    const int64_t n_edges = count_edges_cpp(n_free, n_elig);
    int64_t* src = edge_index_out;
    int64_t* dst = edge_index_out + n_edges;
    int64_t pos = 0;

    // Clique actif-actif, par scenario (bloc de n_free noeuds consecutifs).
    for (int s = 0; s < n_elig; ++s) {
        const int64_t base = static_cast<int64_t>(s) * n_free;
        for (int a = 0; a < n_free; ++a) {
            for (int b = 0; b < n_free; ++b) {
                if (a == b) continue;
                src[pos] = base + a;
                dst[pos] = base + b;
                ++pos;
            }
        }
    }

    // Etoile scenario-scenario, a actif fixe : scenario 0 (du sous-ensemble
    // eligible) relie a tous les autres, pour chaque actif i.
    if (n_elig > 1) {
        for (int i = 0; i < n_free; ++i) {
            const int64_t anchor = i;  // 0*n_free + i
            for (int s = 1; s < n_elig; ++s) {
                const int64_t other = static_cast<int64_t>(s) * n_free + i;
                src[pos] = anchor; dst[pos] = other; ++pos;
            }
            for (int s = 1; s < n_elig; ++s) {
                const int64_t other = static_cast<int64_t>(s) * n_free + i;
                src[pos] = other; dst[pos] = anchor; ++pos;
            }
        }
    }
}

/*
 * build_edge_weights_mkp_cpp — poids facon CCMKP, MEME ordre que build_edges_cpp :
 * clique intra-scenario 1/n_free, etoile inter-scenarios 1 (cf.
 * ccpo_graph.py::build_edge_weights_mkp). w_out : count_edges_cpp(n_free, n_elig).
 */
inline void build_edge_weights_mkp_cpp(int n_free, int n_elig, float* w_out) {
    if (n_elig == 0) return;
    const int64_t n_clique = static_cast<int64_t>(n_elig) * n_free * (n_free - 1);
    const int64_t n_total = count_edges_cpp(n_free, n_elig);
    const float wc = static_cast<float>(1.0 / n_free);
    for (int64_t e = 0; e < n_clique; ++e) w_out[e] = wc;
    for (int64_t e = n_clique; e < n_total; ++e) w_out[e] = 1.0f;
}

}  // namespace ccpo_graph_cpp
