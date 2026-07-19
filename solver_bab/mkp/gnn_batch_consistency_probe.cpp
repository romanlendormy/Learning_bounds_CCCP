/*
 * gnn_batch_consistency_probe.cpp — Étape 2d du plan de batching : vérifie
 * que GNNSdEditNativeBatched::query_batch() (B masques, un seul forward)
 * produit EXACTEMENT le même résultat que B appels séparés de
 * GNNSdEditNative::query() (backend de production, INCHANGÉ) — sur une
 * VRAIE instance MKP, pas des tenseurs synthétiques (contrairement à
 * gnn_batch_forward_probe.cpp, qui teste seulement la vitesse).
 *
 * Les B masques sont générés en fixant aléatoirement (n - n_g) items à 0/1
 * (proportion de 1 tirée pour rester faisable en général) — pas une vraie
 * trajectoire de branchement B&B, mais des masques structurellement
 * identiques à ce qu'un nœud réel produirait (même n_g partagé, même
 * calcul de curr_cost/b_res), suffisant pour valider la cohérence numérique
 * de query_batch() sur des données réalistes.
 *
 * Usage : gnn_batch_consistency_probe <dataset.txt> <checkpoint.pt>
 *         [rep=0] [epsilon=0.1] [device=cuda] [n_g_target=10] [B=8] [seed=0]
 */
#include "../../solver_generic/types.hpp"   // Vec1d/Vec2d/Vec3d — pas de dépendance Gecode ici
#include "gnn_sd_edit_native.hpp"
#include "gnn_sd_edit_native_batched.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// ── Loader (copie locale, même convention que main_gecode.cpp) ──────────────
struct MKPInst {
    Vec1d c; Vec1d p; Vec3d A; Vec2d b;
    int n, m, N;
};

static MKPInst load_song_local(const std::string& fp, int rep = 0)
{
    std::ifstream f(fp);
    if (!f) { std::cerr << "Erreur : " << fp << "\n"; std::exit(1); }

    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string tok;
        std::vector<double> vals;
        while (std::getline(iss, tok, ',')) vals.push_back(std::stod(tok));
        if (static_cast<int>(vals[4]) != rep) continue;
        rows.push_back(std::move(vals));
    }
    if (rows.empty()) { std::cerr << "Aucun scénario pour rep=" << rep << "\n"; std::exit(1); }

    int m = static_cast<int>(rows[0][2]);
    int n = static_cast<int>(rows[0][3]);
    int N = static_cast<int>(rows.size());

    Vec3d A(N, Vec2d(m, Vec1d(n, 0.0)));
    Vec2d b(N, Vec1d(m, 0.0));
    Vec1d c(n, 0.0);

    for (int i = 0; i < N; ++i) {
        const auto& v = rows[i];
        for (int j = 0; j < m; ++j)
            for (int k = 0; k < n; ++k)
                A[i][j][k] = v[5 + j * n + k];
        for (int j = 0; j < m; ++j)
            b[i][j] = v[5 + m * n + j];
        if (i == 0)
            for (int k = 0; k < n; ++k)
                c[k] = v[5 + m * n + m + k];
    }
    Vec1d p(N, 1.0 / N);
    return {c, p, A, b, n, m, N};
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage : gnn_batch_consistency_probe <dataset.txt> <checkpoint.pt> "
                     "[rep=0] [epsilon=0.1] [device=cuda] [n_g_target=10] [B=8] [seed=0]\n";
        return 1;
    }
    std::string dataset   = argv[1];
    std::string ckpt_path = argv[2];
    int    rep        = (argc > 3) ? std::stoi(argv[3]) : 0;
    double epsilon     = (argc > 4) ? std::stod(argv[4]) : 0.10;
    std::string device = (argc > 5) ? argv[5] : "cuda";
    int    n_g_target  = (argc > 6) ? std::stoi(argv[6]) : 10;
    int    B           = (argc > 7) ? std::stoi(argv[7]) : 8;
    int    seed        = (argc > 8) ? std::stoi(argv[8]) : 0;

    MKPInst inst = load_song_local(dataset, rep);
    if (n_g_target > inst.n) {
        std::cerr << "n_g_target=" << n_g_target << " > n=" << inst.n << "\n";
        return 1;
    }
    int depth = inst.n - n_g_target;

    std::cout << "[consistency] dataset=" << dataset << "  n=" << inst.n << "  m=" << inst.m
              << "  N=" << inst.N << "  n_g_target=" << n_g_target << "  depth=" << depth
              << "  B=" << B << "  device=" << device << "\n";

    // ── Génère B masques partageant le même n_g (n_g_target) ─────────────────
    std::mt19937 rng(seed);
    std::bernoulli_distribution fix_to_one(0.4);   // proportion d'items fixés à 1 parmi les items fixés
    std::vector<std::vector<int8_t>> x_masks(B, std::vector<int8_t>(inst.n, 2));
    std::vector<double> curr_costs(B, 0.0);

    for (int bIdx = 0; bIdx < B; ++bIdx) {
        std::vector<int> order(inst.n);
        for (int k = 0; k < inst.n; ++k) order[k] = k;
        std::shuffle(order.begin(), order.end(), rng);
        int n_fixed = inst.n - n_g_target;
        double cc = 0.0;
        for (int idx = 0; idx < n_fixed; ++idx) {
            int k = order[idx];
            int8_t val = fix_to_one(rng) ? 1 : 0;
            x_masks[bIdx][k] = val;
            if (val == 1) cc += inst.c[k];
        }
        curr_costs[bIdx] = cc;
    }

    // ── Backend de référence : GNNSdEditNative (production) ──────────────────
    // query() attend désormais b_res déjà calculé (cf. revue niveau 1 point
    // B — le backend ne le recalcule plus lui-même depuis x_mask) : on le
    // dérive ici localement, cette sonde n'ayant pas de suivi incrémental
    // façon MKPSpaceV1LD::b_res.
    auto compute_b_res = [&](const std::vector<int8_t>& x_mask) {
        Vec2d b_res = inst.b;
        for (int k = 0; k < inst.n; ++k) {
            if (x_mask[k] != 1) continue;
            for (int i = 0; i < inst.N; ++i)
                for (int j = 0; j < inst.m; ++j)
                    b_res[i][j] -= inst.A[i][j][k];
        }
        return b_res;
    };

    GNNSdEditNative ref_backend(ckpt_path, device, inst.n, inst.m, inst.N, epsilon,
                                inst.c, inst.p, inst.A, inst.b);
    std::vector<double> ub_ref(B);
    auto t_ref0 = std::chrono::steady_clock::now();
    for (int bIdx = 0; bIdx < B; ++bIdx) {
        GNNCallTiming timing;
        Vec2d b_res = compute_b_res(x_masks[bIdx]);
        ub_ref[bIdx] = ref_backend.query(x_masks[bIdx], b_res, curr_costs[bIdx], depth, timing);
    }
    auto t_ref1 = std::chrono::steady_clock::now();
    double t_ref_s = std::chrono::duration<double>(t_ref1 - t_ref0).count();

    // ── Backend batché : GNNSdEditNativeBatched (nouveau) ────────────────────
    GNNSdEditNativeBatched batch_backend(ckpt_path, device, inst.n, inst.m, inst.N, epsilon,
                                         inst.c, inst.p, inst.A, inst.b);
    std::vector<double> ub_batched;
    GNNBatchCallTiming batch_timing;
    std::vector<Vec2d> b_res_list(B);
    for (int bIdx = 0; bIdx < B; ++bIdx) b_res_list[bIdx] = compute_b_res(x_masks[bIdx]);
    auto t_batch0 = std::chrono::steady_clock::now();
    batch_backend.query_batch(x_masks, curr_costs, b_res_list, depth, ub_batched, batch_timing);
    auto t_batch1 = std::chrono::steady_clock::now();
    double t_batch_s = std::chrono::duration<double>(t_batch1 - t_batch0).count();

    // ── Comparaison ───────────────────────────────────────────────────────────
    double max_diff = 0.0, mean_diff = 0.0;
    for (int bIdx = 0; bIdx < B; ++bIdx) {
        double d = std::abs(ub_ref[bIdx] - ub_batched[bIdx]);
        max_diff = std::max(max_diff, d);
        mean_diff += d;
        std::cout << "  b=" << bIdx << "  ub_ref=" << ub_ref[bIdx]
                  << "  ub_batched=" << ub_batched[bIdx] << "  diff=" << d << "\n";
    }
    mean_diff /= B;

    std::cout << "\n[consistency] max|diff|=" << max_diff << "  mean|diff|=" << mean_diff << "\n";
    std::cout << "[consistency] t_ref(B appels)=" << t_ref_s << "s  t_batched(1 appel)=" << t_batch_s
              << "s  speedup=" << (t_ref_s / t_batch_s) << "x\n";

    const double tol = 1e-3;   // ζ (B&B exact) + arrondi float32/float64 mixte — cf. tolérances déjà en usage (native vs IPC ~1e-4)
    if (max_diff < tol) {
        std::cout << "[consistency] OK — écart max sous la tolérance (" << tol << ")\n";
        return 0;
    }
    std::cout << "[consistency] ÉCHEC — écart max dépasse la tolérance (" << tol << ")\n";
    return 1;
}
