/*
 * debug_replay_violation.cpp — Rejoue EXACTEMENT le (masque, lambda) capture
 * au moment d'une violation sal_ub reelle dans le B&B (cf. SSP_DEBUG_DUMP
 * dans gecode_bab_ssp.hpp::more()), et verifie l'exactitude de theta_{i,j}/
 * zeta_{i,j} contre la force brute pour CE lambda precis — pas un lambda
 * synthetique aleatoire comme dans debug_verify_oracle.
 *
 * Usage : debug_replay_violation <dir_instance> <dump_file>
 */
#include "ssp_loader.hpp"
#include "../../solver_generic/ssp/ssp_cld_problem_bnb.hpp"

#include <algorithm>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

static constexpr double NEG_INF = -1e30;
static constexpr int SIGMA = 4;

struct Automaton {
    int Q, initial;
    std::vector<int> accepting;
    std::vector<std::vector<std::pair<int,int>>> adj;
};

static Automaton build_automaton(const SSPInstance& inst) {
    Automaton a;
    a.initial = inst.initial;
    a.accepting = inst.accepting;
    a.Q = inst.initial + 1;
    for (auto& [s, sym, d] : inst.transitions) a.Q = std::max({a.Q, s + 1, d + 1});
    for (int q : inst.accepting) a.Q = std::max(a.Q, q + 1);
    a.adj.assign(a.Q, {});
    for (auto& [s, sym, d] : inst.transitions) a.adj[s].push_back({sym, d});
    return a;
}

static double brute_max_employee(const Automaton& a, int T, const double* cost_e,
                                  const std::vector<int>& mask_row,
                                  int force_t = -1, int force_sym = -1)
{
    double best = NEG_INF;
    std::function<void(int,int,double)> dfs = [&](int q, int t, double acc) {
        if (t == T) {
            for (int acc_q : a.accepting) if (acc_q == q) { best = std::max(best, acc); return; }
            return;
        }
        int forced = (t == force_t) ? force_sym
                     : (mask_row[t] >= 0 ? mask_row[t] : -1);
        for (auto [sym, qn] : a.adj[q]) {
            if (forced >= 0 && sym != forced) continue;
            dfs(qn, t + 1, acc + cost_e[t * SIGMA + sym]);
        }
    };
    dfs(a.initial, 0, 0.0);
    return best;
}

static double brute_theta(const Automaton& a, int E, int T, const Vec2i& mask, const Vec1d& cost)
{
    double total = 0.0;
    for (int e = 0; e < E; ++e) {
        double v = brute_max_employee(a, T, &cost[e * T * SIGMA], mask[e]);
        if (v <= NEG_INF / 2) return NEG_INF;
        total += v;
    }
    return total;
}

static double brute_zeta(const Automaton& a, int E, int T, int j, int demand_ij,
                         const Vec2i& mask, const Vec1d& cost)
{
    int already_covered = 0;
    for (int e = 0; e < E; ++e) if (mask[e][j] == 3) ++already_covered;
    int d_eff = std::max(0, demand_ij - already_covered);
    if (d_eff <= 0) return brute_theta(a, E, T, mask, cost);

    std::vector<double> free_val(E), work_val(E, NEG_INF);
    std::vector<bool> already_fixed(E, false), feasible(E, false);
    for (int e = 0; e < E; ++e) {
        const double* cost_e = &cost[e * T * SIGMA];
        free_val[e] = brute_max_employee(a, T, cost_e, mask[e]);
        already_fixed[e] = (mask[e][j] >= 0);
        if (!already_fixed[e]) {
            work_val[e] = brute_max_employee(a, T, cost_e, mask[e], j, 3);
            feasible[e] = (work_val[e] > NEG_INF / 2);
        }
    }

    std::vector<int> cand;
    for (int e = 0; e < E; ++e) if (!already_fixed[e] && feasible[e]) cand.push_back(e);
    if (static_cast<int>(cand.size()) < d_eff) return NEG_INF;

    double base = 0.0;
    for (int e = 0; e < E; ++e) {
        bool is_cand = !already_fixed[e] && feasible[e];
        if (!is_cand) base += free_val[e];
    }

    int K = static_cast<int>(cand.size());
    double best = NEG_INF;
    for (int mask_bits = 0; mask_bits < (1 << K); ++mask_bits) {
        int popcount = __builtin_popcount(mask_bits);
        if (popcount < d_eff) continue;
        double total = base;
        for (int b = 0; b < K; ++b) {
            int e = cand[b];
            total += (mask_bits & (1 << b)) ? work_val[e] : free_val[e];
        }
        best = std::max(best, total);
    }
    return best;
}

static bool both_infeasible(double a, double b) { return a <= NEG_INF/2 && b <= NEG_INF/2; }

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "Usage: debug_replay_violation <dir> <dump_file>\n"; return 1; }
    SSPInstance inst = load_ssp_instance(argv[1]);
    Automaton   aut  = build_automaton(inst);
    const int   E = inst.E, T = inst.T, N = inst.N, d = E * T * SIGMA;

    std::ifstream f(argv[2]);
    std::string line;
    std::getline(f, line);  // "mask"
    Vec2i mask(E, Vec1i(T));
    for (int e = 0; e < E; ++e) {
        std::getline(f, line);
        std::istringstream iss(line);
        for (int t = 0; t < T; ++t) iss >> mask[e][t];
    }
    std::getline(f, line);  // "lambda_mode cld"
    std::getline(f, line);  // "lambda_v1cld N m d"
    std::istringstream hdr(line);
    std::string tag; int NN, mm, dd;
    hdr >> tag >> NN >> mm >> dd;
    std::cout << "mode=" << tag << " N=" << NN << " m=" << mm << " d=" << dd << "\n";

    Vec3d lam(NN, Vec2d(mm, Vec1d(dd)));
    for (int i = 0; i < NN; ++i)
        for (int j = 0; j < mm; ++j) {
            std::getline(f, line);
            std::istringstream iss(line);
            for (int k = 0; k < dd; ++k) iss >> lam[i][j][k];
        }

    // Reconstruire c (cout de base : -salaire sur W, 0 ailleurs)
    Vec1d c(d, 0.0);
    for (int e = 0; e < E; ++e)
        for (int t = 0; t < T; ++t)
            c[e * T * SIGMA + t * SIGMA + 3] = -inst.salary[t];

    Vec1d Lambda_sum(d, 0.0);
    for (int i = 0; i < NN; ++i)
        for (int j = 0; j < mm; ++j)
            for (int k = 0; k < d; ++k)
                Lambda_sum[k] += lam[i][j][k];

    SSPCLDProblemBnB prob(inst.transitions, inst.accepting, inst.initial,
                          inst.salary, inst.demand, inst.p, inst.epsilon, E, mask);

    int n_checked = 0, n_mismatch = 0;
    double max_abs_diff = 0.0;
    const double inv_m = 1.0 / mm;

    Vec2d theta_vals(NN, Vec1d(mm)), delta_ij(NN, Vec1d(mm));
    double min_raw_delta = 1e30;
    int    n_negative_raw = 0;

    for (int i = 0; i < NN; ++i) {
        double p_ij = inst.p[i] * inv_m;
        for (int j = 0; j < mm; ++j) {
            Vec1d cost_ij(d);
            for (int k = 0; k < d; ++k)
                cost_ij[k] = c[k] + lam[i][j][k] / p_ij - Lambda_sum[k];

            auto [vt, xt] = prob.solve_theta(j, cost_ij);
            double vt_brute = brute_theta(aut, E, T, mask, cost_ij);
            double dt = std::abs(vt - vt_brute);
            n_checked++;
            if (dt > 1e-6 * std::max(1.0, std::abs(vt_brute)) && !both_infeasible(vt, vt_brute)) {
                n_mismatch++;
                std::cout << "  ECART theta i=" << i << " j=" << j << " code=" << vt
                          << " brute=" << vt_brute << " diff=" << dt << "\n";
            }
            if (!both_infeasible(vt, vt_brute)) max_abs_diff = std::max(max_abs_diff, dt);
            theta_vals[i][j] = vt;

            auto [vz, xz] = prob.solve_zeta(i, j, cost_ij);
            double vz_brute = brute_zeta(aut, E, T, j, inst.demand[i][j], mask, cost_ij);
            double dz = std::abs(vz - vz_brute);
            n_checked++;
            if (dz > 1e-6 * std::max(1.0, std::abs(vz_brute)) && !both_infeasible(vz, vz_brute)) {
                n_mismatch++;
                std::cout << "  ECART zeta  i=" << i << " j=" << j << " code=" << vz
                          << " brute=" << vz_brute << " diff=" << dz << "\n";
            }
            if (!both_infeasible(vz, vz_brute)) max_abs_diff = std::max(max_abs_diff, dz);
            delta_ij[i][j] = std::max(0.0, vt - vz);
            double raw = vt - vz;
            if (raw < min_raw_delta) min_raw_delta = raw;
            if (raw < -1e-6) n_negative_raw++;
        }
    }

    std::cout << std::fixed << std::setprecision(9);
    std::cout << "\n" << n_checked << " verifications, " << n_mismatch << " ecarts.\n";
    std::cout << "max|code - brute| = " << std::scientific << max_abs_diff << "\n";
    std::cout << "min (theta_ij - zeta_ij) SANS clamp max(0,.) = " << min_raw_delta
              << "  (" << n_negative_raw << " paires negatives sur " << (NN * mm) << ")\n";

    // ── Recalcul de L1 a partir de CES theta/zeta (valides exacts) ───────────
    // Reproduit exactement la formule de v1cld_solver.cpp (A-D), pour voir si
    // le bug est dans la formule d'agregation plutot que dans l'oracle.
    Vec1d effective_delta(NN, 0.0);
    for (int i = 0; i < NN; ++i) {
        double sum = 0.0;
        for (int j = 0; j < mm; ++j) sum += delta_ij[i][j];
        effective_delta[i] = inv_m * sum;
    }
    std::vector<int> order(NN);
    for (int i = 0; i < NN; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b){ return effective_delta[a] < effective_delta[b]; });
    double cumsum = 0.0, knapsack_val = 0.0;
    for (int idx : order) {
        cumsum += inst.p[idx];
        knapsack_val += inst.p[idx] * effective_delta[idx];
        if (cumsum >= 1.0 - inst.epsilon - 1e-9) break;
    }
    double L1 = -knapsack_val;
    for (int i = 0; i < NN; ++i)
        for (int j = 0; j < mm; ++j)
            L1 += inst.p[i] * inv_m * theta_vals[i][j];

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nRecalcul manuel (formule v1cld_solver.cpp, mais theta/zeta 100% verifies exacts) :\n";
    std::cout << "  knapsack_val = " << knapsack_val << "\n";
    std::cout << "  L1           = " << L1 << "\n";
    std::cout << "  lb_salary    = " << -L1 << "\n";

    // ── Verification force brute du knapsack glouton (min_z) ─────────────────
    // knapsack_min (v1cld_solver.cpp) trie par effective_delta croissant et
    // s'arrete des que le cumul de p atteint 1-eps — optimal pour poids p_i
    // UNIFORMES (cardinalite fixee), pas garanti optimal en general (0/1
    // covering knapsack, NP-difficile). N<=24 ici : enumeration exhaustive
    // des 2^N sous-ensembles pour verifier independamment.
    if (NN <= 24) {
        uint64_t best_mask = 0;
        double   brute_knap = 1e30;
        for (uint64_t bits = 0; bits < (1ull << NN); ++bits) {
            double cs = 0.0, val = 0.0;
            for (int i = 0; i < NN; ++i)
                if (bits & (1ull << i)) { cs += inst.p[i]; val += inst.p[i] * effective_delta[i]; }
            if (cs >= 1.0 - inst.epsilon - 1e-9 && val < brute_knap) { brute_knap = val; best_mask = bits; }
        }
        std::cout << "\nVerification knapsack (force brute, 2^" << NN << " sous-ensembles) :\n";
        std::cout << "  knapsack_val glouton = " << std::setprecision(9) << knapsack_val << "\n";
        std::cout << "  knapsack_val brute   = " << brute_knap << "\n";
        std::cout << "  ecart = " << (knapsack_val - brute_knap)
                   << (std::abs(knapsack_val - brute_knap) > 1e-6 ? "  <-- GLOUTON SOUS-OPTIMAL" : "  (glouton optimal)")
                   << "\n";
        (void)best_mask;
    }

    return n_mismatch > 0 ? 1 : 0;
}
