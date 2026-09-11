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
#include "../../solver_generic/ssp/ssp_dld_problem_bnb.hpp"
#include "../../solver_generic/ssp/ssp_problem_bnb.hpp"

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
    std::getline(f, line);  // "lambda_mode dld"
    std::getline(f, line);  // "lambda_dld N m d"
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

    SSPDLDProblemBnB prob(inst.transitions, inst.accepting, inst.initial,
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
    // Reproduit exactement la formule de dld_solver.cpp (A-D), pour voir si
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
    std::cout << "\nRecalcul manuel (formule dld_solver.cpp, mais theta/zeta 100% verifies exacts) :\n";
    std::cout << "  knapsack_val = " << knapsack_val << "\n";
    std::cout << "  L1           = " << L1 << "\n";
    std::cout << "  lb_salary    = " << -L1 << "\n";

    // ── TEST HYPOTHESE (2026-08-12) : Delta_i = max_j(delta_ij) au lieu de
    // (1/m) Sum_j(delta_ij) -- meme theta_vals/theta part de L1, seule la
    // selection y* (via Delta_i) et le terme soustrait changent.
    {
        Vec1d delta_max(NN, 0.0);
        for (int i = 0; i < NN; ++i) {
            double m = 0.0;
            for (int j = 0; j < mm; ++j) m = std::max(m, delta_ij[i][j]);
            delta_max[i] = m;
        }
        std::vector<int> order_max(NN);
        for (int i = 0; i < NN; ++i) order_max[i] = i;
        std::sort(order_max.begin(), order_max.end(),
                  [&](int a, int b){ return delta_max[a] < delta_max[b]; });
        double cumsum_max = 0.0, knapsack_val_max = 0.0;
        for (int idx : order_max) {
            cumsum_max += inst.p[idx];
            knapsack_val_max += inst.p[idx] * delta_max[idx];
            if (cumsum_max >= 1.0 - inst.epsilon - 1e-9) break;
        }
        double L1_max = -knapsack_val_max;
        for (int i = 0; i < NN; ++i)
            for (int j = 0; j < mm; ++j)
                L1_max += inst.p[i] * inv_m * theta_vals[i][j];
        std::cout << "\nVariante Delta_i = max_j(delta_ij) (au lieu de (1/m)Sum_j) :\n";
        std::cout << "  knapsack_val = " << knapsack_val_max << "\n";
        std::cout << "  L1           = " << L1_max << "\n";
        std::cout << "  lb_salary    = " << -L1_max << "\n";
    }

    // ── TEST DECISIF (2026-08-13) : centrage PAR SCENARIO (Sum_j lam[i][j]=0
    // pour CHAQUE i) au lieu du centrage GLOBAL (Lambda_sum sur (i,j) ensemble)
    // -- cf. preuve : theta_j/zeta_j ne sont des sur-estimes valides de
    // theta_i/zeta_i QUE sous cette condition. p_ij est constant en j pour i
    // fixe, donc cost2_ij[k] = c[k] + (lam[i][j][k] - lam_bar_i[k])/p_ij,
    // ou lam_bar_i[k] = moyenne sur j de lam[i][j][k] -- le terme Lambda_sum
    // global disparait algebriquement.
    {
        Vec2d theta2(NN, Vec1d(mm)), zeta2(NN, Vec1d(mm));
        double min_raw2 = 1e30;
        int n_negative2 = 0;
        for (int i = 0; i < NN; ++i) {
            double p_ij = inst.p[i] * inv_m;
            Vec1d lam_bar_i(d, 0.0);
            for (int j = 0; j < mm; ++j)
                for (int k = 0; k < d; ++k)
                    lam_bar_i[k] += lam[i][j][k];
            for (int k = 0; k < d; ++k) lam_bar_i[k] *= inv_m;

            for (int j = 0; j < mm; ++j) {
                Vec1d cost2_ij(d);
                for (int k = 0; k < d; ++k)
                    cost2_ij[k] = c[k] + (lam[i][j][k] - lam_bar_i[k]) / p_ij;

                auto [vt2, xt2] = prob.solve_theta(j, cost2_ij);
                auto [vz2, xz2] = prob.solve_zeta(i, j, cost2_ij);
                theta2[i][j] = vt2;
                zeta2[i][j]  = vz2;
                double raw2 = vt2 - vz2;
                if (raw2 < min_raw2) min_raw2 = raw2;
                if (raw2 < -1e-6) n_negative2++;
            }
        }
        std::cout << "\n=== Centrage par scenario (Sum_j lam[i][j]=0 pour chaque i) ===\n";
        std::cout << "min (theta2_ij - zeta2_ij) = " << min_raw2
                  << "  (" << n_negative2 << " paires negatives sur " << (NN * mm) << ")\n";
        {
            double min_t = 1e300, max_t = -1e300, min_z = 1e300, max_z = -1e300;
            int n_inf_t = 0, n_inf_z = 0;
            for (int i = 0; i < NN; ++i)
                for (int j = 0; j < mm; ++j) {
                    min_t = std::min(min_t, theta2[i][j]); max_t = std::max(max_t, theta2[i][j]);
                    min_z = std::min(min_z, zeta2[i][j]);  max_z = std::max(max_z, zeta2[i][j]);
                    if (theta2[i][j] < -1e29) n_inf_t++;
                    if (zeta2[i][j]  < -1e29) n_inf_z++;
                }
            std::cout << "theta2 range=[" << min_t << ", " << max_t << "]  n_inf=" << n_inf_t << "\n";
            std::cout << "zeta2  range=[" << min_z << ", " << max_z << "]  n_inf=" << n_inf_z << "\n";
            for (int i = 0; i < NN; ++i)
                for (int j = 0; j < mm; ++j)
                    if (zeta2[i][j] < -1e29)
                        std::cout << "  -> scenario infeasible : i=" << i << " j=" << j << "\n";
        }

        Vec1d delta2(NN, 0.0);
        for (int i = 0; i < NN; ++i) {
            double sum = 0.0;
            for (int j = 0; j < mm; ++j) sum += (theta2[i][j] - zeta2[i][j]);
            delta2[i] = inv_m * sum;
        }
        std::vector<int> order2(NN);
        for (int i = 0; i < NN; ++i) order2[i] = i;
        std::sort(order2.begin(), order2.end(),
                  [&](int a, int b){ return delta2[a] < delta2[b]; });
        double cumsum2 = 0.0, knapsack_val2 = 0.0;
        std::cout << "  selection (delta2 croissant) : ";
        for (int idx : order2) {
            cumsum2 += inst.p[idx];
            knapsack_val2 += inst.p[idx] * delta2[idx];
            std::cout << idx << "(d=" << delta2[idx] << ") ";
            if (cumsum2 >= 1.0 - inst.epsilon - 1e-9) break;
        }
        std::cout << "\n  n_scenarios total=" << NN << "  requis (1-eps)*N=" << (1.0-inst.epsilon)*NN << "\n";
        double L1_2 = -knapsack_val2;
        for (int i = 0; i < NN; ++i)
            for (int j = 0; j < mm; ++j)
                L1_2 += inst.p[i] * inv_m * theta2[i][j];

        double sal_ub_raw = 0.0;
        for (int t = 0; t < T; ++t) sal_ub_raw += std::max(0.0, inst.salary[t]);
        sal_ub_raw *= E;

        std::cout << "  knapsack_val = " << knapsack_val2 << "\n";
        std::cout << "  L1           = " << L1_2 << "\n";
        std::cout << "  lb_salary    = " << -L1_2 << "\n";
        std::cout << "  sal_ub (plafond universel, E*Sum_t salaire_t) = " << sal_ub_raw << "\n";
        if (-L1_2 > sal_ub_raw + 1e-6)
            std::cout << "  *** TOUJOURS INVALIDE (lb_salary > sal_ub) ***\n";
        else
            std::cout << "  *** VALIDE (lb_salary <= sal_ub) ***\n";

        // ── TEST DECISIF (2026-08-13) : verification DIRECTE de l'inegalite de
        // dualite faible theta_i(0) <= mean_j theta2_ij(lambda) et
        // zeta_i(0) <= mean_j zeta2_ij(lambda), pour lambda=0 EXACT (pas de
        // correction du tout) -- theta_i(0) = solve_theta(*, c) (j ignore),
        // zeta_i(0) = VRAI joint (les 8 contraintes ensemble, solveur SLD/
        // CP-SAT), pour chaque scenario SELECTIONNE. Si l'inegalite est
        // fausse pour un i, la preuve de dualite faible a un trou reel ici,
        // pas juste un probleme de magnitude de lambda.
        std::cout << "\n=== Verification directe theta_i(0) / zeta_i(0) (lambda=0) ===\n";
        {
            SSPProblemBnB prob_sld(inst.transitions, inst.accepting, inst.initial,
                                     inst.salary, inst.demand, inst.p, inst.epsilon, E, mask);
            auto [theta0, xt0] = prob.solve_theta(0, c);  // j ignore ; meme pour tout i
            std::cout << "theta_i(0) = " << theta0 << "  (identique pour tout i)\n";
            for (int idx : order2) {
                double mean_theta2 = 0.0, mean_zeta2 = 0.0;
                for (int j = 0; j < mm; ++j) { mean_theta2 += theta2[idx][j]; mean_zeta2 += zeta2[idx][j]; }
                mean_theta2 *= inv_m; mean_zeta2 *= inv_m;
                auto [zeta0, xz0] = prob_sld.solve_zeta(idx, c);  // vrai joint, 8 contraintes ensemble
                bool ok_theta = (theta0 <= mean_theta2 + 1e-6);
                bool ok_zeta  = (zeta0  <= mean_zeta2  + 1e-6);
                std::cout << "  i=" << idx
                          << "  zeta_i(0)=" << zeta0
                          << "  mean_zeta2=" << mean_zeta2
                          << "  [" << (ok_zeta ? "OK" : "*** VIOLE ***") << "]"
                          << "   mean_theta2=" << mean_theta2
                          << "  [" << (ok_theta ? "OK" : "*** VIOLE ***") << "]"
                          << "\n";
            }
        }
    }

    // ── Verification force brute du knapsack glouton (min_z) ─────────────────
    // knapsack_min (dld_solver.cpp) trie par effective_delta croissant et
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
