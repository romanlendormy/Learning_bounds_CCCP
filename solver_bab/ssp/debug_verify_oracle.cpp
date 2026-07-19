/*
 * debug_verify_oracle.cpp — Verification independante (force brute) de
 * l'exactitude de theta_{i,j} et zeta_{i,j} (SSPCLDProblem) sur l'instance
 * qui a revele le probleme de validite de v1CLD.
 *
 * Pour chaque employe, l'ensemble des sequences faisables est enumere par
 * DFS exhaustive sur l'automate (independant de la DP dp_one_employee du
 * code teste) — comparaison a l'aveugle, pas de reutilisation de la meme
 * logique. Pour zeta, TOUS les sous-ensembles d'employes "libres" pouvant
 * etre forces a j sont essayes (E <= ~6 ici, 2^E trivial), au lieu de
 * supposer que le tri glouton par delta est optimal.
 *
 * Teste plusieurs regimes de cout : la structure de base (c), des couts
 * "moderes" aleatoires, et des couts "extremes" (magnitude ~ce qu'on a
 * observe pendant la divergence du sous-gradient, jusqu'a plusieurs
 * centaines/milliers).
 *
 * Usage : debug_verify_oracle <dir_instance> [seed]
 */
#include "ssp_loader.hpp"
#include "../../solver_generic/ssp/ssp_cld_problem_bnb.hpp"
#include "../../solver_generic/ssp/ssp_problem_bnb.hpp"

#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

static constexpr double NEG_INF = -1e30;
static constexpr int SIGMA = 4;

// Adjacence de l'automate (sym, q_next) par etat, reconstruite independamment
// de SSPCLDProblem (pas de reutilisation de code).
struct Automaton {
    int Q, initial;
    std::vector<int> accepting;
    std::vector<std::vector<std::pair<int,int>>> adj;  // adj[q] = [(sym, q_next)]
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

// DFS exhaustive : max sur TOUTES les sequences faisables de longueur T,
// respectant mask_row (t fixe si >=0), et force_t/force_sym (contrainte
// additionnelle, -1 = aucune). cost_e : pointeur vers T*SIGMA valeurs
// (c[e*T*SIGMA + t*SIGMA + sym]).
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

// Le code (contrairement a brute_*, qui s'arrete au premier employe infaisable)
// SOMME les contributions de plusieurs employes infaisables : -1e30, -2e30,
// -3e30... selon combien. Les deux signalent correctement l'infaisabilite,
// pas besoin d'egalite bit-a-bit — seuil qualitatif, pas quantitatif.
static bool both_infeasible(double code_val, double brute_val) {
    return code_val <= NEG_INF / 2 && brute_val <= NEG_INF / 2;
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

    // Base : contribution de tous les employes NON candidats a la selection
    // gloutonne — deja decides (already_fixed), OU libres mais incapables de
    // travailler a j (!feasible). Ces derniers contribuent quand meme via
    // free_val (qui peut valoir -inf s'ils sont reellement infaisables) :
    // un seul employe sans completion valide rend tout le sous-probleme
    // infaisable, meme s'il n'est pas candidat a etre force a j. (Bug initial :
    // ce cas etait omis, la contribution de ces employes silencieusement
    // traitee comme 0 au lieu de leur vraie valeur.)
    double base = 0.0;
    for (int e = 0; e < E; ++e) {
        bool is_cand = !already_fixed[e] && feasible[e];
        if (!is_cand) base += free_val[e];
    }

    // Force brute sur TOUS les sous-ensembles de cand (taille >= d_eff, pas
    // seulement == d_eff — on ne suppose PAS que delta>=0 implique exactement
    // d_eff optimal, on le verifie).
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

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "Usage: debug_verify_oracle <dir> [seed]\n"; return 1; }
    std::string dir = argv[1];
    unsigned seed = (argc > 2) ? static_cast<unsigned>(std::atoi(argv[2])) : 42;

    SSPInstance inst = load_ssp_instance(dir);
    Automaton   aut  = build_automaton(inst);
    const int   E = inst.E, T = inst.T, N = inst.N, d = E * T * SIGMA;
    std::cout << "Instance : E=" << E << " T=" << T << " N=" << N
              << " Q=" << aut.Q << " d=" << d << "\n";

    std::mt19937 rng(seed);

    // Masques testes : racine (tout libre) + quelques masques partiels
    // aleatoires (imitent un noeud en profondeur dans l'arbre).
    std::vector<Vec2i> masks;
    masks.push_back(Vec2i(E, Vec1i(T, -1)));  // racine
    for (int trial = 0; trial < 3; ++trial) {
        Vec2i m(E, Vec1i(T, -1));
        std::uniform_int_distribution<int> dfrac(0, 100);
        std::uniform_int_distribution<int> dsym(0, 3);
        for (int e = 0; e < E; ++e)
            for (int t = 0; t < T; ++t)
                if (dfrac(rng) < 40) m[e][t] = dsym(rng);  // ~40% des positions fixees
        masks.push_back(m);
    }

    struct Regime { const char* name; double scale; };
    std::vector<Regime> regimes = {
        {"base (c = -salaire/0)", 0.0},      // cas special : cost = c directement
        {"modere (~[-10,10])",    10.0},
        {"large (~[-200,200])",   200.0},
        {"extreme (~[-2000,2000])", 2000.0},
    };

    int n_checked = 0, n_mismatch = 0;
    double max_abs_diff = 0.0;

    for (auto& mask : masks) {
        for (auto& reg : regimes) {
            Vec1d cost(d);
            if (reg.scale == 0.0) {
                std::fill(cost.begin(), cost.end(), 0.0);
                for (int e = 0; e < E; ++e)
                    for (int t = 0; t < T; ++t)
                        cost[e * T * SIGMA + t * SIGMA + 3] = -inst.salary[t];
            } else {
                std::uniform_real_distribution<double> dc(-reg.scale, reg.scale);
                for (auto& v : cost) v = dc(rng);
            }

            SSPCLDProblemBnB prob(inst.transitions, inst.accepting, inst.initial,
                                  inst.salary, inst.demand, inst.p, inst.epsilon,
                                  E, mask);
            SSPProblemBnB    prob_ld(inst.transitions, inst.accepting, inst.initial,
                                     inst.salary, inst.demand, inst.p, inst.epsilon,
                                     E, mask);

            // theta (j arbitraire, n'a pas d'effet) — v1CLD
            {
                auto [vt, xt] = prob.solve_theta(0, cost);
                double vt_brute = brute_theta(aut, E, T, mask, cost);
                double diff = std::abs(vt - vt_brute);
                n_checked++;
                if (diff > 1e-6 * std::max(1.0, std::abs(vt_brute)) && !both_infeasible(vt, vt_brute)) {
                    n_mismatch++;
                    std::cout << "  ECART theta[CLD]  [" << reg.name << "] code=" << vt
                              << " brute=" << vt_brute << " diff=" << diff << "\n";
                }
                if (!both_infeasible(vt, vt_brute)) max_abs_diff = std::max(max_abs_diff, diff);
            }

            // theta (v1LD, meme oracle sous-jacent dp_one_employee mais dans
            // SSPShiftProblem — verifie si le meme defaut s'y trouve)
            {
                auto [vt_ld, xt_ld] = prob_ld.solve_theta(cost);
                double vt_brute = brute_theta(aut, E, T, mask, cost);
                double diff = std::abs(vt_ld - vt_brute);
                n_checked++;
                if (diff > 1e-6 * std::max(1.0, std::abs(vt_brute)) && !both_infeasible(vt_ld, vt_brute)) {
                    n_mismatch++;
                    std::cout << "  ECART theta[LD]   [" << reg.name << "] code=" << vt_ld
                              << " brute=" << vt_brute << " diff=" << diff << "\n";
                }
                if (!both_infeasible(vt_ld, vt_brute)) max_abs_diff = std::max(max_abs_diff, diff);
            }

            // zeta pour quelques (i,j) au hasard
            std::uniform_int_distribution<int> di(0, N - 1), dj(0, T - 1);
            for (int k = 0; k < 5; ++k) {
                int i = di(rng), j = dj(rng);
                auto [vz, xz] = prob.solve_zeta(i, j, cost);
                double vz_brute = brute_zeta(aut, E, T, j, inst.demand[i][j], mask, cost);
                double diff = std::abs(vz - vz_brute);
                n_checked++;
                if (diff > 1e-6 * std::max(1.0, std::abs(vz_brute)) && !both_infeasible(vz, vz_brute)) {
                    n_mismatch++;
                    std::cout << "  ECART zeta   [" << reg.name << "] i=" << i << " j=" << j
                              << " demand=" << inst.demand[i][j]
                              << " code=" << vz << " brute=" << vz_brute
                              << " diff=" << diff << "  mask=";
                    for (int e = 0; e < E; ++e) { std::cout << "{"; for (int t : mask[e]) std::cout << t << " "; std::cout << "} "; }
                    std::cout << "\n";
                }
                if (!both_infeasible(vz, vz_brute)) max_abs_diff = std::max(max_abs_diff, diff);
            }
        }
    }

    std::cout << std::fixed << std::setprecision(9);
    std::cout << "\n" << n_checked << " verifications, " << n_mismatch << " ecarts.\n";
    std::cout << "max|code - brute| = " << std::scientific << max_abs_diff << "\n";
    return n_mismatch > 0 ? 1 : 0;
}
