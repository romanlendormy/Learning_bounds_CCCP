/*
 * debug_verify_v1ld_zeta.cpp — Verifie l'exactitude de zeta_i (v1LD,
 * SSPProblemBnB::solve_zeta, CP-SAT) contre une force brute JOINTE
 * (contrairement a v1CLD, zeta_i impose les T contraintes de couverture
 * SIMULTANEMENT -> il faut enumerer les E employes ENSEMBLE, pas periode
 * par periode).
 *
 * Objectif : tester specifiquement l'hypothese "arrondi CPSAT_SCALE=1e6" -
 * solve_zeta_masked (ssp_problem.cpp) demande a CP-SAT de MAXIMISER
 * l'objectif ARRONDI (round(cost*1e6)), puis reconstruit 'val' via le cout
 * BRUT (non arrondi) de la solution trouvee. Si l'arrondi fait basculer
 * l'argmax de CP-SAT vers une solution differente du vrai optimum en cout
 * brut, 'val' sous-estime le vrai zeta_i -> borne v1LD invalide.
 *
 * Usage : debug_verify_v1ld_zeta <dir_instance> <dump_file_v1ld>
 * (dump_file_v1ld : capture SSP_DEBUG_DUMP en mode V1LD, cf.
 * debug_capture_v1ld.cpp)
 */
#include "ssp_loader.hpp"
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

// Enumere TOUTES les sequences [T] valides pour un employe (DFS sur le DAG,
// respectant mask_row), retourne la liste de (valeur_brute, work_indicator[T]).
// Independant de dp_one_employee (pas de reutilisation de code).
static void enumerate_employee(const Automaton& a, int T, const double* cost_e,
                                const std::vector<int>& mask_row,
                                std::vector<std::pair<double, std::vector<int>>>& out)
{
    std::vector<int> path;
    std::function<void(int,int,double)> dfs = [&](int q, int t, double acc) {
        if (t == T) {
            for (int acc_q : a.accepting)
                if (acc_q == q) { out.push_back({acc, path}); return; }
            return;
        }
        int forced = mask_row[t] >= 0 ? mask_row[t] : -1;
        for (auto [sym, qn] : a.adj[q]) {
            if (forced >= 0 && sym != forced) continue;
            path.push_back(sym);
            dfs(qn, t + 1, acc + cost_e[t * SIGMA + sym]);
            path.pop_back();
        }
    };
    dfs(a.initial, 0, 0.0);
}

// Force brute JOINTE : max sur TOUS les E employes simultanement, sous la
// contrainte de couverture Sum_e work[e][t] >= demand[i][t] POUR TOUS LES t
// A LA FOIS (contrairement a v1CLD qui decompose par t). E<=4 ici -> produit
// cartesien des sequences feasibles par employe, trivial.
static double brute_zeta_joint(const Automaton& a, int E, int T,
                                const Vec1i& demand_i,
                                const Vec2i& mask, const Vec1d& cost)
{
    std::vector<std::vector<std::pair<double, std::vector<int>>>> per_employee(E);
    for (int e = 0; e < E; ++e)
        enumerate_employee(a, T, &cost[e * T * SIGMA], mask[e], per_employee[e]);

    for (int e = 0; e < E; ++e)
        if (per_employee[e].empty()) return NEG_INF;  // employe sans completion valide

    std::vector<int> idx(E, 0);
    double best = NEG_INF;
    std::function<void(int,double,std::vector<std::vector<int>>&)> rec =
        [&](int e, double acc, std::vector<std::vector<int>>& chosen) {
        if (e == E) {
            for (int t = 0; t < T; ++t) {
                int cov = 0;
                for (int ee = 0; ee < E; ++ee) if (chosen[ee][t] == 3) ++cov;
                if (cov < demand_i[t]) return;  // couverture violee a t
            }
            best = std::max(best, acc);
            return;
        }
        for (auto& [val, seq] : per_employee[e]) {
            chosen[e] = seq;
            rec(e + 1, acc + val, chosen);
        }
    };
    std::vector<std::vector<int>> chosen(E);
    rec(0, 0.0, chosen);
    return best;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "Usage: debug_verify_v1ld_zeta <dir> <dump_file_v1ld>\n"; return 1; }
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
    std::getline(f, line);  // "lambda_mode ld"
    std::getline(f, line);  // "lambda_v1ld N d"
    std::istringstream hdr(line);
    std::string tag; int NN, dd;
    hdr >> tag >> NN >> dd;
    std::cout << "mode=" << tag << " N=" << NN << " d=" << dd << "\n";

    Vec2d lam(NN, Vec1d(dd));
    for (int i = 0; i < NN; ++i) {
        std::getline(f, line);
        std::istringstream iss(line);
        for (int k = 0; k < dd; ++k) iss >> lam[i][k];
    }

    Vec1d c(d, 0.0);
    for (int e = 0; e < E; ++e)
        for (int t = 0; t < T; ++t)
            c[e * T * SIGMA + t * SIGMA + 3] = -inst.salary[t];

    Vec1d Lambda_sum(d, 0.0);
    for (int i = 0; i < NN; ++i)
        for (int k = 0; k < d; ++k)
            Lambda_sum[k] += lam[i][k];

    SSPProblemBnB prob(inst.transitions, inst.accepting, inst.initial,
                       inst.salary, inst.demand, inst.p, inst.epsilon, E, mask);

    int n_checked = 0, n_mismatch = 0;
    double max_abs_diff = 0.0;

    for (int i = 0; i < NN; ++i) {
        Vec1d cost_i(d);
        for (int k = 0; k < d; ++k)
            cost_i[k] = c[k] + lam[i][k] / inst.p[i] - Lambda_sum[k];

        auto [vz, xz] = prob.solve_zeta(i, cost_i);
        double vz_brute = brute_zeta_joint(aut, E, T, inst.demand[i], mask, cost_i);
        double diff = std::abs(vz - vz_brute);
        n_checked++;
        bool both_inf = (vz <= NEG_INF/2 && vz_brute <= NEG_INF/2);
        if (diff > 1e-6 * std::max(1.0, std::abs(vz_brute)) && !both_inf) {
            n_mismatch++;
            std::cout << std::setprecision(9)
                      << "  ECART zeta_v1ld i=" << i << " code(CP-SAT)=" << vz
                      << " brute(joint)=" << vz_brute << " diff=" << diff << "\n";
        }
        if (!both_inf) max_abs_diff = std::max(max_abs_diff, diff);
    }

    std::cout << std::fixed << std::setprecision(9);
    std::cout << "\n" << n_checked << " verifications (zeta_i, v1LD/CP-SAT vs force brute jointe), "
              << n_mismatch << " ecarts.\n";
    std::cout << "max|code - brute| = " << std::scientific << max_abs_diff << "\n";

    return n_mismatch > 0 ? 1 : 0;
}
