/*
 * bench_v1cld_subproblems.cpp — Benchmark élémentaire du coût de calcul de
 * ζ_{i,j} pour V1CLD (MKP) : trois méthodes, sur les MÊMES cost vectors,
 * à λ=0 (cold start, cost_{i,j} = c pour tout i,j — cf. v1cld_base.hpp:27).
 *
 *   bnb  : MKPCLDProblem::solve_zeta       — B&B 1D (solve_zeta_i_exact_bb,
 *     mkp_subproblems.hpp). MÉTHODE RÉELLEMENT UTILISÉE en production
 *     (solver_bab/mkp/gecode_bab.hpp:1035) malgré le libellé "V1CLD (ζ DP 1D
 *     exact)" imprimé par main_gecode.cpp — ce libellé décrit la nature
 *     mathématique du sous-problème (1D knapsack, exactement résoluble par
 *     DP), pas l'algorithme réellement exécuté.
 *   dp   : MKPCLDProblemDP::solve_zeta     — DP O(n × B) avec buffer plat
 *     pré-alloué (mkp_cld_problem_dp.cpp). Jamais utilisée en C++ avant ce
 *     benchmark (seulement câblée côté bindings Python). Corrigée pour
 *     scaler poids/capacité par SCALE=10000 (même valeur que
 *     MKPSharedData::SCALE, gecode_bab.hpp:162) avant d'arrondir à l'entier.
 *     VERDICT (mini-manifest n=10) : exacte (max|diff| vs bnb ~1e-13) mais
 *     ~86000x PLUS LENTE que bnb — le O(n×B) avec B~3.2e7 (capacité réaliste
 *     × SCALE) domine complètement. Piste écartée pour l'accélération.
 *   mitm : solve_zeta_i_exact_mitm         — meet-in-the-middle
 *     (Horowitz-Sahni, mkp_subproblems.hpp). Travaille en flottant natif
 *     (pas de discrétisation poids/capacité) : pas de risque mémoire/
 *     précision contrairement à dp. O(2^(n/2) × n/2).
 *
 * ATTENTION MÉMOIRE (dp) : la table DP est (n+1)×(B+1) doubles, où
 * B = floor(max_j(b_j) × SCALE). Avec SCALE=10000 et des capacités
 * réalistes (b_j ~ quelques milliers), B peut atteindre ~3-4×10^7 →
 * plusieurs Go pour une seule table. Ce benchmark imprime la taille
 * projetée AVANT de construire MKPCLDProblemDP.
 *
 * Les valeurs ζ_{i,j} de dp et mitm doivent coïncider exactement avec bnb
 * (même problème résolu par des algorithmes différents) — l'écart max de
 * chacune vs bnb (référence de production) est rapporté comme vérification
 * de correction, pas seulement de vitesse : si l'écart n'est pas nul,
 * comparer les vitesses n'a pas de sens.
 *
 * Usage :
 *   bench_v1cld_subproblems <manifest.txt> [epsilon=0.10]
 *
 * manifest.txt : une ligne par instance "chemin_dataset.txt,rep" (même
 * format que bench_v1ld_subproblems.cpp / main_gecode.cpp).
 */

#include "mkp_cld_problem.hpp"
#include "mkp_cld_problem_dp.hpp"
#include "mkp_subproblems.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// Doit rester synchronisé avec SCALE dans mkp_cld_problem_dp.cpp (dupliqué là
// pour la même raison : solver_generic ne dépend pas de solver_bab). Utilisé
// ici uniquement pour PROJETER la taille mémoire avant de construire l'objet
// DP réel — ne change aucun calcul.
constexpr int SCALE_FOR_PROJECTION = 10000;

// ── Loader (copie locale — cf. bench_v1ld_subproblems.cpp / main_gecode.cpp) ─
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
    if (rows.empty()) { std::cerr << "Aucun scénario pour rep=" << rep << " (" << fp << ")\n"; std::exit(1); }

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

// ── Statistiques de flux ──────────────────────────────────────────────────
struct Stats {
    long   n = 0;
    double sum = 0.0, sumsq = 0.0;
    double min_v = std::numeric_limits<double>::infinity(), max_v = 0.0;

    void add(double x) {
        ++n; sum += x; sumsq += x * x;
        min_v = std::min(min_v, x);
        max_v = std::max(max_v, x);
    }
    double mean() const { return n ? sum / n : 0.0; }
    double var() const {
        if (n < 2) return 0.0;
        double m = mean();
        return std::max(0.0, sumsq / n - m * m);
    }
    double stdev() const { return std::sqrt(var()); }
};

static const std::vector<std::string> METHOD_NAMES = {"bnb", "dp", "mitm"};
constexpr int N_METHODS   = 3;
constexpr int PROD_METHOD = 0; // bnb — référence de production ET de correction

struct InstanceResult {
    std::string tag;
    int    n = 0, m = 0, N = 0;
    long   n_calls = 0;
    std::vector<Stats>  zeta;              // taille N_METHODS
    std::vector<double> max_abs_diff;      // taille N_METHODS, vs PROD_METHOD (0.0 pour lui-même)
    double t_loop_total_s = 0.0;
    double projected_dp_gb = 0.0;
    long long projected_B_max = 0;
};

static InstanceResult bench_one_instance(const std::string& path, int rep, double epsilon, bool run_dp)
{
    MKPInst inst = load_song_local(path, rep);

    // Projection AVANT de construire l'objet DP (qui alloue dans son ctor).
    double b_max = 0.0;
    for (int i = 0; i < inst.N; ++i)
        for (int j = 0; j < inst.m; ++j)
            b_max = std::max(b_max, inst.b[i][j]);
    long long B_max_proj = static_cast<long long>(std::floor(b_max * SCALE_FOR_PROJECTION));
    double dp_bytes = static_cast<double>(inst.n + 1) * static_cast<double>(B_max_proj + 1) * sizeof(double);

    InstanceResult res;
    res.tag = path + "#rep" + std::to_string(rep);
    res.n = inst.n; res.m = inst.m; res.N = inst.N;
    res.zeta.assign(N_METHODS, Stats{});
    res.max_abs_diff.assign(N_METHODS, 0.0);
    res.projected_dp_gb = dp_bytes / 1e9;
    res.projected_B_max = B_max_proj;

    std::cout << "  [" << res.tag << "] n=" << inst.n << " m=" << inst.m << " N=" << inst.N
              << "  b_max=" << b_max << "  B_max(scale=" << SCALE_FOR_PROJECTION << ")=" << B_max_proj
              << "  table_dp_proj=" << std::fixed << std::setprecision(3) << res.projected_dp_gb << " Go"
              << (run_dp ? "" : "  [dp SKIPPÉE --skip-dp]") << "\n"
              << std::flush;

    MKPCLDProblem prob_bnb(inst.c, inst.p, inst.A, inst.b, epsilon);
    std::unique_ptr<MKPCLDProblemDP> prob_dp;
    if (run_dp) prob_dp = std::make_unique<MKPCLDProblemDP>(inst.c, inst.p, inst.A, inst.b, epsilon);

    using ZetaFn = std::function<std::pair<double, Vec1d>(int, int, const Vec1d&)>;
    std::vector<ZetaFn> solvers = {
        [&](int i, int j, const Vec1d& c) { return prob_bnb.solve_zeta(i, j, c); },
        [&](int i, int j, const Vec1d& c) { return run_dp ? prob_dp->solve_zeta(i, j, c) : std::pair<double, Vec1d>{0.0, Vec1d{}}; },
        [&](int i, int j, const Vec1d& c) { return solve_zeta_i_exact_mitm(c, inst.A[i][j], inst.b[i][j]); },
    };

    // λ=0 (cold start) ⇒ cost_{i,j} = c pour tout i,j (v1cld_base.hpp:27).
    const Vec1d& cost = inst.c;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < inst.N; ++i) {
        for (int j = 0; j < inst.m; ++j) {
            std::vector<double> vals(N_METHODS);
            auto t_prev = std::chrono::high_resolution_clock::now();
            for (int mi = 0; mi < N_METHODS; ++mi) {
                if (mi == 1 && !run_dp) { vals[mi] = vals[PROD_METHOD]; continue; } // dp skippée : n'alimente ni le temps ni l'écart
                auto z_r = solvers[mi](i, j, cost);
                auto t_now = std::chrono::high_resolution_clock::now();
                res.zeta[mi].add(std::chrono::duration<double>(t_now - t_prev).count());
                vals[mi] = z_r.first;
                t_prev = t_now;
            }
            for (int mi = 0; mi < N_METHODS; ++mi)
                if (mi != PROD_METHOD && !(mi == 1 && !run_dp))
                    res.max_abs_diff[mi] = std::max(res.max_abs_diff[mi],
                        std::abs(vals[mi] - vals[PROD_METHOD]));
            ++res.n_calls;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    res.t_loop_total_s = std::chrono::duration<double>(t1 - t0).count();
    return res;
}

static void print_instance_row(const InstanceResult& r)
{
    std::cout << std::left << std::setw(46) << r.tag.substr(r.tag.find_last_of('/') + 1)
              << std::right << std::setw(4) << r.n
              << std::setw(4) << r.m
              << std::setw(5) << r.N
              << std::fixed << std::setprecision(3)
              << std::setw(9) << r.t_loop_total_s;
    for (int mi = 0; mi < N_METHODS; ++mi)
        std::cout << std::setw(12) << r.zeta[mi].mean() * 1000.0;
    for (int mi = PROD_METHOD + 1; mi < N_METHODS; ++mi) {
        double su = (r.zeta[mi].mean() > 1e-15) ? r.zeta[PROD_METHOD].mean() / r.zeta[mi].mean() : 0.0;
        std::cout << std::setprecision(2) << std::setw(9) << su;
    }
    double worst_diff = *std::max_element(r.max_abs_diff.begin(), r.max_abs_diff.end());
    std::cout << std::scientific << std::setprecision(1) << std::setw(10) << worst_diff
              << std::fixed << "\n";
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: bench_v1cld_subproblems <manifest.txt> [epsilon=0.10] [--skip-dp]\n";
        return 1;
    }
    std::string manifest_path = argv[1];
    double      epsilon       = (argc > 2 && std::string(argv[2]) != "--skip-dp") ? std::atof(argv[2]) : 0.10;
    bool        run_dp        = true;
    for (int a = 2; a < argc; ++a)
        if (std::string(argv[a]) == "--skip-dp") run_dp = false;

    std::vector<std::pair<std::string, int>> instances;
    {
        std::ifstream f(manifest_path);
        if (!f) { std::cerr << "Manifest introuvable : " << manifest_path << "\n"; return 1; }
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            auto comma = line.find_last_of(',');
            if (comma == std::string::npos) continue;
            std::string path = line.substr(0, comma);
            int rep = std::atoi(line.substr(comma + 1).c_str());
            instances.emplace_back(path, rep);
        }
    }
    if (instances.empty()) { std::cerr << "Manifest vide : " << manifest_path << "\n"; return 1; }

    std::cout << "=== bench_v1cld_subproblems : " << instances.size()
              << " instances, epsilon=" << epsilon << " ===\n";
    std::cout << "ζ_{i,j} chronométré avec 3 méthodes : bnb (B&B 1D, PRODUCTION actuelle) / "
                 "dp (DP O(n×B)) / mitm (meet-in-the-middle O(2^(n/2))).\n\n";

    std::cout << std::left << std::setw(46) << "instance"
              << std::right << std::setw(4) << "n"
              << std::setw(4) << "m"
              << std::setw(5) << "N"
              << std::setw(9) << "t_tot(s)";
    for (const auto& nm : METHOD_NAMES) std::cout << std::setw(12) << (nm + "(ms)");
    for (int mi = PROD_METHOD + 1; mi < N_METHODS; ++mi) std::cout << std::setw(9) << (METHOD_NAMES[mi] + ".x");
    std::cout << std::setw(10) << "max|diff|" << "\n";
    std::cout << std::string(46 + 4 + 4 + 5 + 9 + 12 * N_METHODS + 9 * (N_METHODS - PROD_METHOD - 1) + 10, '-')
              << "\n" << std::flush;

    std::vector<Stats> global_zeta(N_METHODS);
    std::vector<double> worst_max_abs_diff(N_METHODS, 0.0);
    long n_calls_total = 0;

    for (auto& [path, rep] : instances) {
        InstanceResult r = bench_one_instance(path, rep, epsilon, run_dp);
        print_instance_row(r);
        std::cout << std::flush;

        for (int mi = 0; mi < N_METHODS; ++mi) {
            global_zeta[mi].n += r.zeta[mi].n;
            global_zeta[mi].sum += r.zeta[mi].sum;
            global_zeta[mi].sumsq += r.zeta[mi].sumsq;
            global_zeta[mi].min_v = std::min(global_zeta[mi].min_v, r.zeta[mi].min_v);
            global_zeta[mi].max_v = std::max(global_zeta[mi].max_v, r.zeta[mi].max_v);
            worst_max_abs_diff[mi] = std::max(worst_max_abs_diff[mi], r.max_abs_diff[mi]);
        }
        n_calls_total += r.n_calls;
    }

    std::cout << "\n=== Agrégat (" << instances.size() << " instances, "
              << n_calls_total << " paires (i,j) résolues par méthode) ===\n";
    std::cout << std::fixed << std::setprecision(4);
    for (int mi = 0; mi < N_METHODS; ++mi)
        std::cout << "ζ_" << std::left << std::setw(4) << METHOD_NAMES[mi] << std::right
                  << " (pooled sur " << global_zeta[mi].n << " appels) : moyenne="
                  << global_zeta[mi].mean() * 1000.0 << " ms  écart-type=" << global_zeta[mi].stdev() * 1000.0
                  << " ms  min=" << global_zeta[mi].min_v * 1000.0 << " ms  max=" << global_zeta[mi].max_v * 1000.0 << " ms\n";

    std::cout << std::setprecision(2);
    for (int mi = PROD_METHOD + 1; mi < N_METHODS; ++mi) {
        double su_pooled = (global_zeta[mi].mean() > 1e-15)
            ? global_zeta[PROD_METHOD].mean() / global_zeta[mi].mean() : 0.0;
        std::cout << "\n>>> Accélération " << METHOD_NAMES[mi] << " vs bnb (référence production) — pooled : x" << su_pooled << "\n";
    }

    std::cout << std::scientific << std::setprecision(3);
    std::cout << "\nVérification croisée (chaque méthode doit trouver la même valeur ζ_{i,j} que bnb) :\n";
    for (int mi = 0; mi < N_METHODS; ++mi) {
        if (mi == PROD_METHOD) continue;
        std::cout << "  " << METHOD_NAMES[mi] << " vs bnb : max|diff| = " << worst_max_abs_diff[mi]
                  << (worst_max_abs_diff[mi] > 1e-6 ? "  [ATTENTION : écart > 1e-6 — la comparaison de vitesse n'a pas de sens tant que ceci n'est pas résolu]" : "  [OK]") << "\n";
    }
    std::cout << std::fixed << std::setprecision(4);

    return 0;
}
