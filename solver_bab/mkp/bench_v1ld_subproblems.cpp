/*
 * bench_v1ld_subproblems.cpp — Benchmark élémentaire du coût de calcul de
 * v1LD (MKP) : UNE seule itération (λ=0, cold start), PAS de Branch and
 * Bound — juste les résolutions de sous-problèmes que fait une itération
 * de solve_v1ld (cf. solver_generic/v1ld_solver.cpp, bloc (B)).
 *
 * ζ_i est chronométré avec TROIS méthodes distinctes, sur les MÊMES cost
 * vectors :
 *   - cpsat   : MKPProblem::solve_zeta            — OR-Tools CP-SAT
 *     (reconstruit un modèle CP à chaque appel). Utilisé nulle part en
 *     production C++ (seulement ce benchmark + binding Python) — sert de
 *     référence "solveur générique" et de vérité de base pour la
 *     vérification croisée (solveur mature, indépendant).
 *   - bnb     : MKPProblemBnB::solve_zeta          — B&B maison Dantzig,
 *     borne = min de m relaxations 1D séparées. MÉTHODE RÉELLEMENT UTILISÉE
 *     par le B&B Gecode en production (solver_bab/mkp/gecode_bab.hpp:766).
 *   - surr    : MKPProblemBnBSurrogate::solve_zeta — même squelette B&B que
 *     bnb, mais borne = relaxation SURROGATE (agrégation pondérée des m
 *     contraintes en une seule) — cf. mkp_surrogate_bound.hpp. "Option A"
 *     de l'exploration accélération ζ : garde le même v1LD, garde le même
 *     moteur de recherche, cherche juste une borne plus serrée.
 *
 * (Deux autres méthodes testées et mises de côté, code conservé mais retiré
 * de la comparaison active ci-dessous :
 *   - solve_zeta_gecode() — espace Gecode natif avec la même borne surrogate
 *     injectée via un Brancher custom, cf. zeta_gecode_solver.cpp — bug de
 *     correction non résolu (solution sous-optimale sur certains scénarios,
 *     cause isolée à la fonction de borne mais pas trouvée) ET déjà plus
 *     lente que bnb avant même ce bug.
 *   - MKPProblemBnBLP::solve_zeta, cf. mkp_problem_bnb_lp.hpp — "Option B" :
 *     même squelette de branchement que bnb/surr, borne = relaxation LP
 *     JOINTE résolue par GLOP (warm-startée nœud à nœud). Correcte (vérifiée
 *     croisée vs cpsat) mais ~44x PLUS LENTE que bnb (pooled, 50 instances,
 *     job SLURM 15985510) — le coût de GLOP::Solve() par nœud, même
 *     warm-starté, domine largement le gain de la borne plus serrée sur le
 *     nombre de nœuds. Piste écartée sans changement d'approche (LP par
 *     dimension au lieu de jointe, ou warm-start différent).)
 *
 * Les 3 valeurs ζ_i doivent coïncider (même problème résolu exactement par
 * 3 algorithmes différents) — l'écart max de chacune vs cpsat (référence)
 * est rapporté comme vérification croisée de correction, pas seulement de
 * vitesse.
 *
 * Pour chaque instance :
 *   - construit les 3 objets MKPProblem / MKPProblemBnB / MKPProblemBnBSurrogate
 *     (racine, aucune variable fixée)
 *   - à λ=0, Λ_sum=0 donc cost_i = c pour tout i=1..N (identique à la
 *     toute première itération de solve_v1ld)
 *   - chronomètre séparément solve_theta(cost) et les 3 solve_zeta(i, cost)
 *   - rapporte : nb sous-problèmes résolus, temps total, moyenne/écart-type
 *     (et min/max) du temps par méthode, + accélération vs bnb (référence
 *     de production)
 *
 * Répété sur un lot d'instances (manifest) pour une moyenne consistante —
 * un résumé agrégé (pooled + inter-instance) est imprimé à la fin.
 *
 * Usage :
 *   bench_v1ld_subproblems <manifest.txt> [epsilon=0.10] [refdir]
 *
 * manifest.txt : une ligne par instance "chemin_dataset.txt,rep" (même
 * format SONG/Pisinger que load_song_local, cf. main_gecode.cpp).
 * Lignes vides ou commençant par '#' ignorées.
 */

#include "mkp_problem_bnb.hpp"
#include "mkp_problem_bnb_surrogate.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

// ── Loader (copie locale — cf. main_gecode.cpp / gnn_native_probe.cpp) ───────
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

// ── Statistiques de flux (mean/var en un passage, pas besoin de stocker) ─────
struct Stats {
    long   n = 0;
    double sum = 0.0, sumsq = 0.0;
    double min_v = std::numeric_limits<double>::infinity(), max_v = 0.0;

    void add(double x) {
        ++n; sum += x; sumsq += x * x;
        min_v = std::min(min_v, x);
        max_v = std::max(max_v, x);
    }
    void merge(const Stats& o) {
        n += o.n; sum += o.sum; sumsq += o.sumsq;
        min_v = std::min(min_v, o.min_v);
        max_v = std::max(max_v, o.max_v);
    }
    double mean() const { return n ? sum / n : 0.0; }
    // Variance de population (n, pas n-1) : nb d'échantillons assez grand
    // (N=100/instance, jusqu'à 5000 au global) pour que le biais de Bessel
    // soit négligeable.
    double var() const {
        if (n < 2) return 0.0;
        double m = mean();
        return std::max(0.0, sumsq / n - m * m);
    }
    double stdev() const { return std::sqrt(var()); }
};

// Nom court + index de référence (0 = cpsat, la vérité de base pour la
// vérification croisée) — dans cet ordre partout (tableaux/CSV/agrégats).
// Indice 1 (bnb) = référence de production pour les accélérations : tout
// mi >= 2 est un candidat "plus rapide que bnb ?" (surr, ...).
static const std::vector<std::string> METHOD_NAMES = {"cpsat", "bnb", "surr"};
constexpr int N_METHODS = 3;
constexpr int REF_METHOD = 0;  // cpsat
constexpr int PROD_METHOD = 1; // bnb — référence des accélérations

struct InstanceResult {
    std::string tag;
    int    n = 0, m = 0, N = 0;
    Stats  theta;
    std::vector<Stats> zeta;              // taille N_METHODS
    std::vector<Vec1d> zeta_values;       // taille N_METHODS, chacun (N,)
    std::vector<double> max_abs_diff;     // taille N_METHODS, vs REF_METHOD (0.0 pour REF_METHOD lui-même)
    double t_loop_total_s = 0.0;
    double theta_value = 0.0;             // identique pour tout i à λ=0
};

static InstanceResult bench_one_instance(const std::string& path, int rep, double epsilon)
{
    MKPInst inst = load_song_local(path, rep);
    MKPProblem             prob_cpsat(inst.c, inst.p, inst.A, inst.b, epsilon);
    MKPProblemBnB          prob_bnb(inst.c, inst.p, inst.A, inst.b, epsilon);
    MKPProblemBnBSurrogate prob_surr(inst.c, inst.p, inst.A, inst.b, epsilon);

    using ZetaFn = std::function<std::pair<double, Vec1d>(int, const Vec1d&)>;
    std::vector<ZetaFn> solvers = {
        [&](int i, const Vec1d& c) { return prob_cpsat.solve_zeta(i, c); },
        [&](int i, const Vec1d& c) { return prob_bnb.solve_zeta(i, c); },
        [&](int i, const Vec1d& c) { return prob_surr.solve_zeta(i, c); },
    };

    InstanceResult res;
    res.tag = path + "#rep" + std::to_string(rep);
    res.n = inst.n; res.m = inst.m; res.N = inst.N;
    res.zeta.assign(N_METHODS, Stats{});
    res.zeta_values.assign(N_METHODS, Vec1d(inst.N));
    res.max_abs_diff.assign(N_METHODS, 0.0);

    // λ=0 (cold start, 1ère itération de solve_v1ld) ⇒ Λ_sum=0 et
    // λ_i/p_i=0 pour tout i ⇒ cost_i = c, IDENTIQUE pour tout i. On la
    // recalcule quand même à chaque i (comme le ferait solve_v1ld) plutôt
    // que de supposer une implémentation future qui pourrait la varier.
    const Vec1d& cost = inst.c;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < inst.N; ++i) {
        auto ta = std::chrono::high_resolution_clock::now();
        auto theta_r = prob_cpsat.solve_theta(cost);   // solve_theta identique (hérité) sur les 3 classes MKPProblem*
        auto tb = std::chrono::high_resolution_clock::now();
        res.theta.add(std::chrono::duration<double>(tb - ta).count());
        res.theta_value = theta_r.first;

        auto t_prev = tb;
        for (int mi = 0; mi < N_METHODS; ++mi) {
            auto z_r = solvers[mi](i, cost);
            auto t_now = std::chrono::high_resolution_clock::now();
            res.zeta[mi].add(std::chrono::duration<double>(t_now - t_prev).count());
            res.zeta_values[mi][i] = z_r.first;
            t_prev = t_now;
        }
        for (int mi = 0; mi < N_METHODS; ++mi)
            if (mi != REF_METHOD)
                res.max_abs_diff[mi] = std::max(res.max_abs_diff[mi],
                    std::abs(res.zeta_values[mi][i] - res.zeta_values[REF_METHOD][i]));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    res.t_loop_total_s = std::chrono::duration<double>(t1 - t0).count();
    return res;
}

static void print_instance_row(const InstanceResult& r)
{
    double sub_sum_s = r.theta.sum;
    for (auto& s : r.zeta) sub_sum_s += s.sum;
    double worst_diff = *std::max_element(r.max_abs_diff.begin(), r.max_abs_diff.end());

    std::cout << std::left << std::setw(46) << r.tag.substr(r.tag.find_last_of('/') + 1)
              << std::right << std::setw(4) << r.n
              << std::setw(4) << r.m
              << std::setw(5) << r.N
              << std::fixed << std::setprecision(3)
              << std::setw(9) << sub_sum_s;
    for (int mi = 0; mi < N_METHODS; ++mi)
        std::cout << std::setw(11) << r.zeta[mi].mean() * 1000.0;
    std::cout << std::setprecision(2);
    for (int mi = PROD_METHOD + 1; mi < N_METHODS; ++mi) {
        double su = (r.zeta[mi].mean() > 1e-15) ? r.zeta[PROD_METHOD].mean() / r.zeta[mi].mean() : 0.0;
        std::cout << std::setw(9) << su;
    }
    std::cout << std::scientific << std::setprecision(1)
              << std::setw(10) << worst_diff
              << std::fixed << "\n";
}

static void print_csv_rows(const InstanceResult& r)
{
    // Format LONG (une ligne par méthode) — plus simple à analyser (pandas/awk)
    // qu'une ligne large par instance.
    // CSV_ROW,tag,n,m,N,method,n_calls,t_total_s,mean_ms,std_ms,min_ms,max_ms,checksum,max_abs_diff_vs_cpsat
    for (int mi = 0; mi < N_METHODS; ++mi) {
        double checksum = 0.0;
        for (double v : r.zeta_values[mi]) checksum += v;
        std::cout << std::fixed << std::setprecision(9)
                  << "CSV_ROW," << r.tag << "," << r.n << "," << r.m << "," << r.N << ","
                  << METHOD_NAMES[mi] << "," << r.zeta[mi].n << ","
                  << r.zeta[mi].sum << ","
                  << r.zeta[mi].mean() * 1000.0 << "," << r.zeta[mi].stdev() * 1000.0 << ","
                  << r.zeta[mi].min_v * 1000.0 << "," << r.zeta[mi].max_v * 1000.0 << ","
                  << std::setprecision(17) << checksum << ","
                  << std::scientific << r.max_abs_diff[mi] << std::fixed << "\n";
    }
}

// Écrit une valeur double à précision totale (17 chiffres significatifs
// round-trip exactement pour un double) — pour une comparaison bit-exacte
// entre deux runs (cf. vérification de reproductibilité) ET pour la
// vérification croisée entre les 3 méthodes (colonnes doivent coïncider à
// tol flottante près).
static void write_reference_file(const std::string& refdir, const InstanceResult& r)
{
    std::string safe_tag = r.tag;
    for (char& c : safe_tag) if (c == '/' || c == '#') c = '_';
    std::ofstream f(refdir + "/" + safe_tag + ".txt");
    f << std::setprecision(17);
    f << "# instance=" << r.tag << " n=" << r.n << " m=" << r.m << " N=" << r.N << "\n";
    f << "# methods_order=";
    for (auto& nm : METHOD_NAMES) f << nm << " ";
    f << "\n# max_abs_diff_vs_cpsat=";
    for (double d : r.max_abs_diff) f << d << " ";
    f << "\n";
    f << "theta_value " << r.theta_value << "\n";
    for (int i = 0; i < r.N; ++i) {
        f << "zeta_value " << i;
        for (int mi = 0; mi < N_METHODS; ++mi) f << " " << r.zeta_values[mi][i];
        f << "\n";
    }
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: bench_v1ld_subproblems <manifest.txt> [epsilon=0.10] [refdir]\n";
        std::cerr << "  refdir : si fourni, écrit un fichier de référence (theta_value + "
                     "N zeta_value pour les 3 méthodes, précision totale) par instance — "
                     "ground truth pour vérifier la reproductibilité / comparer une future "
                     "méthode de calcul de v1LD.\n";
        return 1;
    }
    std::string manifest_path = argv[1];
    double      epsilon       = (argc > 2) ? std::atof(argv[2]) : 0.10;
    std::string refdir        = (argc > 3) ? argv[3] : "";

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

    std::cout << "=== bench_v1ld_subproblems : " << instances.size()
              << " instances, epsilon=" << epsilon << " ===\n";
    std::cout << "Une seule itération v1LD (λ=0, racine, sans B&B) par instance.\n";
    std::cout << "ζ chronométré avec 3 méthodes : cpsat (référence CP-SAT) / bnb (B&B maison, "
                 "PRODUCTION actuelle) / surr (B&B maison, borne surrogate — option A).\n\n";

    std::cout << std::left << std::setw(46) << "instance"
              << std::right << std::setw(4) << "n"
              << std::setw(4) << "m"
              << std::setw(5) << "N"
              << std::setw(9) << "t_sub(s)";
    for (const auto& nm : METHOD_NAMES) std::cout << std::setw(11) << (nm + "(ms)");
    for (int mi = PROD_METHOD + 1; mi < N_METHODS; ++mi) std::cout << std::setw(9) << (METHOD_NAMES[mi] + ".x");
    std::cout << std::setw(10) << "max|diff|"
              << "\n";
    std::cout << std::string(46 + 4 + 4 + 5 + 9 + 11 * N_METHODS + 9 * (N_METHODS - PROD_METHOD - 1) + 10, '-')
              << "\n" << std::flush;

    std::vector<Stats> global_zeta(N_METHODS);
    Stats global_theta;
    Stats per_instance_total_time;
    std::vector<Stats> per_instance_speedup(N_METHODS);  // vs bnb (indice 1), REF_METHOD/bnb non utilisés
    long  n_theta_total = 0, n_zeta_total = 0;
    std::vector<double> worst_max_abs_diff(N_METHODS, 0.0);

    for (auto& [path, rep] : instances) {
        InstanceResult r = bench_one_instance(path, rep, epsilon);
        print_instance_row(r);
        print_csv_rows(r);
        std::cout << std::flush;
        if (!refdir.empty()) write_reference_file(refdir, r);

        global_theta.merge(r.theta);
        n_theta_total += r.theta.n;
        for (int mi = 0; mi < N_METHODS; ++mi) {
            global_zeta[mi].merge(r.zeta[mi]);
            worst_max_abs_diff[mi] = std::max(worst_max_abs_diff[mi], r.max_abs_diff[mi]);
        }
        n_zeta_total += r.zeta[0].n;

        double sub_sum_s = r.theta.sum;
        for (auto& s : r.zeta) sub_sum_s += s.sum;
        per_instance_total_time.add(sub_sum_s);

        // Accélération de chaque méthode candidate (surr, ...) vs bnb (référence de production).
        for (int mi = PROD_METHOD + 1; mi < N_METHODS; ++mi)
            if (r.zeta[mi].mean() > 1e-15)
                per_instance_speedup[mi].add(r.zeta[PROD_METHOD].mean() / r.zeta[mi].mean());
    }

    double grand_total_s = global_theta.sum;
    for (auto& s : global_zeta) grand_total_s += s.sum;

    std::cout << "\n=== Agrégat (" << instances.size() << " instances, "
              << (n_theta_total + N_METHODS * n_zeta_total) << " sous-problèmes résolus : "
              << n_theta_total << " θ + " << N_METHODS << "×" << n_zeta_total << " ζ) ===\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Temps total passé à résoudre : " << grand_total_s << " s\n";
    std::cout << "  θ : " << global_theta.sum << " s";
    for (int mi = 0; mi < N_METHODS; ++mi)
        std::cout << "   |  ζ_" << METHOD_NAMES[mi] << " : " << global_zeta[mi].sum << " s";
    std::cout << "\n";

    std::cout << std::setprecision(4);
    std::cout << "\nθ         (pooled sur " << global_theta.n << " appels) : moyenne="
              << global_theta.mean() * 1000.0 << " ms  écart-type=" << global_theta.stdev() * 1000.0 << " ms\n";
    for (int mi = 0; mi < N_METHODS; ++mi)
        std::cout << "ζ_" << std::left << std::setw(7) << METHOD_NAMES[mi] << std::right
                  << " (pooled sur " << global_zeta[mi].n << " appels) : moyenne="
                  << global_zeta[mi].mean() * 1000.0 << " ms  écart-type=" << global_zeta[mi].stdev() * 1000.0
                  << " ms  min=" << global_zeta[mi].min_v * 1000.0 << " ms  max=" << global_zeta[mi].max_v * 1000.0 << " ms\n";

    std::cout << std::setprecision(2);
    for (int mi = PROD_METHOD + 1; mi < N_METHODS; ++mi) {
        double su_pooled = (global_zeta[mi].mean() > 1e-15)
            ? global_zeta[PROD_METHOD].mean() / global_zeta[mi].mean() : 0.0;
        std::cout << "\n>>> Accélération " << METHOD_NAMES[mi] << " vs bnb (référence production) — pooled : x"
                  << su_pooled
                  << "   |  moyenne inter-instance : x" << per_instance_speedup[mi].mean()
                  << " (écart-type x" << per_instance_speedup[mi].stdev() << ")\n";
    }

    std::cout << std::scientific << std::setprecision(3);
    std::cout << "\nVérification croisée (toutes les méthodes doivent trouver la même valeur ζ_i que cpsat) :\n";
    for (int mi = 0; mi < N_METHODS; ++mi) {
        if (mi == REF_METHOD) continue;
        std::cout << "  " << METHOD_NAMES[mi] << " vs cpsat : max|diff| = " << worst_max_abs_diff[mi]
                  << (worst_max_abs_diff[mi] > 1e-6 ? "  [ATTENTION : écart > 1e-6, à investiguer]" : "  [OK]") << "\n";
    }
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "\nVariabilité inter-instance : temps total/instance : moyenne="
              << per_instance_total_time.mean() << " s  écart-type=" << per_instance_total_time.stdev() << " s\n";

    return 0;
}
