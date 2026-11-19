/*
 * ccpo_lagrangian_bind.cpp — Interface C pour evaluate_ccpo_lagrangian, sur
 * le patron de solver_bab/mkp/bnb_solver.cpp (CCMKP) : aucune dépendance externe (pas
 * de pybind11, pas de Gecode), fonction extern "C" à pointeurs bruts,
 * appelée depuis Python via ctypes.
 *
 * Compilation : cible CMake ccpo_lagrangian (solver_generic/CMakeLists.txt), avec OpenMP
 * (evaluate_ccpo_lagrangian_omp_c).
 *
 * Ce binding expose evaluate_ccpo_lagrangian (ccpo_sld.hpp) : L(λ) et son
 * sous-gradient pour un λ DONNÉ, valide pour N'IMPORTE QUEL λ (dualité
 * faible) -- c'est la brique destinée à l'entraînement d'un GNN (cf.
 * train_ccpo_gnn.py) : le forward PyTorch appelle cette
 * fonction, le backward utilise directement subgrad_out (théorème de
 * l'enveloppe, pas de différentiation du code combinatoire lui-même).
 */

#include "ccpo_sld.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <omp.h>

extern "C" {

/*
 * ccpo_omp_set_num_threads — force le nombre de threads OpenMP pour CE
 * processus. NECESSAIRE quand ce .so est chargé (via ctypes) dans un
 * processus Python qui importe aussi torch : torch.set_num_threads(k)
 * appelle omp_set_num_threads(k) au niveau de libgomp, une bibliothèque
 * PARTAGÉE par tout le processus -- ça écrase silencieusement OMP_NUM_THREADS
 * pour CE .so aussi (constaté : speedup exactement 1.00x malgré
 * OMP_NUM_THREADS=32, à cause de train_ccpo_gnn.py qui fait
 * torch.set_num_threads(1) à l'import). Appeler ceci APRÈS l'import de torch,
 * AVANT tout appel à evaluate_ccpo_lagrangian_omp_c, pour reprendre la main.
 */
void ccpo_omp_set_num_threads(int n) {
    omp_set_num_threads(n);
}

/*
 * ccpo_omp_diag — DIAGNOSTIC : imprime sur stderr le nombre de
 * threads effectivement vus par CE .so, AVANT et PENDANT une region
 * parallele triviale -- isole si le probleme vient de omp_set_num_threads
 * lui-meme ou d'autre chose (schedule(dynamic) sur un update trop fin,
 * exception silencieuse, etc.).
 */
void ccpo_omp_diag() {
    fprintf(stderr, "[ccpo_omp_diag] omp_get_max_threads()=%d\n", omp_get_max_threads());
    #pragma omp parallel
    {
        #pragma omp critical
        fprintf(stderr, "[ccpo_omp_diag] thread %d/%d\n", omp_get_thread_num(), omp_get_num_threads());
    }
}

/*
 * evaluate_ccpo_lagrangian_c — évalue L(λ) et son sous-gradient.
 *
 * Paramètres (tous C-contiguous, row-major) :
 *   n, N, k, Q, L, U, min_feas : dimensions/paramètres scalaires
 *   cost_base            : (n,) float64            -- coût de base, SANS λ
 *   Sigma                : (N, n, n) float64        -- covariance dense/scénario
 *   B_vec                : (N,) float64             -- budget de risque résiduel/scénario
 *   infeasible_scenario  : (N,) uint8 (0/1)         -- presolve (cf. ccpo_sld.hpp)
 *   lambda               : (N, n) float64           -- multiplicateurs (entrée)
 *   order                : (n,) int32               -- ordre de branchement pour zeta_solve
 *                           (précalculé côté appelant -- ne dépend que de
 *                           cost_base, évite de le retrier à chaque appel
 *                           dans une boucle d'entraînement)
 *   R_ext                : (N, n) float64            -- correction linéaire externe par
 *                           scénario (terme croisé fixé×libre, cf. ccpo_sld.hpp
 *                           "CORRECTIF -- terme croisé") -- (N,n) de zéros pour un
 *                           sous-problème RACINE (aucune variable fixée par
 *                           l'arbre englobant).
 *
 * Sorties :
 *   subgrad_out : (N, n) float64, pré-alloué par l'appelant -- ∂L/∂λ_j[i]
 *   L_out       : (1,) float64 -- L(λ). NaN si feasible_out=0.
 *   feasible_out: (1,) uint8 -- 0 ssi le sous-problème est prouvé infaisable
 *                 (cf. SLDResult::feasible / LagrangianEval::feasible) ;
 *                 dans ce cas subgrad_out N'EST PAS écrit -- l'appelant
 *                 Python doit vérifier ce flag avant d'utiliser subgrad_out.
 */
void evaluate_ccpo_lagrangian_c(
    int n, int N, int k, int Q, int L, int U, int min_feas,
    const double* cost_base,
    const double* Sigma,
    const double* B_vec,
    const uint8_t* infeasible_scenario,
    const double* lambda,
    const int* order,
    const double* R_ext,
    double* subgrad_out,
    double* L_out,
    uint8_t* feasible_out)
{
    std::vector<double> cost_base_v(cost_base, cost_base + n);

    std::vector<std::vector<std::vector<double>>> Sigma_v(
        N, std::vector<std::vector<double>>(n, std::vector<double>(n)));
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < n; ++i)
            std::memcpy(Sigma_v[j][i].data(), Sigma + (static_cast<size_t>(j) * n + i) * n,
                        sizeof(double) * n);

    std::vector<double> B_vec_v(B_vec, B_vec + N);

    std::vector<bool> infeasible_v(N);
    for (int j = 0; j < N; ++j) infeasible_v[j] = (infeasible_scenario[j] != 0);

    std::vector<std::vector<double>> lambda_v(N, std::vector<double>(n));
    for (int j = 0; j < N; ++j)
        std::memcpy(lambda_v[j].data(), lambda + static_cast<size_t>(j) * n, sizeof(double) * n);

    std::vector<int> order_v(order, order + n);

    std::vector<std::vector<double>> R_ext_v(N, std::vector<double>(n));
    for (int j = 0; j < N; ++j)
        std::memcpy(R_ext_v[j].data(), R_ext + static_cast<size_t>(j) * n, sizeof(double) * n);

    LagrangianEval ev = evaluate_ccpo_lagrangian(
        n, N, k, Q, L, U, cost_base_v, Sigma_v, B_vec_v, min_feas,
        infeasible_v, lambda_v, order_v, R_ext_v);

    feasible_out[0] = ev.feasible ? 1 : 0;
    L_out[0] = ev.L;
    if (ev.feasible) {
        for (int j = 0; j < N; ++j)
            std::memcpy(subgrad_out + static_cast<size_t>(j) * n, ev.subgrad[j].data(),
                        sizeof(double) * n);
    }
}

/*
 * solve_ccpo_sld_c — sous-gradient SLD complet (T itérations), pour calculer
 * les cibles λ* du préchauffage supervisé (sur le patron du
 * warmup_on_roots de train_scaled_sd_edit.py côté MKP : verify_sld(g,T) y
 * calcule un λ* de référence par sous-gradient AVANT l'entraînement par la
 * perte Lagrangienne elle-même).
 *
 * Mêmes paramètres d'entrée que evaluate_ccpo_lagrangian_c, plus T (nombre
 * d'itérations), rel_tol/tol_window (mêmes défauts que solve_ccpo_sld) et
 * lambda_init (N,n) -- 0 pour cold start. Sortie : lambda_star (N,n),
 * L_star (1,), feasible_out (1,).
 */
/*
 * evaluate_ccpo_lagrangian_omp_c — IDENTIQUE à evaluate_ccpo_lagrangian_c
 * (mêmes paramètres, même contrat), mais appelle evaluate_ccpo_lagrangian_omp
 * (boucle scénario parallélisée OpenMP, cf. ccpo_sld.hpp) -- pour tester
 * correction (comparaison bit-à-bit du résultat avec la version séquentielle)
 * et gain de temps.
 */
void evaluate_ccpo_lagrangian_omp_c(
    int n, int N, int k, int Q, int L, int U, int min_feas,
    const double* cost_base,
    const double* Sigma,
    const double* B_vec,
    const uint8_t* infeasible_scenario,
    const double* lambda,
    const int* order,
    const double* R_ext,
    double* subgrad_out,
    double* L_out,
    uint8_t* feasible_out)
{
    std::vector<double> cost_base_v(cost_base, cost_base + n);

    std::vector<std::vector<std::vector<double>>> Sigma_v(
        N, std::vector<std::vector<double>>(n, std::vector<double>(n)));
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < n; ++i)
            std::memcpy(Sigma_v[j][i].data(), Sigma + (static_cast<size_t>(j) * n + i) * n,
                        sizeof(double) * n);

    std::vector<double> B_vec_v(B_vec, B_vec + N);

    std::vector<bool> infeasible_v(N);
    for (int j = 0; j < N; ++j) infeasible_v[j] = (infeasible_scenario[j] != 0);

    std::vector<std::vector<double>> lambda_v(N, std::vector<double>(n));
    for (int j = 0; j < N; ++j)
        std::memcpy(lambda_v[j].data(), lambda + static_cast<size_t>(j) * n, sizeof(double) * n);

    std::vector<int> order_v(order, order + n);

    std::vector<std::vector<double>> R_ext_v(N, std::vector<double>(n));
    for (int j = 0; j < N; ++j)
        std::memcpy(R_ext_v[j].data(), R_ext + static_cast<size_t>(j) * n, sizeof(double) * n);

    LagrangianEval ev = evaluate_ccpo_lagrangian_omp(
        n, N, k, Q, L, U, cost_base_v, Sigma_v, B_vec_v, min_feas,
        infeasible_v, lambda_v, order_v, R_ext_v);

    feasible_out[0] = ev.feasible ? 1 : 0;
    L_out[0] = ev.L;
    if (ev.feasible) {
        for (int j = 0; j < N; ++j)
            std::memcpy(subgrad_out + static_cast<size_t>(j) * n, ev.subgrad[j].data(),
                        sizeof(double) * n);
    }
}

/*
 * ccpo_zeta_timing_diag_c — DIAGNOSTIC : temps INDIVIDUEL de
 * chaque zeta_solve(j), j=0..N-1 (sequentiel, chronometre un par un) --
 * teste l'hypothese "cout domine par quelques scenarios 'durs' (stragglers),
 * pas uniforme sur les N scenarios". Memes parametres que
 * evaluate_ccpo_lagrangian_c, sortie : t_ms_out (N,) -- 0.0 si
 * infeasible_scenario[j] (zeta_solve jamais appele dessus, cf. ccpo_sld.hpp).
 */
void ccpo_zeta_timing_diag_c(
    int n, int N, int k, int Q, int L, int U,
    const double* cost_base,
    const double* Sigma,
    const double* B_vec,
    const uint8_t* infeasible_scenario,
    const double* lambda,
    const int* order,
    const double* R_ext,
    double* t_ms_out)
{
    std::vector<double> cost_base_v(cost_base, cost_base + n);
    std::vector<std::vector<std::vector<double>>> Sigma_v(
        N, std::vector<std::vector<double>>(n, std::vector<double>(n)));
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < n; ++i)
            std::memcpy(Sigma_v[j][i].data(), Sigma + (static_cast<size_t>(j) * n + i) * n,
                        sizeof(double) * n);
    std::vector<double> B_vec_v(B_vec, B_vec + N);
    std::vector<bool> infeasible_v(N);
    for (int j = 0; j < N; ++j) infeasible_v[j] = (infeasible_scenario[j] != 0);
    std::vector<std::vector<double>> lambda_v(N, std::vector<double>(n));
    for (int j = 0; j < N; ++j)
        std::memcpy(lambda_v[j].data(), lambda + static_cast<size_t>(j) * n, sizeof(double) * n);
    std::vector<int> order_v(order, order + n);
    std::vector<std::vector<double>> R_ext_v(N, std::vector<double>(n));
    for (int j = 0; j < N; ++j)
        std::memcpy(R_ext_v[j].data(), R_ext + static_cast<size_t>(j) * n, sizeof(double) * n);

    const std::vector<double> p(N, 1.0 / N);
    std::vector<double> Lambda_sum(n, 0.0);
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < n; ++i) Lambda_sum[i] += lambda_v[j][i];

    for (int j = 0; j < N; ++j) {
        if (infeasible_v[j]) { t_ms_out[j] = 0.0; continue; }
        std::vector<double> cost_j(n);
        for (int i = 0; i < n; ++i)
            cost_j[i] = cost_base_v[i] + lambda_v[j][i] / p[j] - Lambda_sum[i];
        auto t0 = std::chrono::steady_clock::now();
        ZetaResult zr = zeta_solve(n, Q, k, L, U, cost_j, Sigma_v[j], B_vec_v[j], order_v, R_ext_v[j]);
        auto t1 = std::chrono::steady_clock::now();
        (void)zr;
        t_ms_out[j] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
}

/*
 * ccpo_zeta_omp_timeline_diag_c — DIAGNOSTIC : IDENTIQUE à
 * ccpo_zeta_timing_diag_c, mais la boucle EST parallélisée OpenMP (comme
 * evaluate_ccpo_lagrangian_omp), et chaque itération imprime sur stderr son
 * thread id + heure de début/fin (omp_get_wtime(), horloge murale absolue)
 * -- permet de voir directement si les threads se recouvrent dans le temps
 * (vrai parallélisme) ou s'exécutent en fait en série malgré le thread
 * count correct (contention malloc ? autre verrou global ?).
 */
void ccpo_zeta_omp_timeline_diag_c(
    int n, int N, int k, int Q, int L, int U,
    const double* cost_base,
    const double* Sigma,
    const double* B_vec,
    const uint8_t* infeasible_scenario,
    const double* lambda,
    const int* order,
    const double* R_ext)
{
    std::vector<double> cost_base_v(cost_base, cost_base + n);
    std::vector<std::vector<std::vector<double>>> Sigma_v(
        N, std::vector<std::vector<double>>(n, std::vector<double>(n)));
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < n; ++i)
            std::memcpy(Sigma_v[j][i].data(), Sigma + (static_cast<size_t>(j) * n + i) * n,
                        sizeof(double) * n);
    std::vector<double> B_vec_v(B_vec, B_vec + N);
    std::vector<bool> infeasible_v(N);
    for (int j = 0; j < N; ++j) infeasible_v[j] = (infeasible_scenario[j] != 0);
    std::vector<std::vector<double>> lambda_v(N, std::vector<double>(n));
    for (int j = 0; j < N; ++j)
        std::memcpy(lambda_v[j].data(), lambda + static_cast<size_t>(j) * n, sizeof(double) * n);
    std::vector<int> order_v(order, order + n);
    std::vector<std::vector<double>> R_ext_v(N, std::vector<double>(n));
    for (int j = 0; j < N; ++j)
        std::memcpy(R_ext_v[j].data(), R_ext + static_cast<size_t>(j) * n, sizeof(double) * n);

    const std::vector<double> p(N, 1.0 / N);
    std::vector<double> Lambda_sum(n, 0.0);
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < n; ++i) Lambda_sum[i] += lambda_v[j][i];

    const double t_ref = omp_get_wtime();
    fprintf(stderr, "[timeline] omp_get_max_threads()=%d  N=%d  t_ref=0.0\n", omp_get_max_threads(), N);

    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < N; ++j) {
        if (infeasible_v[j]) continue;
        std::vector<double> cost_j(n);
        for (int i = 0; i < n; ++i)
            cost_j[i] = cost_base_v[i] + lambda_v[j][i] / p[j] - Lambda_sum[i];
        const double t0 = omp_get_wtime() - t_ref;
        ZetaResult zr = zeta_solve(n, Q, k, L, U, cost_j, Sigma_v[j], B_vec_v[j], order_v, R_ext_v[j]);
        const double t1 = omp_get_wtime() - t_ref;
        (void)zr;
        fprintf(stderr, "[timeline] j=%3d thread=%2d/%2d  t0=%8.3f  t1=%8.3f  dt=%7.3f\n",
                j, omp_get_thread_num(), omp_get_num_threads(), t0, t1, t1 - t0);
    }
}

/*
 * ccpo_zeta_order_diag_c — DIAGNOSTIC : pourquoi zeta_solve est-il bien plus
 * lent avec lambda_GNN qu'avec lambda=0 ? Hypothese : l'ORDRE DE BRANCHEMENT `order` est
 * calcule UNE fois sur cost_base (cf. evaluate_ccpo_lagrangian) alors que zeta_solve
 * optimise cost_j = cost_base + lambda_j/p_j - Sigma_lambda, different par scenario des
 * que lambda != 0 : le premier plongeon (valeur haute d'abord, dans cet ordre) donne alors
 * un mauvais incumbent -> elagage tardif. mode=0 : `order` fourni ; mode=1 : ordre propre a
 * chaque scenario, argsort(-cost_j). zeta_solve est EXACT quel que soit l'ordre : la valeur
 * ζ_j doit etre identique (verifie cote Python). Sorties (N,) : temps ms, noeuds explores,
 * valeur ζ, faisabilite.
 */
void ccpo_zeta_order_diag_c(
    int n, int N, int k, int Q, int L, int U, int mode,
    const double* cost_base, const double* Sigma, const double* B_vec,
    const uint8_t* infeasible_scenario, const double* lambda, const int* order,
    const double* R_ext,
    double* t_ms_out, int64_t* nodes_out, double* value_out, uint8_t* feas_out)
{
    std::vector<double> cost_base_v(cost_base, cost_base + n);
    std::vector<std::vector<std::vector<double>>> Sigma_v(
        N, std::vector<std::vector<double>>(n, std::vector<double>(n)));
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < n; ++i)
            std::memcpy(Sigma_v[j][i].data(), Sigma + (static_cast<size_t>(j) * n + i) * n, sizeof(double) * n);
    std::vector<std::vector<double>> lambda_v(N, std::vector<double>(n)), R_ext_v(N, std::vector<double>(n));
    for (int j = 0; j < N; ++j) {
        std::memcpy(lambda_v[j].data(), lambda + static_cast<size_t>(j) * n, sizeof(double) * n);
        std::memcpy(R_ext_v[j].data(), R_ext + static_cast<size_t>(j) * n, sizeof(double) * n);
    }
    std::vector<int> order_v(order, order + n);
    const double pj = 1.0 / N;
    std::vector<double> Lambda_sum(n, 0.0);
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < n; ++i) Lambda_sum[i] += lambda_v[j][i];

    for (int j = 0; j < N; ++j) {
        t_ms_out[j] = 0.0; nodes_out[j] = 0; value_out[j] = 0.0; feas_out[j] = 0;
        if (infeasible_scenario[j]) continue;
        std::vector<double> cost_j(n);
        for (int i = 0; i < n; ++i) cost_j[i] = cost_base_v[i] + lambda_v[j][i] / pj - Lambda_sum[i];
        std::vector<int> ord = order_v;
        if (mode == 1)
            std::sort(ord.begin(), ord.end(), [&](int a, int b) { return cost_j[a] > cost_j[b]; });
        auto t0 = std::chrono::steady_clock::now();
        ZetaResult zr = zeta_solve(n, Q, k, L, U, cost_j, Sigma_v[j], B_vec[j], ord, R_ext_v[j]);
        auto t1 = std::chrono::steady_clock::now();
        t_ms_out[j] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        nodes_out[j] = zr.nodes_explored;
        value_out[j] = zr.value;
        feas_out[j] = zr.feasible ? 1 : 0;
    }
}

void ccpo_zeta_warm_diag_c(
    int n, int N, int k, int Q, int L, int U, int warm_mode,
    const double* cost_base, const double* Sigma, const double* B_vec,
    const uint8_t* infeasible_scenario, const double* lambda, const int* order,
    const double* R_ext,
    double* t_ms_out, int64_t* nodes_out, double* value_out, uint8_t* feas_out, double* warm_out)
{
    std::vector<double> cost_base_v(cost_base, cost_base + n);
    std::vector<std::vector<std::vector<double>>> Sigma_v(
        N, std::vector<std::vector<double>>(n, std::vector<double>(n)));
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < n; ++i)
            std::memcpy(Sigma_v[j][i].data(), Sigma + (static_cast<size_t>(j) * n + i) * n, sizeof(double) * n);
    std::vector<std::vector<double>> lambda_v(N, std::vector<double>(n)), R_ext_v(N, std::vector<double>(n));
    for (int j = 0; j < N; ++j) {
        std::memcpy(lambda_v[j].data(), lambda + static_cast<size_t>(j) * n, sizeof(double) * n);
        std::memcpy(R_ext_v[j].data(), R_ext + static_cast<size_t>(j) * n, sizeof(double) * n);
    }
    std::vector<int> order_v(order, order + n);
    const double pj = 1.0 / N;
    std::vector<double> Lambda_sum(n, 0.0);
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < n; ++i) Lambda_sum[i] += lambda_v[j][i];

    for (int j = 0; j < N; ++j) {
        t_ms_out[j] = 0.0; nodes_out[j] = 0; value_out[j] = 0.0; feas_out[j] = 0; warm_out[j] = std::nan("");
        if (infeasible_scenario[j]) continue;
        std::vector<double> cost_j(n);
        for (int i = 0; i < n; ++i) cost_j[i] = cost_base_v[i] + lambda_v[j][i] / pj - Lambda_sum[i];
        const std::vector<int>& ord = order_v;
        auto t0 = std::chrono::steady_clock::now();
        ZetaResult zr = zeta_solve(n, Q, k, L, U, cost_j, Sigma_v[j], B_vec[j], ord, R_ext_v[j], warm_mode);
        auto t1 = std::chrono::steady_clock::now();
        t_ms_out[j] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        nodes_out[j] = zr.nodes_explored;
        value_out[j] = zr.value;
        feas_out[j] = zr.feasible ? 1 : 0;
        warm_out[j] = zr.warm_value;
    }
}

/*
 * ccpo_zeta_branch_diag_c — DIAGNOSTIC : effet de l'ORDRE DE BRANCHEMENT sur zeta_solve (exact
 * dans tous les cas : ζ doit etre identique, verifie cote Python). strategy :
 *   0 naif croissant (0..n-1) | 1 ordre courant (`order`, cost_base desc) | 2 cost_j[i]/sqrt(Sigma_ii) desc
 *   3 Sigma_ii croissant | 4 q* racine (relaxation) decroissant | 5 DYNAMIQUE : argmax q* du noeud, valeurs
 *   par proximite de q* | 6 q* racine statique + valeurs par proximite (isole l'effet de l'ordre des valeurs).
 * Ordres 2,3,4,6 calcules PAR SCENARIO. warm_mode comme zeta_solve (0/1/2).
 */
void ccpo_zeta_branch_diag_c(
    int n, int N, int k, int Q, int L, int U, int strategy, int warm_mode,
    const double* cost_base, const double* Sigma, const double* B_vec,
    const uint8_t* infeasible_scenario, const double* lambda, const int* order,
    const double* R_ext,
    double* t_ms_out, int64_t* nodes_out, double* value_out, uint8_t* feas_out)
{
    std::vector<double> cost_base_v(cost_base, cost_base + n);
    std::vector<std::vector<std::vector<double>>> Sigma_v(
        N, std::vector<std::vector<double>>(n, std::vector<double>(n)));
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < n; ++i)
            std::memcpy(Sigma_v[j][i].data(), Sigma + (static_cast<size_t>(j) * n + i) * n, sizeof(double) * n);
    std::vector<std::vector<double>> lambda_v(N, std::vector<double>(n)), R_ext_v(N, std::vector<double>(n));
    for (int j = 0; j < N; ++j) {
        std::memcpy(lambda_v[j].data(), lambda + static_cast<size_t>(j) * n, sizeof(double) * n);
        std::memcpy(R_ext_v[j].data(), R_ext + static_cast<size_t>(j) * n, sizeof(double) * n);
    }
    std::vector<int> order_v(order, order + n);
    const double pj = 1.0 / N;
    std::vector<double> Lambda_sum(n, 0.0);
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < n; ++i) Lambda_sum[i] += lambda_v[j][i];

    for (int j = 0; j < N; ++j) {
        t_ms_out[j] = 0.0; nodes_out[j] = 0; value_out[j] = 0.0; feas_out[j] = 0;
        if (infeasible_scenario[j]) continue;
        std::vector<double> cost_j(n);
        for (int i = 0; i < n; ++i) cost_j[i] = cost_base_v[i] + lambda_v[j][i] / pj - Lambda_sum[i];
        auto t0 = std::chrono::steady_clock::now();
        std::vector<int> ord = order_v;
        const auto& S = Sigma_v[j];
        if (strategy == 0) { for (int i = 0; i < n; ++i) ord[i] = i; }
        else if (strategy == 2) {
            std::vector<double> key(n);
            for (int i = 0; i < n; ++i) key[i] = cost_j[i] / std::sqrt(std::max(S[i][i], 1e-12));
            std::sort(ord.begin(), ord.end(), [&](int a, int b) { return key[a] > key[b]; });
        } else if (strategy == 3) {
            std::sort(ord.begin(), ord.end(), [&](int a, int b) { return S[a][a] < S[b][b]; });
        } else if (strategy == 4 || strategy == 6) {
            std::vector<double> qs;
            solve_quadratic_relaxation_bound(S, cost_j, R_ext_v[j], static_cast<double>(Q), B_vec[j], &qs);
            std::sort(ord.begin(), ord.end(), [&](int a, int b) { return qs[a] > qs[b]; });
        }
        const int var_sel = (strategy == 5) ? 1 : 0;
        const int val_sel = (strategy >= 5) ? 1 : 0;
        ZetaResult zr = zeta_solve_ex(n, Q, k, L, U, cost_j, S, B_vec[j], ord, R_ext_v[j], warm_mode, var_sel, val_sel,
                                     ccpo_zeta_config().infeas_prune);
        auto t1 = std::chrono::steady_clock::now();
        t_ms_out[j] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        nodes_out[j] = zr.nodes_explored;
        value_out[j] = zr.value;
        feas_out[j] = zr.feasible ? 1 : 0;
    }
}

/* ccpo_zeta_set_config_c / ccpo_zeta_get_config_c — réglage de zeta_solve depuis Python (tests, cf. ccpo_sld.hpp).
 * À appeler hors section parallèle. */
void ccpo_zeta_set_config_c(int warm_mode, int branching)
{
    ccpo_zeta_config().warm_mode = warm_mode;
    ccpo_zeta_config().branching = branching;
}
void ccpo_zeta_set_config2_c(int warm_mode, int branching, int infeas_prune, int inherit_infeas)
{
    ccpo_zeta_config().warm_mode = warm_mode;
    ccpo_zeta_config().branching = branching;
    ccpo_zeta_config().infeas_prune = infeas_prune;
    ccpo_zeta_config().inherit_infeas = inherit_infeas;
}
void ccpo_zeta_get_config_c(int* warm_mode, int* branching)
{
    *warm_mode = ccpo_zeta_config().warm_mode;
    *branching = ccpo_zeta_config().branching;
}

void solve_ccpo_sld_c(
    int n, int N, int k, int Q, int L, int U, int min_feas,
    const double* cost_base,
    const double* Sigma,
    const double* B_vec,
    const uint8_t* infeasible_scenario,
    const double* R_ext,
    const double* lambda_init,
    int T,
    double rel_tol,
    int tol_window,
    double* lambda_star_out,
    double* L_out,
    uint8_t* feasible_out)
{
    std::vector<double> cost_base_v(cost_base, cost_base + n);

    std::vector<std::vector<std::vector<double>>> Sigma_v(
        N, std::vector<std::vector<double>>(n, std::vector<double>(n)));
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < n; ++i)
            std::memcpy(Sigma_v[j][i].data(), Sigma + (static_cast<size_t>(j) * n + i) * n,
                        sizeof(double) * n);

    std::vector<double> B_vec_v(B_vec, B_vec + N);

    std::vector<bool> infeasible_v(N);
    for (int j = 0; j < N; ++j) infeasible_v[j] = (infeasible_scenario[j] != 0);

    std::vector<std::vector<double>> R_ext_v(N, std::vector<double>(n));
    for (int j = 0; j < N; ++j)
        std::memcpy(R_ext_v[j].data(), R_ext + static_cast<size_t>(j) * n, sizeof(double) * n);

    std::vector<std::vector<double>> lambda_v(N, std::vector<double>(n));
    for (int j = 0; j < N; ++j)
        std::memcpy(lambda_v[j].data(), lambda_init + static_cast<size_t>(j) * n, sizeof(double) * n);

    SLDResult res = solve_ccpo_sld(
        n, N, k, Q, L, U, cost_base_v, Sigma_v, B_vec_v, min_feas, infeasible_v,
        T, lambda_v, R_ext_v, rel_tol, tol_window);

    feasible_out[0] = res.feasible ? 1 : 0;
    L_out[0] = res.L_star;
    if (res.feasible) {
        for (int j = 0; j < N; ++j)
            std::memcpy(lambda_star_out + static_cast<size_t>(j) * n, res.lambda_star[j].data(),
                        sizeof(double) * n);
    }
}

}  // extern "C"
