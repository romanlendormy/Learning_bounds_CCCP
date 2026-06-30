/*
 * test_ssp.cpp — Tests unitaires pour SSPShiftProblem.
 *
 * Instance jouet : E=3, T=8, Q=3 états, N=20 scénarios.
 *
 * DFA de test (simplifié, 3 états) :
 *   État 0 : pré-service (O,P,R gardent l'état 0 ; W → état 1)
 *   État 1 : en service  (W → état 1 ; O → état 2 ; transitions P/R supprimées)
 *   État 2 : post-service (O → état 2)
 *   État initial : 0
 *   États acceptants : {2}
 *
 * Cette logique accepte toute séquence du type O* W+ O+ (bloc de W encadré
 * d'au moins un O final). T=8, donc "WWWWOOOO", "OOWWWOOO", etc. sont valides.
 *
 * Tests réalisés (§8 du cahier des charges) :
 *   T1. DÉCOUPLAGE theta  : solve_theta == E × meilleur quart individuel (employés interchangeables ici)
 *   T2. EXACTITUDE zeta   : solve_zeta coïncide avec solve_theta sur un scénario d=0 (couverture triviale)
 *   T3. δ_i ≥ 0 ∀i       : cohérence de signe (θ ≥ ζ car X^i ⊆ domaine de base)
 *   T4. GRAPHE une seule fois : adj_ construit dans le constructeur — aucune reconstruction en boucle
 *   T5. DÉTERMINISME      : mêmes entrées → mêmes objectifs (appels répétés de solve_theta)
 *   T6. POIDS NÉGATIFS    : cost négatif → pas de crash, DP renvoie l'optimum correct
 *   T7. COUVERTURE obligatoire : solve_zeta sur scénario strict (d[t]>0) retourne ζ ≤ θ
 *
 * Compilation et exécution (depuis solver_generic/) :
 *   cmake --build build_cc --target test_ssp_shift
 *   ./build_cc/test_ssp_shift
 */

#include "ssp_problem.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

// ── Couleur terminal (optionnel) ───────────────────────────────────────────────
#define PASS "\033[32m[PASS]\033[0m"
#define FAIL "\033[31m[FAIL]\033[0m"

static int n_pass = 0, n_fail = 0;

static void check(bool cond, const std::string& name) {
    if (cond) { std::cout << PASS " " << name << "\n"; ++n_pass; }
    else       { std::cout << FAIL " " << name << "\n"; ++n_fail; }
}

// ── Construction de l'instance jouet ─────────────────────────────────────────

static SSPShiftProblem make_toy_problem() {
    //  Σ = {O=0, P=1, R=2, W=3}
    //
    //  Transitions du DFA simplifié (3 états) :
    //    (0, O=0, 0)  (0, P=1, 0)  (0, R=2, 0)  (0, W=3, 1)   pré-service
    //    (1, W=3, 1)  (1, O=0, 2)                                en service
    //    (2, O=0, 2)  (2, P=1, 2)  (2, R=2, 2)                  post-service
    //
    //  Acceptant : {2}  — "au moins un O après la dernière heure de travail"
    using T3 = std::tuple<int,int,int>;
    std::vector<T3> transitions = {
        {0, 0, 0}, {0, 1, 0}, {0, 2, 0}, {0, 3, 1},  // pré-service
        {1, 3, 1}, {1, 0, 2},                          // en service
        {2, 0, 2}, {2, 1, 2}, {2, 2, 2},               // post-service
    };
    std::vector<int> accepting = {2};
    int initial = 0;

    const int E = 3, T = 8, N = 20;

    // Salaires croissants par période (heures de pointe en milieu de journée)
    Vec1d salary = {1.0, 1.5, 2.0, 2.5, 2.5, 2.0, 1.5, 1.0};

    // Demandes aléatoires mais reproductibles
    // scénario 0 : d[t] = 0 (pas de couverture requise)
    // scénarios 1..10 : d[t] ∈ {0,1} — au plus 1 employé requis par période
    // scénarios 11..19 : d[t] ∈ {0,1,2} — au plus 2 employés requis par période
    Vec2i demand(N, Vec1i(T, 0));
    // Scénario 0 : tout zéro (couverture triviale)
    // Scénario 1 : exige 1 employé à t=2,3,4
    demand[1][2] = 1; demand[1][3] = 1; demand[1][4] = 1;
    // Scénario 2 : exige 2 employés à t=3,4
    demand[2][3] = 2; demand[2][4] = 2;
    // Scénario 3 : exige 3 employés à t=3 (= E = challenge max)
    demand[3][3] = 3;
    // Remplir les autres avec d∈{0,1}
    for (int i = 4; i < N; ++i)
        for (int t = 2; t < 6; ++t)
            demand[i][t] = (i + t) % 2;

    Vec1d p(N, 1.0 / N);
    const double eps = 0.10;

    return SSPShiftProblem(transitions, accepting, initial,
                           salary, demand, std::move(p), eps, E);
}

// ── Vecteur de coûts constant (coût = −salary pour W, 0 ailleurs) ─────────────

static Vec1d make_base_cost(const SSPShiftProblem& prob) {
    // Récupère le vecteur de base c()
    return prob.c();
}

// ── Helpers ───────────────────────────────────────────────────────────────────

static double solve_theta_val(const SSPShiftProblem& prob, const Vec1d& cost) {
    return prob.solve_theta(cost).first;
}

static double solve_zeta_val(const SSPShiftProblem& prob, int i, const Vec1d& cost) {
    return prob.solve_zeta(i, cost).first;
}

// ── T1 : DÉCOUPLAGE theta ─────────────────────────────────────────────────────
// solve_theta == E × best_individual_shift
// Pour E employés interchangeables (même cost par employé), l'optimum est E fois
// la valeur du meilleur quart individuel.

static void test_theta_decoupling(const SSPShiftProblem& prob) {
    const Vec1d cost = make_base_cost(prob);

    auto [theta_val, theta_x] = prob.solve_theta(cost);

    // Vérifier que x est bien une assignation valide (one-hot par (e,t))
    const int E = prob.n_vars() / (8 * 4);  // E = n_vars / (T * |Σ|)
    bool valid_onehot = true;
    for (int e = 0; e < E; ++e) {
        for (int t = 0; t < 8; ++t) {
            double sum = 0.0;
            for (int s = 0; s < 4; ++s)
                sum += theta_x[e * 8 * 4 + t * 4 + s];
            if (std::abs(sum - 1.0) > 1e-9) { valid_onehot = false; break; }
        }
    }
    check(valid_onehot, "T1a: solve_theta renvoie un x one-hot valide");

    // La valeur doit être ≤ 0 (on maximise −salary, donc optimal ≤ 0 si salary ≥ 0)
    // En fait : on maximise −salary pour W, donc la valeur est −Σ salary[t]*w[e,t] ≤ 0.
    // Mais avec des coûts lagrangiens, elle peut être positive.
    // Pour les coûts de base (−salary, 0 pour O/P/R), optimal = −min_salary × work_hours ≤ 0.
    check(theta_val <= 1e-9,
          "T1b: solve_theta(base cost) ≤ 0 (maximise −salary → non-positif)");

    // Si E=1 et cost ne dépend pas de e, la valeur = E × meilleur individuel
    // Ici les 3 employés ont le même cost (cost est identique par e avec le coût de base)
    // → theta_val = 3 × val_individu
    // Vérifier en reconstruisant manuellement la meilleure séquence individuelle
    // (meilleure : ne travailler qu'aux périodes de moindre coût, i.e. plus de salary)
    // Le DFA force d'aller à l'état 2 (post-service) en terminant → il faut finir par O.
    // Séquences valides candidates : celles qui maximisent −Σ salary[t]*W
    // = minimisent Σ salary[t]*W → travailler le moins possible.
    // Le DFA exige au moins 1 W (transition 0→1), suivi d'au moins 1 O (transition 1→2).
    // Donc au moins 1 heure de travail. Pour T=8, meilleure séquence : O^6 W^1 O^1.
    // La dernière période (t=7) a le plus bas salary (1.0) → travailler là.
    // → val_individu = −1.0 (travailler 1h à salary=1.0)
    // → theta = 3 × (−1.0) = −3.0
    const double expected_individual = -1.0;  // salary[7]=1.0, travailler à t=7
    const double expected_theta_E3   = 3.0 * expected_individual;
    check(std::abs(theta_val - expected_theta_E3) < 1e-6,
          "T1c: solve_theta = E × meilleur quart individuel (−1.0 each, total −3.0)");
}

// ── T2 : EXACTITUDE zeta sur scénario trivial (d=0) ──────────────────────────
// Scénario 0 : d[t,0] = 0 ∀t  →  contrainte de couverture triviale  →  ζ = θ.

static void test_zeta_trivial_coverage(const SSPShiftProblem& prob) {
    const Vec1d cost = make_base_cost(prob);
    double theta = solve_theta_val(prob, cost);
    double zeta0 = solve_zeta_val(prob, 0, cost);

    check(std::abs(theta - zeta0) < 1e-4,
          "T2: solve_zeta(scén. d=0) == solve_theta (couverture triviale)");
}

// ── T3 : δ_i ≥ 0 ∀i ─────────────────────────────────────────────────────────
// X^i ⊆ domaine de base  →  θ_i ≥ ζ_i  →  δ_i = max(0, θ_i − ζ_i) ≥ 0.

static void test_delta_nonneg(const SSPShiftProblem& prob) {
    const Vec1d cost = make_base_cost(prob);
    double theta = solve_theta_val(prob, cost);

    bool all_nonneg = true;
    for (int i = 0; i < prob.N_scen(); ++i) {
        double zeta = solve_zeta_val(prob, i, cost);
        double delta = theta - zeta;  // δ_i = θ_i − ζ_i
        if (delta < -1e-4) {          // tolérance numérique
            std::cerr << "  δ_" << i << " = " << delta << " < 0 !\n";
            all_nonneg = false;
        }
    }
    check(all_nonneg, "T3: δ_i = θ_i − ζ_i ≥ 0 pour tous les scénarios");
}

// ── T4 : GRAPHE construit une seule fois ──────────────────────────────────────
// Le constructeur initialise adj_. Les appels solve_theta/solve_zeta relisent
// adj_ sans le reconstruire. Testé ici par la cohérence de valeurs sur 3 appels.

static void test_graph_immutable(const SSPShiftProblem& prob) {
    const Vec1d cost = make_base_cost(prob);
    double v1 = solve_theta_val(prob, cost);
    double v2 = solve_theta_val(prob, cost);
    double v3 = solve_theta_val(prob, cost);
    check(v1 == v2 && v2 == v3,
          "T4: solve_theta retourne la même valeur sur 3 appels consécutifs (graphe stable)");
}

// ── T5 : DÉTERMINISME ─────────────────────────────────────────────────────────

static void test_determinism(const SSPShiftProblem& prob) {
    Vec1d cost = make_base_cost(prob);
    // Modifier légèrement avec des valeurs pseudo-aléatoires mais fixées
    for (int k = 0; k < (int)cost.size(); ++k)
        cost[k] += 0.01 * (k % 7 - 3);  // offset déterministe

    double v1 = solve_theta_val(prob, cost);
    double v2 = solve_theta_val(prob, cost);
    double z1 = solve_zeta_val(prob, 1, cost);
    double z2 = solve_zeta_val(prob, 1, cost);

    check(v1 == v2 && z1 == z2,
          "T5: résultats déterministes (solve_theta et solve_zeta)");
}

// ── T6 : POIDS NÉGATIFS ───────────────────────────────────────────────────────
// Le cost lagrangien peut être négatif. La DP topologique le gère nativement.

static void test_negative_weights(const SSPShiftProblem& prob) {
    Vec1d cost_neg(prob.n_vars(), -1.0);  // tout à −1

    bool no_crash = true;
    double theta = -1e30, zeta = -1e30;
    try {
        theta = solve_theta_val(prob, cost_neg);
        zeta  = solve_zeta_val(prob, 0, cost_neg);
    } catch (...) {
        no_crash = false;
    }
    check(no_crash, "T6a: poids négatifs — pas de crash");

    // Avec cost=-1 partout, chaque σ contribue −1 par période.
    // Le DFA doit parcourir T=8 symboles. Toute séquence valide de longueur 8
    // a exactement 8 transitions → valeur = 8 × (−1) = −8 par employé,
    // theta = E × (−8) = −24.
    const double expected = prob.n_vars() / 4 * (-8.0) / 3;
    // n_vars/4 = E*T, par employé : −8.  E=3 → total = −24
    check(std::abs(theta - (-24.0)) < 1e-4,
          "T6b: poids négatifs — solve_theta retourne l'optimum correct (−24)");
    check(theta - zeta >= -1e-4,
          "T6c: poids négatifs — δ_i ≥ 0 encore vérifiée");
}

// ── T7 : COUVERTURE OBLIGATOIRE CHANGE LA VALEUR ──────────────────────────────
// Scénario 2 exige 2 employés à t=3,4 → ζ_2 ≤ θ (couverture force des W coûteux).

static void test_coverage_changes_value(const SSPShiftProblem& prob) {
    const Vec1d cost = make_base_cost(prob);
    double theta = solve_theta_val(prob, cost);
    double zeta2 = solve_zeta_val(prob, 2, cost);  // scén. 2 : d[3]=d[4]=2

    // θ optimise librement → travaille peu ; ζ_2 doit couvrir t=3,4 avec 2 employés
    // → ζ_2 ≤ θ (stritement ici car salary[3]=2.5, salary[4]=2.5 > salary[7]=1.0)
    check(theta - zeta2 >= -1e-4,
          "T7a: θ ≥ ζ_2 (couverture rend le sous-problème plus contraint)");

    // La valeur ζ_2 doit être finie (le scénario est faisable avec E=3 employés)
    check(zeta2 > -1e29,
          "T7b: solve_zeta(scén. 2) est faisable (valeur finie)");

    // Vérifier que x_2 satisfait effectivement la couverture
    auto [zeta2_val, x2] = prob.solve_zeta(2, cost);
    int cov3 = 0, cov4 = 0;
    const int T = 8;
    for (int e = 0; e < 3; ++e) {
        cov3 += (int)std::round(x2[e * T * 4 + 3 * 4 + 3]);  // sigma W=3, t=3
        cov4 += (int)std::round(x2[e * T * 4 + 4 * 4 + 3]);  // sigma W=3, t=4
    }
    check(cov3 >= 2 && cov4 >= 2,
          "T7c: x* de solve_zeta(2) satisfait bien la couverture d[3]=d[4]=2");
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== Tests SSPShiftProblem ===\n\n";

    SSPShiftProblem prob = make_toy_problem();

    test_theta_decoupling     (prob);
    test_zeta_trivial_coverage(prob);
    test_delta_nonneg         (prob);
    test_graph_immutable      (prob);
    test_determinism          (prob);
    test_negative_weights     (prob);
    test_coverage_changes_value(prob);

    std::cout << "\n=== Résultat : "
              << n_pass << " PASS, " << n_fail << " FAIL ===\n";

    return (n_fail == 0) ? 0 : 1;
}
