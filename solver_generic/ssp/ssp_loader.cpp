/*
 * ssp_loader.cpp — Implémentation du chargeur d'instances SSP depuis texte.
 */

#include "ssp_loader.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ── Utilitaires de parsing ────────────────────────────────────────────────────

static std::ifstream open_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("SSP loader: impossible d'ouvrir " + path);
    return f;
}

static std::string read_key(std::ifstream& f, const std::string& expected_key) {
    std::string key;
    if (!(f >> key))
        throw std::runtime_error("SSP loader: fin de fichier inattendue (attendu: " + expected_key + ")");
    if (!expected_key.empty() && key != expected_key)
        throw std::runtime_error("SSP loader: clé inattendue '" + key + "' (attendu: " + expected_key + ")");
    return key;
}

// ── Chargement de instance.txt ────────────────────────────────────────────────

static void load_instance_txt(const std::string& path, SSPInstance& inst) {
    auto f = open_file(path);
    std::string key;
    while (f >> key) {
        if      (key == "E")              f >> inst.E;
        else if (key == "T")              f >> inst.T;
        else if (key == "N")              f >> inst.N;
        else if (key == "EPSILON")        f >> inst.epsilon;
        else if (key == "INTERCHANGEABLE") {
            int v; f >> v; inst.interchangeable = (v != 0);
        }
        // Clés inconnues ignorées silencieusement
    }
    if (inst.E <= 0 || inst.T <= 0 || inst.N <= 0)
        throw std::runtime_error("SSP loader: instance.txt invalide (E, T ou N manquants)");
}

// ── Chargement de automaton.txt ───────────────────────────────────────────────

static void load_automaton_txt(const std::string& path, SSPInstance& inst) {
    auto f = open_file(path);
    int num_states = 0, n_acc = 0, n_trans = 0;

    read_key(f, "NUM_STATES");  f >> num_states;
    read_key(f, "INITIAL");     f >> inst.initial;
    read_key(f, "N_ACCEPTING"); f >> n_acc;

    inst.accepting.resize(n_acc);
    for (int i = 0; i < n_acc; ++i)
        f >> inst.accepting[i];

    read_key(f, "N_TRANSITIONS"); f >> n_trans;
    inst.transitions.resize(n_trans);
    for (int i = 0; i < n_trans; ++i) {
        int src, sym, dst;
        f >> src >> sym >> dst;
        inst.transitions[i] = {src, sym, dst};
    }

    if (!f && n_trans > 0)
        throw std::runtime_error("SSP loader: automaton.txt tronqué");
}

// ── Chargement de cost.txt ────────────────────────────────────────────────────

static void load_cost_txt(const std::string& path, SSPInstance& inst) {
    auto f = open_file(path);
    int T = 0;
    read_key(f, "T"); f >> T;
    if (T != inst.T && inst.T > 0)
        throw std::runtime_error("SSP loader: T dans cost.txt (" + std::to_string(T)
                                 + ") != T dans instance.txt (" + std::to_string(inst.T) + ")");
    inst.T = T;

    read_key(f, "SALARY");
    inst.salary.resize(T);
    for (int t = 0; t < T; ++t)
        f >> inst.salary[t];

    if (!f)
        throw std::runtime_error("SSP loader: cost.txt tronqué");
}

// ── Chargement de demand.txt ──────────────────────────────────────────────────

static void load_demand_txt(const std::string& path, SSPInstance& inst) {
    auto f = open_file(path);
    int T = 0, N = 0;
    read_key(f, "T"); f >> T;
    read_key(f, "N"); f >> N;

    if ((inst.T > 0 && T != inst.T) || (inst.N > 0 && N != inst.N))
        throw std::runtime_error("SSP loader: dimensions dans demand.txt incompatibles");
    inst.T = T; inst.N = N;

    // N lignes, chacune avec T entiers : demande du scénario i à la période t
    inst.demand.assign(N, Vec1i(T));
    for (int i = 0; i < N; ++i)
        for (int t = 0; t < T; ++t)
            f >> inst.demand[i][t];

    if (!f)
        throw std::runtime_error("SSP loader: demand.txt tronqué");
}

// ── Interface publique ────────────────────────────────────────────────────────

SSPInstance load_ssp_instance(const std::string& dir) {
    const std::string sep = (dir.empty() || dir.back() == '/') ? "" : "/";
    SSPInstance inst{};

    load_instance_txt (dir + sep + "instance.txt",  inst);
    load_automaton_txt(dir + sep + "automaton.txt", inst);
    load_cost_txt     (dir + sep + "cost.txt",      inst);
    load_demand_txt   (dir + sep + "demand.txt",    inst);

    // Probabilités uniformes par défaut
    inst.p.assign(inst.N, 1.0 / inst.N);

    return inst;
}

SSPShiftProblem make_ssp_problem(const SSPInstance& inst) {
    return SSPShiftProblem(
        inst.transitions,
        inst.accepting,
        inst.initial,
        inst.salary,
        inst.demand,
        inst.p,
        inst.epsilon,
        inst.E
    );
}
