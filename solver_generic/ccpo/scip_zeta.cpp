/*
 * scip_zeta.cpp — hook SCIP pour le mode hybride de zeta_solve (cf. ccpo_sld.hpp::zeta_solve_cfg, CCPOZetaConfig::scip_mode).
 * Compilé SEULEMENT avec SCIP (-DCCPO_WITH_SCIP dans main/test ; lien libscip). Une instance SCIP par thread (thread_local), réutilisée
 * d'un appel à l'autre (SCIPfreeProb entre deux appels) ; validé sur 2176 appels à 1/8/24 threads (scip_zeta_probe : 0 écart).
 * Modèle : max c'q, Σq=Q, L·y ≤ q ≤ U·y, Σy ≤ k, q entier, y binaire, q'Σq + 2r'q ≤ B (objectif/∥c∥∞ et risque/B pour l'échelle),
 * tolérances serrées (feastol 1e-9, gaps 1e-9) ; solution initiale = incumbent du B&B ; q arrondi RE-VÉRIFIÉ exactement par nos soins.
 */
#include "ccpo_sld.hpp"

#include <scip/scip.h>
#include <scip/scipdefplugins.h>

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
// Interrupteurs de DIAGNOSTIC (lus une fois ; défauts = comportement par défaut) pour l'enquête sur les échecs de SCIP :
//   CCPO_SCIP_CLOCK     : 0 = défaut SCIP (horloge CPU du processus) | 1 = CPU | 2 = horloge MUR (timing/clocktype)
//   CCPO_SCIP_FEASTOL   : numerics/feastol (défaut 1e-9)
//   CCPO_SCIP_NLP       : 0 (DEFAUT) = désactive le NLP/Ipopt (nlp/disable) | 1 = comportement SCIP par défaut
//   CCPO_SCIP_HEUR      : 0 (DEFAUT) = désactive toutes les heuristiques (l'incumbent du B&B est déjà fourni) | 1 = défaut SCIP
//   Pourquoi : les échecs de SCIP observés en e2e (jusqu'à 31 % des appels délégués) passaient tous par heur_nlpdiving -> Ipopt (retour -12) ;
//   désactiver NLP+heuristiques supprime ces erreurs et réduit le temps SCIP d'environ 40 % (enquête, sans écart de valeur).
//   CCPO_SCIP_DUMP_PARAMS : chemin ; écrit une fois tous les paramètres SCIP effectifs
struct DiagCfg {
    int clock = 0; double feastol = 1e-9; int nlp = 0; int heur = 0; const char* dump = nullptr;
    DiagCfg() {
        if (const char* e = std::getenv("CCPO_SCIP_CLOCK")) clock = std::atoi(e);
        if (const char* e = std::getenv("CCPO_SCIP_FEASTOL")) feastol = std::atof(e);
        if (const char* e = std::getenv("CCPO_SCIP_NLP")) nlp = std::atoi(e);
        if (const char* e = std::getenv("CCPO_SCIP_HEUR")) heur = std::atoi(e);
        dump = std::getenv("CCPO_SCIP_DUMP_PARAMS");
    }
};
const DiagCfg& diag_cfg() { static DiagCfg d; return d; }

void chk(SCIP_RETCODE r) { if (r != SCIP_OKAY) throw std::runtime_error("SCIP error"); }

struct ScipHolder {
    SCIP* s = nullptr;
    ~ScipHolder() { if (s) SCIPfree(&s); }
    SCIP* get() {
        if (!s) {
            if (SCIPcreate(&s) != SCIP_OKAY) { s = nullptr; return nullptr; }
            if (SCIPincludeDefaultPlugins(s) != SCIP_OKAY) { SCIPfree(&s); s = nullptr; return nullptr; }
            SCIPsetMessagehdlrQuiet(s, TRUE);
        }
        return s;
    }
};
}  // namespace

int ccpo_scip_zeta_solve(int n, int Q, int k, int L, int U, const std::vector<double>& cost,
                         const std::vector<std::vector<double>>& S, double B, const std::vector<double>& r,
                         const std::vector<int>& init_q, double init_val, double tlim,
                         std::vector<int>& q_out, double& primal, double& dual)
{
    (void)init_val;
    static thread_local ScipHolder holder;
    SCIP* scip = holder.get();
    if (!scip) return 3;
    int rc = 3;
    std::vector<SCIP_VAR*> q(n, nullptr), y(n, nullptr);
    std::vector<SCIP_CONS*> cons;
    bool created = false;
    try {
        chk(SCIPcreateProbBasic(scip, "zeta")); created = true;
        chk(SCIPsetObjsense(scip, SCIP_OBJSENSE_MAXIMIZE));
        double cs = 1.0; for (double x : cost) cs = std::max(cs, std::abs(x));
        const double bs = (B > 1.0) ? B : 1.0;
        for (int i = 0; i < n; ++i) {
            chk(SCIPcreateVarBasic(scip, &q[i], ("q" + std::to_string(i)).c_str(), 0.0, U, cost[i] / cs, SCIP_VARTYPE_INTEGER));
            chk(SCIPaddVar(scip, q[i]));
            chk(SCIPcreateVarBasic(scip, &y[i], ("y" + std::to_string(i)).c_str(), 0.0, 1.0, 0.0, SCIP_VARTYPE_BINARY));
            chk(SCIPaddVar(scip, y[i]));
        }
        const double inf = SCIPinfinity(scip);
        auto add_lin = [&](std::vector<SCIP_VAR*> vs, std::vector<double> cf, double lhs, double rhs) {
            SCIP_CONS* c_ = nullptr;
            chk(SCIPcreateConsBasicLinear(scip, &c_, "lin", (int)vs.size(), vs.data(), cf.data(), lhs, rhs));
            chk(SCIPaddCons(scip, c_)); cons.push_back(c_);
        };
        for (int i = 0; i < n; ++i) {
            add_lin({q[i], y[i]}, {1.0, -(double)L}, 0.0, inf);
            add_lin({q[i], y[i]}, {1.0, -(double)U}, -inf, 0.0);
        }
        add_lin(y, std::vector<double>(n, 1.0), -inf, (double)k);
        add_lin(q, std::vector<double>(n, 1.0), (double)Q, (double)Q);
        {
            std::vector<SCIP_VAR*> qa, qb, lv; std::vector<double> qc, lc;
            for (int i = 0; i < n; ++i) {
                for (int j = i; j < n; ++j) {
                    const double v = (i == j ? S[i][i] : 2.0 * S[i][j]) / bs;
                    if (v != 0.0) { qa.push_back(q[i]); qb.push_back(q[j]); qc.push_back(v); }
                }
                if (r[i] != 0.0) { lv.push_back(q[i]); lc.push_back(2.0 * r[i] / bs); }
            }
            SCIP_CONS* c_ = nullptr;
            chk(SCIPcreateConsBasicQuadraticNonlinear(scip, &c_, "risk", (int)lv.size(), lv.data(), lc.data(), (int)qa.size(),
                                                      qa.data(), qb.data(), qc.data(), -inf, B / bs));
            chk(SCIPaddCons(scip, c_)); cons.push_back(c_);
        }
        if ((int)init_q.size() == n) {          // solution initiale = incumbent du B&B (SCIP n'a plus qu'à prouver l'optimalité)
            // SCIPtrySolFree plante (SIGSEGV, SCIP 9.0) a l'etape PROBLEM : on transforme d'abord le probleme (etape TRANSFORMED).
            chk(SCIPtransformProb(scip));
            SCIP_SOL* sol = nullptr; SCIP_Bool stored = FALSE;
            chk(SCIPcreateSol(scip, &sol, nullptr));
            for (int i = 0; i < n; ++i) {
                chk(SCIPsetSolVal(scip, sol, q[i], (double)init_q[i]));
                chk(SCIPsetSolVal(scip, sol, y[i], init_q[i] > 0 ? 1.0 : 0.0));
            }
            chk(SCIPtrySolFree(scip, &sol, FALSE, FALSE, TRUE, TRUE, TRUE, &stored));
        }
        chk(SCIPsetRealParam(scip, "limits/time", tlim));
        chk(SCIPsetRealParam(scip, "limits/gap", 1e-9));
        chk(SCIPsetRealParam(scip, "limits/absgap", 1e-9));
        chk(SCIPsetRealParam(scip, "numerics/feastol", diag_cfg().feastol));
        if (diag_cfg().clock > 0) chk(SCIPsetIntParam(scip, "timing/clocktype", diag_cfg().clock));
        if (diag_cfg().nlp == 0) chk(SCIPsetBoolParam(scip, "nlp/disable", TRUE));
        if (diag_cfg().heur == 0) chk(SCIPsetHeuristics(scip, SCIP_PARAMSETTING_OFF, TRUE));
        if (diag_cfg().dump) {
            static std::atomic<int> done{0};
            if (done.exchange(1) == 0) chk(SCIPwriteParams(scip, diag_cfg().dump, TRUE, FALSE));
        }
        chk(SCIPsolve(scip));
        const SCIP_STATUS st = SCIPgetStatus(scip);
        rc = (st == SCIP_STATUS_OPTIMAL) ? 0 : (st == SCIP_STATUS_INFEASIBLE) ? 1 : (st == SCIP_STATUS_TIMELIMIT) ? 2 : 3;
        SCIP_SOL* best = SCIPgetBestSol(scip);
        if (best && rc == 0) {
            q_out.assign(n, 0);
            for (int i = 0; i < n; ++i) q_out[i] = (int)std::llround(SCIPgetSolVal(scip, best, q[i]));
            // vérification EXACTE de notre côté (somme, cardinalité, plage, risque <= B + 1e-6)
            int sum = 0, act = 0; bool rng = true; double val = 0.0, risk = 0.0;
            for (int i = 0; i < n; ++i) {
                sum += q_out[i];
                if (q_out[i]) { ++act; if (q_out[i] < L || q_out[i] > U) rng = false; }
                val += cost[i] * q_out[i];
                double g = 0.0; for (int j = 0; j < n; ++j) g += S[i][j] * q_out[j];
                risk += q_out[i] * (g + 2.0 * r[i]);
            }
            if (!(sum == Q && act <= k && rng && risk <= B + 1e-6)) rc = 3;
            else { primal = val; dual = SCIPgetDualbound(scip) * cs; }
        } else if (rc == 0) rc = 3;
    } catch (const std::exception&) { rc = 3; }
    try {
        for (auto* c_ : cons) if (c_) SCIPreleaseCons(scip, &c_);
        for (int i = 0; i < n; ++i) { if (q[i]) SCIPreleaseVar(scip, &q[i]); if (y[i]) SCIPreleaseVar(scip, &y[i]); }
        if (created) SCIPfreeProb(scip);
    } catch (...) { rc = 3; }
    return rc;
}

void ccpo_scip_install_hook() { ccpo_scip_hook() = &ccpo_scip_zeta_solve; }
