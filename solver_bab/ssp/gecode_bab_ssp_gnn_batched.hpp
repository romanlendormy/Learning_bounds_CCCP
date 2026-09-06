#pragma once
/*
 * gecode_bab_ssp_gnn_batched.hpp — Batching multi-nœuds pour la borne GNN SSP,
 * mirror structurel de solver_bab/mkp/gecode_bab_gnn_batched.hpp (MKP).
 *
 * Casse la fusion commit()->more()->status() de SSPSpaceBnB (gecode_bab_ssp.hpp) :
 * ici, commit() fixe juste la variable, AUCUN calcul de borne. Le calcul est
 * différé, regroupé par bucket, et effectué en UN SEUL appel batché par
 * bucket flush (Étape B2, pas encore branché ici -- cf. driver .cpp).
 *
 * Différence clé avec MKP pour le bucketing : côté MKP, le sous-problème
 * résiduel rétrécit sur un seul axe (n_g = items encore libres), donc un
 * bucket par valeur de n_g suffit (vector<vector<PendingEntry>>). Côté SSP,
 * DEUX axes rétrécissent indépendamment : G = |{(e,t) libres}| (via le
 * branchement sur x_flat) ET N_z = |{scénarios libres}| (via la résolution
 * incidente de z par propagation CP, cf. discussion 2026-08-19 -- j'avais
 * affirmé à tort que le graphe SSP était de taille constante, ce qui est
 * faux : G et N_z varient tous les deux). Le bucket doit donc être indexé par
 * la PAIRE (G, N_z), pas un seul entier -- cf. driver .cpp (std::map).
 *
 * Nouvelle classe SSPSpaceGNNBatched (PAS une réutilisation de SSPSpaceBnB).
 * Réutilise SSPSharedData (gecode_bab_ssp.hpp) TELLE QUELLE.
 *
 * Fichier séparé de gecode_bab_ssp.hpp/.cpp -- AUCUNE modification du code de
 * production (même convention que pour MKP).
 */

#include "gecode_bab_ssp.hpp"

// ═════════════════════════════════════════════════════════════════════════════
// SSPSpaceGNNBatched — espace de recherche Gecode, borne calculée EXTERNEMENT
// ═════════════════════════════════════════════════════════════════════════════
class SSPSpaceGNNBatched : public IntMinimizeSpace {
public:
    IntVarArray  x_flat;
    IntVarArray  y;
    BoolVarArray z;
    IntVar       total_cost;
    const SSPSharedData* data;   // read-only, non cloné

    // ── SimpleBrancher — identique à BoundBrancher (gecode_bab_ssp.hpp) sauf
    // que commit() NE CALCULE AUCUNE BORNE : fixe juste la variable. Le calcul
    // est fait par le driver, en dehors du cycle commit()/status() de Gecode.
    class SimpleBrancher : public Brancher {
    protected:
        ViewArray<Int::IntView> vars;

        struct PosVal : public Choice {
            int pos;
            PosVal(const SimpleBrancher& b, int p)
                : Choice(b, SSP_BAB_SIGMA_SIZE), pos(p) {}
            void archive(Archive& e) const override { Choice::archive(e); e << pos; }
        };

    public:
        SimpleBrancher(Home home, ViewArray<Int::IntView>& v) : Brancher(home), vars(v) {}

        static void post(Home home, ViewArray<Int::IntView>& v) {
            new (home) SimpleBrancher(home, v);
        }

        size_t dispose(Space& home) override { Brancher::dispose(home); return sizeof(*this); }

        SimpleBrancher(Space& home, SimpleBrancher& b) : Brancher(home, b) { vars.update(home, b.vars); }

        Brancher* copy(Space& home) override { return new (home) SimpleBrancher(home, *this); }

        bool status(const Space&) const override {
            for (int i = 0; i < vars.size(); ++i)
                if (!vars[i].assigned()) return true;
            return false;
        }

        // Même heuristique de branchement que BoundBrancher (RATIO/SIZE_MIN) —
        // pour rester comparable au mode GNN non batché (test_gnn_v1cld_bab.cpp
        // utilise SIZE_MIN).
        Choice* choice(Space& home) override {
            const auto* data = static_cast<SSPSpaceGNNBatched&>(home).data;
            if (data->branch_heuristic == SSPBranchHeuristic::RATIO) {
                for (int idx : data->branch_order)
                    if (!vars[idx].assigned()) return new PosVal(*this, idx);
                GECODE_NEVER;
                return nullptr;
            }
            int best = -1, best_size = std::numeric_limits<int>::max();
            for (int i = 0; i < vars.size(); ++i) {
                if (vars[i].assigned()) continue;
                int sz = vars[i].size();
                if (sz < best_size) { best_size = sz; best = i; }
            }
            if (best < 0) GECODE_NEVER;
            return new PosVal(*this, best);
        }

        Choice* choice(const Space&, Archive& e) override {
            int pos; e >> pos;
            return new PosVal(*this, pos);
        }

        // commit() : fixe vars[pos]=valeur, RIEN D'AUTRE — pas de more(), pas
        // de borne. C'est la différence structurelle avec BoundBrancher.
        ExecStatus commit(Space& home, const Choice& c, unsigned int a) override {
            const PosVal& pv = static_cast<const PosVal&>(c);
            auto& space = static_cast<SSPSpaceGNNBatched&>(home);
            static const int VALUE_ORDER_RATIO[SSP_BAB_SIGMA_SIZE] = {
                SSP_BAB_SIGMA_W, 0, 1, 2
            };
            static const int VALUE_ORDER_SIZE_MIN[SSP_BAB_SIGMA_SIZE] = {
                0, 1, 2, SSP_BAB_SIGMA_W
            };
            const int* value_order = (space.data->branch_heuristic == SSPBranchHeuristic::RATIO)
                                      ? VALUE_ORDER_RATIO : VALUE_ORDER_SIZE_MIN;
            return me_failed(vars[pv.pos].eq(home, value_order[a])) ? ES_FAILED : ES_OK;
        }

        void print(const Space& home, const Choice& c, unsigned int a,
                   std::ostream& os) const override {
            const PosVal& pv = static_cast<const PosVal&>(c);
            static const int VALUE_ORDER_RATIO[SSP_BAB_SIGMA_SIZE] = {
                SSP_BAB_SIGMA_W, 0, 1, 2
            };
            static const int VALUE_ORDER_SIZE_MIN[SSP_BAB_SIGMA_SIZE] = {
                0, 1, 2, SSP_BAB_SIGMA_W
            };
            const auto& data = static_cast<const SSPSpaceGNNBatched&>(home).data;
            const int* value_order = (data->branch_heuristic == SSPBranchHeuristic::RATIO)
                                      ? VALUE_ORDER_RATIO : VALUE_ORDER_SIZE_MIN;
            os << "x_flat[" << pv.pos << "] = " << value_order[a];
        }
    };

    // ── Constructeur racine — IDENTIQUE au modèle CP de SSPSpaceBnB
    // (gecode_bab_ssp.hpp) — juste SimpleBrancher au lieu de BoundBrancher. ──
    explicit SSPSpaceGNNBatched(const SSPSharedData* d)
        : x_flat    (*this, d->E * d->T, 0, SSP_BAB_SIGMA_SIZE - 1),
          y         (*this, d->T, 0, d->E),
          z         (*this, d->N, 0, 1),
          total_cost(*this, 0, d->sal_ub),
          data(d)
    {
        const int E = d->E, T = d->T, N = d->N;

        for (int e = 0; e < E; ++e) {
            IntVarArgs row(T);
            for (int t = 0; t < T; ++t) row[t] = x_flat[e * T + t];
            extensional(*this, row, d->dfa);
        }

        BoolVarArray works(*this, E * T, 0, 1);
        for (int e = 0; e < E; ++e)
            for (int t = 0; t < T; ++t)
                dom(*this, x_flat[e * T + t], d->work_set, works[e * T + t]);

        for (int t = 0; t < T; ++t) {
            BoolVarArgs col(E);
            for (int e = 0; e < E; ++e) col[e] = works[e * T + t];
            linear(*this, col, IRT_EQ, y[t]);
        }

        BoolVarArray zti(*this, N * T, 0, 1);
        for (int i = 0; i < N; ++i) {
            for (int t = 0; t < T; ++t) {
                int dmd = d->demand[i][t];
                if (dmd <= 0)      rel(*this, zti[i * T + t], IRT_EQ, 1);
                else if (dmd > E)  rel(*this, zti[i * T + t], IRT_EQ, 0);
                else               rel(*this, y[t], IRT_GQ, dmd, Reify(zti[i * T + t]));
            }
            BoolVarArgs row_i(T);
            for (int t = 0; t < T; ++t) row_i[t] = zti[i * T + t];
            rel(*this, BOT_AND, row_i, z[i]);
        }
        linear(*this, z, IRT_GQ, d->min_feas);

        IntArgs sal(T);
        for (int t = 0; t < T; ++t) sal[t] = d->sal_sc[t];
        linear(*this, sal, y, IRT_EQ, total_cost);

        for (int e = 0; e < E - 1; ++e) {
            IntVarArgs row_e(T), row_e1(T);
            for (int t = 0; t < T; ++t) {
                row_e [t] = x_flat[ e      * T + t];
                row_e1[t] = x_flat[(e + 1) * T + t];
            }
            rel(*this, row_e, IRT_LQ, row_e1);
        }

        IntVarArgs x_args(E * T);
        for (int k = 0; k < E * T; ++k) x_args[k] = x_flat[k];
        ViewArray<Int::IntView> xv(*this, x_args);
        SimpleBrancher::post(*this, xv);

        branch(*this, z, BOOL_VAR_NONE(), BOOL_VAL_MAX());
    }

    SSPSpaceGNNBatched(SSPSpaceGNNBatched& s) : IntMinimizeSpace(s), data(s.data)
    {
        x_flat    .update(*this, s.x_flat);
        y         .update(*this, s.y);
        z         .update(*this, s.z);
        total_cost.update(*this, s.total_cost);
    }

    Space*  copy() override { return new SSPSpaceGNNBatched(*this); }
    IntVar  cost() const override { return total_cost; }

    // Extrait (x_mask, z_mask, curr_cost, depth) pour l'appel GNN — même
    // logique que more() (gecode_bab_ssp.hpp), Étape 1 (scan) + section GNN.
    // Retourne false si aucune borne n'a de sens ici (any_free==false : feuille
    // en x, total_cost déjà exact -- OU free_z_count==0 : z entièrement
    // résolu par propagation avant tout branchement explicite, zone jamais
    // entraînée) -- mêmes deux court-circuits que more(), le nœud doit alors
    // être traité IMMÉDIATEMENT par le driver (poussé dans `open` avec son
    // total_cost.min() courant, sans borne).
    bool extract_pending(std::vector<int8_t>& x_mask, std::vector<int8_t>& z_mask,
                          double& curr_cost, int& depth, int& G, int& N_z) const {
        const int E = data->E, T = data->T, N = data->N;

        x_mask.assign(E * T, static_cast<int8_t>(SSP_BAB_SIGMA_SIZE));
        curr_cost = 0.0;
        G = 0;
        int free_count = 0;
        for (int e = 0; e < E; ++e)
            for (int t = 0; t < T; ++t) {
                const IntVar& v = x_flat[e * T + t];
                if (v.assigned()) {
                    x_mask[e * T + t] = static_cast<int8_t>(v.val());
                    if (v.val() == SSP_BAB_SIGMA_W) curr_cost += data->salary[t];
                } else {
                    ++G;
                    ++free_count;
                }
            }
        if (G == 0) return false;

        z_mask.assign(N, 0);
        N_z = 0;
        for (int i = 0; i < N; ++i) {
            if (z[i].assigned()) {
                z_mask[i] = static_cast<int8_t>(z[i].val());
            } else {
                z_mask[i] = 2;
                ++N_z;
            }
        }
        if (N_z == 0) return false;

        depth = E * T - free_count;
        return true;
    }

    // Poste la borne calculée EXTERNEMENT (borne inférieure sur le coût
    // salarial réel, déjà convertie -- floor déjà appliqué par l'appelant,
    // même convention que more(), gecode_bab_ssp.hpp).
    void post_gnn_bound(int lb_int) {
        if (lb_int > total_cost.min())
            rel(*this, total_cost >= lb_int);
    }
};

// ═════════════════════════════════════════════════════════════════════════════
// Backend batché — même contrat que SSPGNNBoundBackend (ssp_gnn_bound_backend.hpp)
// mais UN appel pour tout un bucket. cost_lower_out[i] : borne inférieure sur
// le coût salarial réel du membre i (même convention que
// GNNV1CLDNative::query -- valeur directement utilisable, pas de conversion
// de signe supplémentaire côté driver).
// ═════════════════════════════════════════════════════════════════════════════
struct SSPGNNBoundBackendBatch {
    virtual ~SSPGNNBoundBackendBatch() = default;
    virtual void query_batch(const std::vector<std::vector<int8_t>>& x_masks,
                              const std::vector<std::vector<int8_t>>& z_masks,
                              const std::vector<double>& curr_costs,
                              const std::vector<int>& depths,
                              std::vector<double>& cost_lower_out) = 0;
};

// ═════════════════════════════════════════════════════════════════════════════
// Résultat + point d'entrée
// ═════════════════════════════════════════════════════════════════════════════
struct GNNBatchedSSPBabResult {
    double      objective = std::numeric_limits<double>::quiet_NaN();
    bool        feasible  = false;
    bool        optimal   = false;
    std::string status;
    int64_t     nodes = 0, fails = 0, depth = 0, incumbent_improvements = 0;
    double      solve_time_s = 0.0;
    int         K = 1;
    int64_t     n_bucket_flushes = 0;
    double      mean_bucket_fill_at_flush = 0.0;
    int64_t     max_pending_live = 0;
};

GNNBatchedSSPBabResult solve_gecode_bab_ssp_gnn_batched(
    int E, int T, int N, double epsilon,
    const std::vector<std::tuple<int,int,int>>& transitions,
    const std::vector<int>& accepting, int initial,
    const Vec1d& salary, const Vec2i& demand, const Vec1d& p,
    double time_lim, int K,
    SSPGNNBoundBackendBatch* backend,
    SSPBranchHeuristic branch_heuristic = SSPBranchHeuristic::SIZE_MIN);
