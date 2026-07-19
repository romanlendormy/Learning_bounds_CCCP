/*
 * zeta_gecode_solver.cpp — cf. zeta_gecode_solver.hpp pour le contexte.
 *
 * Préfiltrage IDENTIQUE à MKPProblemBnB/MKPProblemBnBSurrogate : items à
 * coût <= 0 fixés à 0 d'office (jamais utile de les inclure, ne relâche
 * aucune contrainte), items "libres" (poids nul dans toutes les dimensions,
 * coût > 0) fixés à 1 d'office. Seuls les items "actifs" restants sont
 * modélisés dans Gecode, à des POSITIONS LOCALES déjà triées par ratio
 * surrogate décroissant (position 0 = meilleur ratio) — le brancher n'a donc
 * qu'à chercher la première position non assignée, pas besoin d'indirection.
 */

#include "zeta_gecode_solver.hpp"
#include "../../solver_generic/mkp/mkp_surrogate_bound.hpp"

#include <gecode/int.hh>
#include <gecode/minimodel.hh>
#include <gecode/search.hh>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace Gecode;

namespace {

constexpr int64_t SCALE = 1000;
constexpr double  EPS   = 1e-9;

// Données partagées en lecture seule (pointeur brut, PAS cloné par Gecode —
// même pattern que MKPSharedData dans gecode_bab.hpp : durée de vie garantie
// par la pile de solve_zeta_gecode, au-delà de la durée de vie de la recherche).
struct ZetaSharedData {
    int na, m;
    std::vector<int64_t>              c_int;    // (na,) coût scalé, position déjà triée
    std::vector<std::vector<int64_t>> A_int;    // (m, na) capacité scalée
    Vec1d                              cost_d;   // (na,) coût non scalé (pour la borne)
    Vec2d                               A_d;      // (m, na) non scalé (pour la borne)
    Vec1d                               agg_w;    // (na,) poids agrégé surrogate, position triée
    Vec1d                               u;        // (m,) poids surrogate par dimension
    Vec1d                               b_orig_d; // (m,) capacité originale non scalée
    std::vector<int64_t>                b_int;    // (m,) capacité scalée (floor, conservatif — cf. gecode_bab.hpp)
    int64_t                             obj_max_int = 0;
};

// Borne surrogate sur le sous-problème résiduel, ROBUSTE à un ordre de
// décision non contigu (la propagation Gecode peut fixer des variables hors
// de l'ordre du brancher) : rebalaye TOUTES les positions à chaque appel
// (na petit — coût négligeable face au reste du nœud Gecode), accumule la
// valeur/capacité des positions fixées à 1, puis remplit glouton sur les
// positions non assignées dans l'ordre déjà trié (0..na-1).
double surrogate_bound_scan(const ZetaSharedData& d, const ViewArray<Int::BoolView>& vars)
{
    double val = 0.0;
    Vec1d rem_b = d.b_orig_d;
    for (int p = 0; p < d.na; ++p) {
        if (vars[p].assigned() && vars[p].val() == 1) {
            val += d.cost_d[p];
            for (int j = 0; j < d.m; ++j) rem_b[j] -= d.A_d[j][p];
        }
    }
    double cap = 0.0;
    for (int j = 0; j < d.m; ++j) cap += d.u[j] * rem_b[j];
    if (std::getenv("ZETA_GECODE_DEBUG_CAP")) {
        bool neg = false;
        for (int j = 0; j < d.m; ++j) if (rem_b[j] < -1e-6) neg = true;
        if (neg || cap < -1e-6) {
            std::cerr << "[CAPDBG] val=" << val << " cap=" << cap << " rem_b=[";
            for (int j = 0; j < d.m; ++j) std::cerr << rem_b[j] << (j+1<d.m?",":"");
            std::cerr << "]\n";
        }
    }

    double v = val;
    for (int p = 0; p < d.na; ++p) {
        if (vars[p].assigned()) continue;   // déjà décidé (0 ou 1 — le 1 est déjà dans 'val')
        double a = d.agg_w[p];
        if (a < EPS)             { v += d.cost_d[p]; }
        else if (a <= cap + EPS) { v += d.cost_d[p]; cap -= a; }
        else                     { v += d.cost_d[p] * cap / a; break; }
    }
    return v;
}

class ZetaSpaceGecode : public IntMaximizeSpace {
public:
    BoolVarArray x;
    IntVar       obj;
    const ZetaSharedData* data;

    class SurrBrancher : public Brancher {
    protected:
        ViewArray<Int::BoolView> vars;
    public:
        struct Pos : public Choice {
            int pos;
            Pos(const SurrBrancher& b, int p) : Choice(b, 2), pos(p) {}
            void archive(Archive& e) const override { Choice::archive(e); e << pos; }
        };

        SurrBrancher(Home home, ViewArray<Int::BoolView>& v) : Brancher(home), vars(v) {}
        static void post(Home home, ViewArray<Int::BoolView>& v) { new (home) SurrBrancher(home, v); }
        size_t dispose(Space& home) override { Brancher::dispose(home); return sizeof(*this); }
        SurrBrancher(Space& home, SurrBrancher& b) : Brancher(home, b) { vars.update(home, b.vars); }
        Brancher* copy(Space& home) override { return new (home) SurrBrancher(home, *this); }

        bool status(const Space&) const override {
            for (int p = 0; p < vars.size(); ++p) if (!vars[p].assigned()) return true;
            return false;
        }

        // Positions déjà triées par ratio décroissant (cf. en-tête du fichier) :
        // il suffit de prendre la première non assignée, pas d'indirection.
        Choice* choice(Space& home) override {
            (void)home;
            for (int p = 0; p < vars.size(); ++p)
                if (!vars[p].assigned())
                    return new Pos(*this, p);
            GECODE_NEVER;
            return nullptr;
        }
        Choice* choice(const Space&, Archive& e) override {
            int pos; e >> pos; return new Pos(*this, pos);
        }

        // a==0 : x[pos]=1 essayé en premier (favorise un bon incumbent rapide,
        // comme MkpBnB::run et BoundBrancher) ; a==1 : x[pos]=0.
        ExecStatus commit(Space& home, const Choice& c, unsigned int a) override {
            auto& sp = static_cast<ZetaSpaceGecode&>(home);
            const Pos& pv = static_cast<const Pos&>(c);
            int val = (a == 0) ? 1 : 0;
            if (me_failed(vars[pv.pos].eq(home, val))) return ES_FAILED;

            // Poste obj <= ub_int : si contradictoire (borne < incumbent), Gecode
            // détecte l'échec au prochain status()/propagate() — ne PAS le
            // vérifier ici (même convention que BoundBrancher::commit() dans
            // gecode_bab.hpp, où more() poste rel() sans vérifier son retour).
            double ub = surrogate_bound_scan(*sp.data, vars);
            int64_t ub_int = static_cast<int64_t>(std::ceil(ub * SCALE + 1e-6));
            static const char* dbg_min = std::getenv("ZETA_GECODE_DEBUG_MIN");
            if (dbg_min && ub_int < std::atoll(dbg_min)) {
                std::cerr << "[DBG] fixe pos=" << pv.pos << " val=" << val
                          << " ub=" << ub << " ub_int=" << ub_int << "  fixes: ";
                for (int p = 0; p < vars.size(); ++p)
                    if (vars[p].assigned())
                        std::cerr << p << "=" << vars[p].val() << " ";
                std::cerr << "\n";
            }
            if (!std::getenv("ZETA_GECODE_NO_BOUND"))
                rel(home, sp.obj <= static_cast<int>(ub_int));
            return ES_OK;
        }

        void print(const Space&, const Choice& c, unsigned int a,
                   std::ostream& os) const override {
            const Pos& pv = static_cast<const Pos&>(c);
            os << "x[" << pv.pos << "] = " << (a == 0 ? 1 : 0);
        }
    };

    explicit ZetaSpaceGecode(const ZetaSharedData* d)
        : x(*this, d->na, 0, 1), obj(*this, 0, static_cast<int>(d->obj_max_int)), data(d)
    {
        IntVarArgs x_int(d->na);
        for (int p = 0; p < d->na; ++p) x_int[p] = channel(*this, x[p]);

        for (int j = 0; j < d->m; ++j) {
            IntArgs coefs(d->na);
            for (int p = 0; p < d->na; ++p) coefs[p] = static_cast<int>(d->A_int[j][p]);
            linear(*this, coefs, x_int, IRT_LQ, static_cast<int>(d->b_int[j]));
        }

        IntArgs c_args(d->na);
        for (int p = 0; p < d->na; ++p) c_args[p] = static_cast<int>(d->c_int[p]);
        linear(*this, c_args, x_int, IRT_EQ, obj);

        BoolVarArgs x_args(d->na);
        for (int p = 0; p < d->na; ++p) x_args[p] = x[p];
        ViewArray<Int::BoolView> xv(*this, x_args);
        SurrBrancher::post(*this, xv);

        // Borne racine (avant tout branchement) — resserre obj dès le départ.
        if (!std::getenv("ZETA_GECODE_NO_BOUND")) {
            double ub0 = surrogate_bound_scan(*d, xv);
            int64_t ub0_int = static_cast<int64_t>(std::ceil(ub0 * SCALE + 1e-6));
            rel(*this, obj <= static_cast<int>(ub0_int));
        }
    }

    ZetaSpaceGecode(ZetaSpaceGecode& s) : IntMaximizeSpace(s), data(s.data) {
        x.update(*this, s.x);
        obj.update(*this, s.obj);
    }
    Space* copy() override { return new ZetaSpaceGecode(*this); }
    IntVar cost() const override { return obj; }
};

} // anonymous namespace

std::pair<double, Vec1d> solve_zeta_gecode(
    int n, int m, const Vec2d& Ai, const Vec1d& bi, const Vec1d& cost)
{
    Vec1d  x_out(n, 0.0);
    double val_free = 0.0;

    std::vector<int> active;
    active.reserve(n);
    for (int k = 0; k < n; ++k) {
        if (cost[k] <= 0.0) continue;
        bool free_item = true;
        for (int j = 0; j < m; ++j)
            if (Ai[j][k] >= 1e-9) { free_item = false; break; }
        if (free_item) { x_out[k] = 1.0; val_free += cost[k]; }
        else           { active.push_back(k); }
    }
    if (active.empty()) return {val_free, std::move(x_out)};

    MkpSurrogatePrep prep = prepare_mkp_surrogate(active, cost, Ai, bi, m);
    int na = static_cast<int>(prep.active_sorted.size());

    ZetaSharedData data;
    data.na = na; data.m = m;
    data.c_int.resize(na);
    data.A_int.assign(m, std::vector<int64_t>(na));
    data.cost_d.resize(na);
    data.A_d.assign(m, Vec1d(na));
    data.agg_w.resize(na);
    data.u = prep.u;
    data.b_orig_d = bi;
    data.b_int.resize(m);
    for (int j = 0; j < m; ++j)
        data.b_int[j] = static_cast<int64_t>(std::floor(bi[j] * SCALE));

    for (int p = 0; p < na; ++p) {
        int k = prep.active_sorted[p];
        data.c_int[p]  = static_cast<int64_t>(std::llround(cost[k] * SCALE));
        data.cost_d[p] = cost[k];
        data.agg_w[p]  = prep.agg_w_sorted[p];
        for (int j = 0; j < m; ++j) {
            data.A_int[j][p] = static_cast<int64_t>(std::llround(Ai[j][k] * SCALE));
            data.A_d[j][p]   = Ai[j][k];
        }
    }
    int64_t obj_max = 0;
    for (int p = 0; p < na; ++p) if (data.c_int[p] > 0) obj_max += data.c_int[p];
    data.obj_max_int = obj_max;

    ZetaSpaceGecode* root = new ZetaSpaceGecode(&data);
    Search::Options opts;
    BAB<ZetaSpaceGecode> engine(root, opts);
    delete root;

    ZetaSpaceGecode* best = nullptr;
    while (ZetaSpaceGecode* sol = engine.next()) {
        delete best;
        best = sol;
    }

    // Recalcule la valeur depuis le coût BRUT (non scalé) des items sélectionnés
    // — comme MKPProblem::solve_zeta (CP-SAT) — plutôt que de diviser
    // best->obj.val() (entier, arrondi) par SCALE, qui introduirait un bruit
    // d'arrondi évitable (repéré via debug_zeta_gecode.cpp : écart ~1e-3 sur
    // une instance n=20, alors que la solution x elle-même était optimale).
    double best_val = val_free;
    if (best != nullptr) {
        for (int p = 0; p < na; ++p)
            if (best->x[p].val() == 1) {
                int k = prep.active_sorted[p];
                x_out[k] = 1.0;
                best_val += cost[k];
            }
        delete best;
    }
    return {best_val, std::move(x_out)};
}
