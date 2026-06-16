#pragma once
/*
 * gecode_bab_collect_base.hpp — B&B Gecode CP pur avec collecte GNN.
 *
 * Identique à gecode_bab_collect mais sans borne Lagrangienne.
 * Le pruning repose uniquement sur la propagation CP de Gecode (comme gecode_base).
 *
 * Format de sortie — deux fichiers CSV :
 *
 *   instances.txt  (une ligne par instance résolue à optimalité)
 *     instance_id, m, n, N, epsilon, opt,
 *     c[0..n-1],
 *     b[k=0..N-1, j=0..m-1],
 *     A[k=0..N-1, j=0..m-1, i=0..n-1]
 *
 *   nodes.txt  (une ligne par nœud de branchement)  — Format C
 *     instance_id, depth, curr_cost,
 *     x[0..n-1],   z[0..N-1]
 *   Encodage ternaire : 0 = fixé à 0, 1 = fixé à 1, 2 = libre
 *   (3 + n + N champs — sans ub_node ni incumbent)
 */

#include "../solver_bab/gecode_bab.hpp"

#include <fstream>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// NodeRecordBase — état d'un nœud sans borne Lagrangienne
// ─────────────────────────────────────────────────────────────────────────────

struct NodeRecordBase {
    int    depth;
    double curr_cost;
    std::vector<int8_t> x_mask;  // 0=fixé_0, 1=fixé_1, 2=libre  (taille n)
    std::vector<int8_t> z_mask;  // 0=fixé_0, 1=fixé_1, 2=libre  (taille N)
};

// ─────────────────────────────────────────────────────────────────────────────
// MKPSpaceBase — espace Gecode CP pur (pas de more())
// ─────────────────────────────────────────────────────────────────────────────

class MKPSpaceBase : public IntMaximizeSpace {
public:
    BoolVarArray x, z;
    IntVar       obj;
    const MKPSharedData* data;

    // Brancher qui suit branch_order mais ne calcule pas de borne Lagrangienne.
    class SimpleBrancher : public Brancher {
    protected:
        ViewArray<Int::BoolView> vars;

        struct PosVal : public Choice {
            int pos, val;
            PosVal(const SimpleBrancher& b, int p, int v)
                : Choice(b, 2), pos(p), val(v) {}
            void archive(Archive& e) const override {
                Choice::archive(e); e << pos << val;
            }
        };

    public:
        SimpleBrancher(Home home, ViewArray<Int::BoolView>& v)
            : Brancher(home), vars(v) {}

        static void post(Home home, ViewArray<Int::BoolView>& v) {
            new (home) SimpleBrancher(home, v);
        }

        size_t dispose(Space& home) override {
            Brancher::dispose(home); return sizeof(*this);
        }

        SimpleBrancher(Space& home, SimpleBrancher& b)
            : Brancher(home, b) { vars.update(home, b.vars); }

        Brancher* copy(Space& home) override {
            return new (home) SimpleBrancher(home, *this);
        }

        bool status(const Space&) const override {
            for (int i = 0; i < vars.size(); ++i)
                if (!vars[i].assigned()) return true;
            return false;
        }

        Choice* choice(Space& home) override {
            const auto& order = static_cast<MKPSpaceBase&>(home).data->branch_order;
            for (int k : order)
                if (!vars[k].assigned())
                    return new PosVal(*this, k, 1);
            GECODE_NEVER;
            return nullptr;
        }

        Choice* choice(const Space&, Archive& e) override {
            int pos, val; e >> pos >> val;
            return new PosVal(*this, pos, val);
        }

        ExecStatus commit(Space& home, const Choice& c, unsigned int a) override {
            const PosVal& pv = static_cast<const PosVal&>(c);
            int val = (a == 0) ? 1 : 0;
            // Pas d'appel à more() : pruning CP uniquement.
            return me_failed(vars[pv.pos].eq(home, val)) ? ES_FAILED : ES_OK;
        }

        void print(const Space&, const Choice& c, unsigned int a,
                   std::ostream& os) const override {
            const PosVal& pv = static_cast<const PosVal&>(c);
            os << "x[" << pv.pos << "] = " << (a == 0 ? 1 : 0);
        }
    };

    explicit MKPSpaceBase(const MKPSharedData* d)
        : x(*this, d->n, 0, 1),
          z(*this, d->N, 0, 1),
          obj(*this, 0, d->obj_max),
          data(d)
    {
        int n = d->n, m = d->m, N = d->N;
        int min_feas = static_cast<int>(std::ceil((1.0 - d->epsilon) * N));

        IntVarArgs x_int(n);
        for (int k = 0; k < n; ++k) x_int[k] = channel(*this, x[k]);

        for (int i = 0; i < N; ++i)
            for (int j = 0; j < m; ++j) {
                IntArgs coefs(n);
                for (int k = 0; k < n; ++k) coefs[k] = d->A_int[i][j][k];
                linear(*this, coefs, x_int, IRT_LQ, d->b_int[i][j],
                       Reify(z[i], RM_IMP));
            }

        linear(*this, z, IRT_GQ, min_feas);

        IntArgs c_args(n);
        for (int k = 0; k < n; ++k) c_args[k] = d->c_int[k];
        linear(*this, c_args, x_int, IRT_EQ, obj);

        if (d->initial_lb_int > 0)
            rel(*this, obj >= d->initial_lb_int);

        BoolVarArgs x_args(n);
        for (int k = 0; k < n; ++k) x_args[k] = x[k];
        ViewArray<Int::BoolView> xv(*this, x_args);
        SimpleBrancher::post(*this, xv);

        branch(*this, z, BOOL_VAR_NONE(), BOOL_VAL_MAX());
    }

    MKPSpaceBase(MKPSpaceBase& s)
        : IntMaximizeSpace(s), data(s.data)
    {
        x.update(*this, s.x);
        z.update(*this, s.z);
        obj.update(*this, s.obj);
    }

    Space*  copy()  override { return new MKPSpaceBase(*this); }
    IntVar  cost()  const override { return obj; }
};

// ─────────────────────────────────────────────────────────────────────────────
// Fonctions de collecte et d'écriture
// ─────────────────────────────────────────────────────────────────────────────

NodeRecordBase collect_node_state_base(
    const MKPSpaceBase* space,
    const MKPSharedData& data,
    int depth);

void write_instance_line_base(
    std::ofstream& out,
    int instance_id,
    const MKPSharedData& data,
    double opt);

// Format C : inst_id, depth, curr_cost, x_mask[n], z_mask[N]  (3 + n + N champs)
void write_node_line_base(
    std::ofstream& out,
    int instance_id,
    const NodeRecordBase& rec);

// ─────────────────────────────────────────────────────────────────────────────
// solve_gecode_bab_mkp_collect_base
// ─────────────────────────────────────────────────────────────────────────────

GecodeV1LDResult solve_gecode_bab_mkp_collect_base(
    int n, int m, int N, double epsilon,
    const Vec1d& c, const Vec1d& p,
    const Vec3d& A, const Vec2d& b,
    std::ofstream& instances_out,
    std::ofstream& nodes_out,
    int       instance_id,
    double    time_lim           = 300.0,
    int       max_depth_collect  = -1);
