/*
 * gecode_solver.cpp — Solveur CP exact (Gecode 6) pour CCMKP et CCSSP.
 *
 * CCMKP : Formulation CP directe avec réification (Reify + RM_IMP)
 *         z_i = 1 → A_i x ≤ b_i   (sans Big-M)
 *
 * CCSSP : Automate produit + extensional + dom-réification
 *
 * SCALE = 1000 pour les coefficients réels → entiers int32.
 */

#include "gecode_solver.hpp"

#include <gecode/int.hh>
#include <gecode/search.hh>
#include <gecode/minimodel.hh>

#include <algorithm>
#include <chrono>
#include <cmath>

using namespace Gecode;

// ── Timer ─────────────────────────────────────────────────────────────────────

using Clock = std::chrono::steady_clock;
static double elapsed_s(std::chrono::time_point<Clock> t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

static constexpr int SCALE = 1000;
static int to_int(double x)       { return static_cast<int>(std::llround(x * SCALE)); }
static int to_int_floor(double x) { return static_cast<int>(std::floor(x * SCALE)); }

// ════════════════════════════════════════════════════════════════════════════════
// CCMKP — CP direct avec réification
// ════════════════════════════════════════════════════════════════════════════════
//
//   max  Σ c_k x_k
//   s.t. z_i = 1  →  Σ_k A_ijk x_k ≤ b_ij   ∀ i, j   (Reify RM_IMP)
//        Σ z_i ≥ ⌈(1-ε)N⌉
//        x ∈ {0,1}^n,  z ∈ {0,1}^N

class MKPSpace : public IntMaximizeSpace {
public:
    BoolVarArray x, z;
    IntVar       obj;

    MKPSpace(int n, int m, int N, int min_feas,
             const std::vector<int>& c_int,
             const std::vector<std::vector<std::vector<int>>>& A_int,
             const std::vector<std::vector<int>>& b_int,
             int obj_max)
        : x(*this, n, 0, 1),
          z(*this, N, 0, 1),
          obj(*this, 0, obj_max)
    {
        // Contraintes réifiées : z_i = 1 → A_i[j] · x ≤ b_i[j]
        IntVarArgs x_int(n);
        for (int k = 0; k < n; ++k) x_int[k] = channel(*this, x[k]);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < m; ++j) {
                IntArgs coefs(n);
                for (int k = 0; k < n; ++k) coefs[k] = A_int[i][j][k];
                linear(*this, coefs, x_int, IRT_LQ, b_int[i][j],
                       Reify(z[i], RM_IMP));
            }
        }

        // Contrainte de chance
        linear(*this, z, IRT_GQ, min_feas);

        // Objectif
        IntArgs c_args(n);
        for (int k = 0; k < n; ++k) c_args[k] = c_int[k];
        linear(*this, c_args, x_int, IRT_EQ, obj);

        branch(*this, x, BOOL_VAR_NONE(), BOOL_VAL_MAX());
        branch(*this, z, BOOL_VAR_NONE(), BOOL_VAL_MAX());
    }

    MKPSpace(MKPSpace& s) : IntMaximizeSpace(s) {
        x.update(*this, s.x);
        z.update(*this, s.z);
        obj.update(*this, s.obj);
    }
    Space*  copy()  override { return new MKPSpace(*this); }
    IntVar  cost()  const override { return obj; }
};

GecodeResult solve_gecode_mkp(
    int n, int m, int N, double epsilon,
    const Vec1d& c, const Vec1d& /*p*/,
    const Vec3d& A, const Vec2d& b,
    double time_lim)
{
    auto t0 = Clock::now();

    int min_feas = static_cast<int>(std::ceil((1.0 - epsilon) * N));

    std::vector<int> c_int(n);
    for (int k = 0; k < n; ++k) c_int[k] = to_int(c[k]);

    std::vector<std::vector<std::vector<int>>> A_int(
        N, std::vector<std::vector<int>>(m, std::vector<int>(n)));
    std::vector<std::vector<int>> b_int(N, std::vector<int>(m));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < n; ++k) A_int[i][j][k] = to_int(A[i][j][k]);
            b_int[i][j] = to_int_floor(b[i][j]);
        }

    int obj_max = 0;
    for (int k = 0; k < n; ++k) obj_max += std::max(0, c_int[k]);

    MKPSpace* model = new MKPSpace(n, m, N, min_feas, c_int, A_int, b_int, obj_max);

    Search::Options opts;
    Search::TimeStop* stop = nullptr;
    if (time_lim > 0.0) {
        stop      = new Search::TimeStop(static_cast<unsigned long long>(time_lim * 1000));
        opts.stop = stop;
    }

    BAB<MKPSpace> engine(model, opts);
    delete model;

    MKPSpace* best = nullptr;
    while (MKPSpace* sol = engine.next()) { delete best; best = sol; }

    bool timed_out = stop && engine.stopped();
    delete stop;

    GecodeResult res;
    res.solve_time_s = elapsed_s(t0);
    if (best) {
        res.objective = static_cast<double>(best->obj.val()) / SCALE;
        res.feasible  = true;
        res.optimal   = !timed_out;
        res.status    = timed_out ? "FEASIBLE" : "OPTIMAL";
        delete best;
    } else {
        res.objective = std::numeric_limits<double>::quiet_NaN();
        res.feasible  = false;
        res.optimal   = false;
        res.status    = timed_out ? "TIMEOUT" : "INFEASIBLE";
    }
    return res;
}

// ════════════════════════════════════════════════════════════════════════════════
// CCSSP — automate produit + extensional + dom-réification
// ════════════════════════════════════════════════════════════════════════════════

static int compute_S(int k, int Q) {
    int S = 1;
    for (int j = 0; j < k; ++j) S *= Q;
    return S;
}

static void decode_state(int s, int k, int Q, std::vector<int>& qs) {
    for (int j = k-1; j >= 0; --j) { qs[j] = s % Q; s /= Q; }
}

static int encode_state(const std::vector<int>& qs, int k, int Q) {
    int s = 0;
    for (int j = 0; j < k; ++j) s = s * Q + qs[j];
    return s;
}

struct ProdAut {
    std::vector<std::vector<int>> trans;  // (S, m)
    std::vector<bool>             acc;    // (S,)
};

static ProdAut build_product_aut(
    int i, int k, int Q, int m, int S,
    const std::vector<std::vector<std::vector<std::vector<int>>>>& delta,
    const std::vector<std::vector<std::vector<int>>>& accepting)
{
    ProdAut pa;
    pa.trans.assign(S, std::vector<int>(m, -1));
    pa.acc.assign(S, false);

    std::vector<int> qs(k), nqs(k);
    for (int s = 0; s < S; ++s) {
        decode_state(s, k, Q, qs);
        bool all_acc = true;
        for (int j = 0; j < k; ++j)
            if (!accepting[i][j][qs[j]]) { all_acc = false; break; }
        pa.acc[s] = all_acc;

        for (int a = 0; a < m; ++a) {
            bool valid = true;
            for (int j = 0; j < k; ++j) {
                int q_next = delta[i][j][qs[j]][a];
                if (q_next < 0) { valid = false; break; }
                nqs[j] = q_next;
            }
            if (valid) pa.trans[s][a] = encode_state(nqs, k, Q);
        }
    }
    return pa;
}

class SSPSpace : public IntMaximizeSpace {
public:
    IntVarArray  action;
    IntVarArray  state_flat;
    BoolVarArray z;
    IntVarArray  profit_var;
    IntVar       total_profit;

    SSPSpace(int n, int m, int N, int S, int min_feas,
             const std::vector<int>& profit_int,
             int profit_min, int profit_max,
             const std::vector<TupleSet>& ts_vec,
             const std::vector<IntSet>&   acc_sets)
        : action(*this, n, 0, m - 1),
          state_flat(*this, N * (n + 1), 0, S),
          z(*this, N, 0, 1),
          profit_var(*this, n, profit_min, profit_max),
          total_profit(*this, n * profit_min, n * profit_max)
    {
        // État initial = 0
        for (int i = 0; i < N; ++i)
            rel(*this, state_flat[i * (n + 1)], IRT_EQ, 0);

        // Transitions
        for (int i = 0; i < N; ++i)
            for (int t = 0; t < n; ++t)
                extensional(*this,
                    IntVarArgs{state_flat[i*(n+1)+t], action[t], state_flat[i*(n+1)+t+1]},
                    ts_vec[i]);

        // Réification : z[i] ↔ état final acceptant
        for (int i = 0; i < N; ++i) {
            if (acc_sets[i].ranges() == 0)
                rel(*this, z[i], IRT_EQ, 0);
            else
                dom(*this, state_flat[i*(n+1)+n], acc_sets[i], z[i]);
        }

        // Contrainte de chance
        linear(*this, z, IRT_GQ, min_feas);

        // Profit par période
        for (int t = 0; t < n; ++t) {
            IntArgs prow(m);
            for (int a = 0; a < m; ++a) prow[a] = profit_int[t * m + a];
            element(*this, prow, action[t], profit_var[t]);
        }
        linear(*this, profit_var, IRT_EQ, total_profit);

        branch(*this, action, INT_VAR_NONE(),  INT_VAL_MAX());
        branch(*this, z,      BOOL_VAR_NONE(), BOOL_VAL_MAX());
    }

    SSPSpace(SSPSpace& s) : IntMaximizeSpace(s) {
        action.update(*this,       s.action);
        state_flat.update(*this,   s.state_flat);
        z.update(*this,            s.z);
        profit_var.update(*this,   s.profit_var);
        total_profit.update(*this, s.total_profit);
    }
    Space*  copy()  override { return new SSPSpace(*this); }
    IntVar  cost()  const override { return total_profit; }
};

GecodeResult solve_gecode_ssp(
    int n, int m, int k, int Q, int N, double epsilon,
    const Vec1d& profit, const Vec1d& /*p*/,
    const std::vector<std::vector<std::vector<std::vector<int>>>>& delta,
    const std::vector<std::vector<std::vector<int>>>& accepting,
    double time_lim)
{
    auto t0 = Clock::now();

    int S        = compute_S(k, Q);
    int min_feas = static_cast<int>(std::ceil((1.0 - epsilon) * N));

    // Profits scalés
    std::vector<int> profit_int(n * m);
    for (int i = 0; i < n * m; ++i) profit_int[i] = to_int(profit[i]);
    int pmin = *std::min_element(profit_int.begin(), profit_int.end());
    int pmax = *std::max_element(profit_int.begin(), profit_int.end());

    // Automates produit → TupleSets + IntSets acceptants (un seul passage)
    std::vector<TupleSet> ts_vec(N);
    std::vector<IntSet>   acc_sets(N);
    for (int i = 0; i < N; ++i) {
        ProdAut pa = build_product_aut(i, k, Q, m, S, delta, accepting);

        TupleSet ts(3);
        std::vector<int> acc_list;
        for (int s = 0; s < S; ++s) {
            for (int a = 0; a < m; ++a)
                ts.add({s, a, pa.trans[s][a] >= 0 ? pa.trans[s][a] : S});
            if (pa.acc[s]) acc_list.push_back(s);
        }
        for (int a = 0; a < m; ++a) ts.add({S, a, S});
        ts.finalize();
        ts_vec[i] = ts;

        if (!acc_list.empty())
            acc_sets[i] = IntSet(acc_list.data(), static_cast<int>(acc_list.size()));
        else
            acc_sets[i] = IntSet(IntSet::empty);
    }

    SSPSpace* model = new SSPSpace(n, m, N, S, min_feas,
                                   profit_int, pmin, pmax,
                                   ts_vec, acc_sets);

    Search::Options opts;
    Search::TimeStop* stop = nullptr;
    if (time_lim > 0.0) {
        stop      = new Search::TimeStop(static_cast<unsigned long long>(time_lim * 1000));
        opts.stop = stop;
    }

    BAB<SSPSpace> engine(model, opts);
    delete model;

    SSPSpace* best = nullptr;
    while (SSPSpace* sol = engine.next()) { delete best; best = sol; }

    bool timed_out = stop && engine.stopped();
    delete stop;

    GecodeResult res;
    res.solve_time_s = elapsed_s(t0);
    if (best) {
        res.objective = static_cast<double>(best->total_profit.val()) / SCALE;
        res.feasible  = true;
        res.optimal   = !timed_out;
        res.status    = timed_out ? "FEASIBLE" : "OPTIMAL";
        delete best;
    } else {
        res.objective = std::numeric_limits<double>::quiet_NaN();
        res.feasible  = false;
        res.optimal   = false;
        res.status    = timed_out ? "TIMEOUT" : "INFEASIBLE";
    }
    return res;
}
