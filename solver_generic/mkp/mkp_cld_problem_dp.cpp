#include "mkp_cld_problem_dp.hpp"
#include "mkp_subproblems.hpp"

#include <algorithm>
#include <cmath>

// SCALE=10000 — même valeur que MKPSharedData::SCALE (solver_bab/mkp/gecode_bab.hpp:162),
// dupliquée ici car solver_generic ne dépend pas de solver_bab. Nécessaire pour
// que la DP indexe sur des poids/capacités entiers SANS perte : les instances
// sont générées avec round(v, 4) (generate_song_tempfile), donc tout multiple de
// 10000 est exact. Un round() brut sur le float (sans ce scaling, comme avant)
// tronquait les décimales et pouvait faire diverger ζ_{i,j} de la valeur exacte
// du B&B.
static constexpr int SCALE = 10000;

MKPCLDProblemDP::MKPCLDProblemDP(Vec1d c, Vec1d p, Vec3d A, Vec2d b, double epsilon)
    : MKPCLDProblem(std::move(c), std::move(p), std::move(A), std::move(b), epsilon)
{
    // Pré-alloue pour la plus grande table (ni+1)×(B+1) possible,
    // i.e., (n+1) × (W_max+1), en utilisant stride = W_max+1.
    // On utilisera ensuite stride = B+1 au moment de l'appel pour éviter
    // les cache-misses quand B << W_max.
    // ATTENTION : avec SCALE=10000, W_max_ peut devenir énorme sur des
    // capacités b_j réalistes (ex. b_j~3771 -> W_max_~3.77e7) — la table
    // dp_buf_ ((n+1)×(W_max_+1) doubles) peut alors atteindre plusieurs Go.
    // Risque de complexité assumé, à mesurer par bench_v1cld_subproblems —
    // pas corrigé ici (le but est justement de le quantifier).
    W_max_ = 0;
    for (int i = 0; i < N_; ++i)
        for (int j = 0; j < m_; ++j)
            // floor, pas round : même convention que b_int dans MKPSharedData
            // (gecode_bab.hpp:350) — ne jamais SURESTIMER la capacité scalée.
            W_max_ = std::max(W_max_, static_cast<int>(std::floor(b_[i][j] * SCALE)));

    // Réserve le buffer pour la plus grande table possible.
    // La réallocation n'aura jamais lieu si toutes les capacités ≤ W_max_.
    dp_buf_.reserve(static_cast<size_t>(n_ + 1) * (W_max_ + 1));
}

std::pair<double, Vec1d>
MKPCLDProblemDP::solve_zeta(int i, int j, const Vec1d& cost) const
{
    int n = n_;
    Vec1d x(n, 0.0);
    double val_free = 0.0;

    const Vec1d& a_row = A_[i][j];
    const double  b_j  = b_[i][j];

    // Items à poids nul : inclus si coût > 0. Items à coût ≤ 0 : exclus.
    std::vector<int> items;
    items.reserve(n);
    for (int k = 0; k < n; ++k) {
        if (a_row[k] < 1e-9) {
            if (cost[k] > 0.0) { x[k] = 1.0; val_free += cost[k]; }
        } else if (cost[k] > 0.0) {
            items.push_back(k);
        }
    }

    int ni = static_cast<int>(items.size());
    if (ni == 0) return {val_free, std::move(x)};

    int B = std::max(0, static_cast<int>(std::floor(b_j * SCALE)));

    std::vector<int> w(ni);
    for (int ii = 0; ii < ni; ++ii)
        w[ii] = static_cast<int>(std::llround(a_row[items[ii]] * SCALE));

    // Buffer avec stride = B+1 (compact, cache-friendly).
    // Ne réalloue que si la table courante est plus grande que toutes
    // les précédentes (stratégie high-watermark).
    const int stride  = B + 1;
    const int needed  = (ni + 1) * stride;
    if (static_cast<int>(dp_buf_.size()) < needed)
        dp_buf_.resize(needed);
    std::fill(dp_buf_.begin(), dp_buf_.begin() + needed, 0.0);

    // DP : dp_buf_[row * stride + c]
    for (int ii = 0; ii < ni; ++ii) {
        const double pi = cost[items[ii]];
        const int    wi = w[ii];
        double* cur  = dp_buf_.data() + (ii + 1) * stride;
        double* prev = dp_buf_.data() +  ii      * stride;
        for (int c = 0; c <= B; ++c) {
            cur[c] = prev[c];
            if (wi <= c) {
                double cand = prev[c - wi] + pi;
                if (cand > cur[c]) cur[c] = cand;
            }
        }
    }

    // Reconstruction par backtracking
    int cap = B;
    for (int ii = ni - 1; ii >= 0; --ii) {
        if (w[ii] <= cap) {
            double* prev = dp_buf_.data() + ii * stride;
            if (prev[cap - w[ii]] + cost[items[ii]] > prev[cap] + 1e-9) {
                x[items[ii]] = 1.0;
                cap -= w[ii];
            }
        }
    }

    return {val_free + dp_buf_[ni * stride + B], std::move(x)};
}
