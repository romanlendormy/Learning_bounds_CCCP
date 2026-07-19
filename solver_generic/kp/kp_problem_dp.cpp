#include "kp_problem_dp.hpp"
#include "mkp_subproblems.hpp"

#include <algorithm>
#include <cmath>

KPProblemDP::KPProblemDP(Vec1d c, Vec1d p, Vec2d A, Vec1d b, double epsilon)
    : KPProblem(std::move(c), std::move(p), std::move(A), std::move(b), epsilon)
{
    B_max_ = 0;
    for (int i = 0; i < N_; ++i)
        B_max_ = std::max(B_max_, static_cast<int>(std::round(b_[i])));

    // Ne réalloue jamais si toutes les capacités ≤ B_max_.
    dp_buf_.reserve(static_cast<size_t>(n_ + 1) * (B_max_ + 1));
}

std::pair<double, Vec1d> KPProblemDP::solve_zeta(int i, const Vec1d& cost) const
{
    int n = n_;
    Vec1d x(n, 0.0);
    double val_free = 0.0;

    const Vec1d& a_row = A_[i];
    const double  b_i  = b_[i];

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

    int B = std::max(0, static_cast<int>(std::round(b_i)));

    std::vector<int> w(ni);
    for (int ii = 0; ii < ni; ++ii)
        w[ii] = static_cast<int>(std::round(a_row[items[ii]]));

    // Buffer avec stride = B+1 (compact, cache-friendly).
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
