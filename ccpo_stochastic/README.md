# ccpo_stochastic — CCPO instance generator

Generates instances of the chance-constrained portfolio optimisation problem (CCPO):

```
max  c' q
s.t. Σ_i q_i = Q,   q_i ∈ {0} ∪ {L, …, U},   |{i : q_i > 0}| ≤ k
     |{j : q' Σ_j q ≤ B}| ≥ ⌈(1 − ε) N⌉          (N covariance scenarios Σ_j)
```

i.e. integer lots, bounded cardinality, and a risk budget that must hold in at least a
fraction 1 − ε of the scenarios. This package only generates data. The solver is in
`solver_bab/ccpo/`, and the episode generation used for training and testing is in `collect/`.

```
instance.py          # mother instance: returns c, one-factor covariance Σ
scenarios.py         # N covariance scenarios Σ_j by perturbing the factors
discretize.py        # integer instance (c_int, Σ_int, B) + blocking checks
writers.py           # text output: ep_XXXXXX/{instance,scenarios,meta}.txt + manifest.txt
readers.py           # independent reader of the text format
txt_format.py        # ε as an exact fraction, SAA threshold
seeding.py           # hierarchical seeds (SeedSequence)
diagnose_epsilon.py  # optimal value vs ε by exhaustive enumeration (calibration tool)
tests/               # pytest suite
```

## Mother instance (`instance.py`)

One-factor model, parameterised by the fraction of variance explained h_i:

```
β_i = σ_i √h_i / σ_f,     ψ_i = (1 − h_i) σ_i²,     Σ = σ_f² ββ' + diag(ψ)
```

ψ_i > 0 holds by construction (h_i < 1). σ_f = 0.0335 is fixed (not identifiable
separately from β). It is chosen so that the mean correlation ρ̄ ≈ h̄ = 0.532 with mean
volatility σ̄ = 0.046.

| Draw | Distribution | Parameters |
|---|---|---|
| σ_i | truncated log-normal | median 0.045, shape 0.15, [0.036, 0.069] |
| h_i | Beta | (5.0, 4.4), mean 0.532 |
| c_i | truncated log-normal | median 2.5e-3, shape 0.85, [1e-4, 1.2e-2] |

An optional rank correlation between c_i and h_i (`capm_alignment`, Iman–Conover) is
available; it is 0 in all experiments.

## Scenarios (`scenarios.py`)

The factors (β, ψ, σ_f) are perturbed, never Σ itself, so every Σ_j is PSD by construction:

| Stage | Perturbation | Default |
|---|---|---|
| (a) | β_j = β (1 + δ), δ ~ N(0, noise_beta²) | noise_beta = 0.10 |
| (b) | ψ_j = ψ exp(η), η ~ N(0, noise_psi²) | noise_psi = 0.10 |
| (c) | σ_f,j = σ_f exp(ζ_j), one ζ_j per scenario (market regime) | noise_market = 0.15 |
| (d) | ψ_j,i × κ with probability p_shock (idiosyncratic shocks) | p_shock = 0.05, κ = 5 |

Stage (c) correlates the scenarios: without it, the SAA constraint is nearly trivial.
Stage (d) gives a heavy tail: without it, ε only cuts a Gaussian tail.
`tests/test_scenarios.py` checks that each stage strictly increases the dispersion of q'Σ_j q.

## Integer discretisation (`discretize.py`)

```
c_int     = ⌊S_c c⌋
Σ_int[j]  = ⌊S Σ_j⌋                         (rounding down: the risk constraint is relaxed, never tightened)
B         = ⌊S γ Σ̄⌋ Q²                     (Σ̄ = mean entry of the MOTHER Σ: one budget for all scenarios)
psi_int[j] = diag(Σ_int[j]) − ⌊S σ_f,j² β_j²⌋   (exact integer rank-1 + diagonal decomposition)
```

Defaults: S = S_c = 10⁶, γ = 1.25 (see Calibration).
Blocking checks (an exception is raised on violation):
1. Σ_j PSD before rounding.
2. ψ > 0 and psi_int > 0.
3. Σ_int symmetric, with a positive diagonal.
4. Structural feasibility: k·L ≤ Q ≤ k·U.
5. No overflow: max|Σ_int| · Q² < 2³⁰. With S = 10⁶ this caps Q at about 470.

Episode generation in `collect/` skips an instance that fails a check.

## Text format (`writers.py`, `readers.py`)

One folder per episode, plus one line per episode in `manifest.txt`:

```
ep_XXXXXX/instance.txt    line 1: n N k Q L U eps_num eps_den
                          line 2: B
                          line 3: c_int (n values)
ep_XXXXXX/scenarios.txt   per scenario j = 1..N: "j", then psi_int[j] (n values),
                          then the upper triangle of Σ_int[j] as "row col value" (1-based)
ep_XXXXXX/meta.txt        key = value (generator settings, seed, statistics)
```

ε is stored as an exact fraction, so the solver computes the SAA threshold
⌈(eps_den − eps_num) N / eps_den⌉ in integers. `scenarios.txt` can be gzipped (`mtime=0`,
so the output stays byte-identical).

## Reproducibility (`seeding.py`)

`episode_rngs(seed_master, ep_id, n_episodes)` derives two independent generators per episode
from `SeedSequence(seed_master).spawn(n_episodes)[ep_id]`. One draws the mother instance, the
other the scenarios. No global numpy seed is used. The episode scripts of the paper
(`collect/`) seed the two generators directly with `default_rng(seed)` and `default_rng(seed + 1)`.

## Calibration

`diagnose_epsilon.py` solves small episodes by exhaustive enumeration for several ε:

```bash
python -m ccpo_stochastic.diagnose_epsilon --n 15 --N 50 --Q 20 --k 5 --L 2 --U 8 \
    --epsilons 0.0 0.05 0.1 0.2 0.3 --gamma 1.5
```

Optimal value vs ε on n = 15, N = 50 (one seed):

| γ | ε = 0 | 0.05 | 0.1 | 0.2 | 0.3 |
|---|---|---|---|---|---|
| 1.2 | infeasible | infeasible | 130971 | 143027 | 158876 |
| 1.5 | 125432 | 144140 | 158876 | 158876 | 158876 |
| 2.0 | 156902 | 158876 | 158876 | 158876 | 158876 |

158876 is the optimum without the risk constraint. γ was then chosen once for the whole
experimental grid, so that the GNN trains on a single distribution. Two quantities were
measured: the share of structurally valid portfolios that also satisfy the SAA constraint
(%feas), and the relative gap between the risk-free optimum and the optimum with the
risk constraint.
They were computed by exhaustive enumeration for n = 15, 20, and by Monte Carlo sampling for
n = 30 (one seed per cell):

| n | N | γ = 1.20 (%feas / gap) | γ = 1.25 (%feas / gap) |
|---|---|---|---|
| 15 | 50 | 0.11% / 17.6% | 0.53% / 13.7% |
| 15 | 100 | 0.39% / 17.2% | 1.82% / 12.8% |
| 15 | 500 | infeasible | 0.009% / 31.0% |
| 15 | 1000 | infeasible | 0.015% / 32.1% |
| 20 | 50 | 2.38% / 21.0% | 5.16% / 17.7% |
| 20 | 100 | 0.24% / 33.3% | 0.84% / 27.9% |
| 20 | 500 | 0.15% / 35.4% | 0.60% / 29.5% |
| 20 | 1000 | 0.085% / 38.1% | 0.40% / 31.2% |
| 30 | 50 | 1.29% / 7.6% | 2.68% / 6.3% |
| 30 | 100 | 0.17% / 15.2% | 0.60% / 11.3% |
| 30 | 500 | 0.056% / 28.1% | 0.26% / 14.4% |
| 30 | 1000 | 0.105% / 20.1% | 0.42% / 13.6% |

γ = 1.20 makes n = 15, N ≥ 500 infeasible. γ = 1.25 is feasible on every cell, with a
substantial gap: it is the default (`GAMMA_DEFAULT`).

## Tests

```bash
python -m pytest ccpo_stochastic        # 54 tests, < 1 s
```

They cover determinism (gzip included), round-trip through the independent reader,
PSD over 1000 scenarios, target statistics of the mother instance (ρ̄ ∈ [0.45, 0.60],
σ̄ ∈ [0.042, 0.050], Σ̄ ∈ [9e-4, 1.4e-3]), dispersion added by each perturbation stage,
the blocking checks, and `diagnose_epsilon.py` on small sizes.
