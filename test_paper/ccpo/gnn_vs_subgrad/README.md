# CCPO: one GNN call vs subgradient at the root (t*)

For each of the 120 test instances (`../instances_list.txt`), this measures how many subgradient iterations t*
are needed to reach the bound that the GNN gives in one call, and how long those t* iterations take compared with
that call. Everything is done on the residual problem of the root, after the CP propagation of the root.

## Protocol (`solver_bab/ccpo/main_root_compare_ccpo.cpp`, binary `bab_ccpo_rootcmp`)

There are two phases, so that a GPU is held only while the forward passes run.

1. **GNN phase** (`run_gnn.sh`, one GPU job for the 120 instances). λ is predicted by the model of the campaign
   (`models/ccpo/`, epoch 57, CUDA forward), using the graph and forward path of a node of the batched B&B with a
   batch of size 1. L(λ) is then evaluated exactly. A warm-up call is excluded, and times are medians over
   `REPS` = 5 calls. λ is written to `$OUT/lambda/`.
2. **Subgradient phase** (`run_subgrad_nodepack.sh`, CPU nodes, 8 instances × 24 physical cores per node).
   L(λ_GNN) is evaluated again on the subgradient's CPU; the bound must match phase 1. The subgradient then
   starts cold (λ = 0, step 1/√(t+1), best iterate, the same as the solver's `solve_ccpo_sld`) and stops as soon
   as its bound is at most L(λ_GNN). t* and its time are therefore measured, not extrapolated. Runs are capped
   at `T_MAX` = 1000 iterations or `TIME_LIM` = 3600 s; t* is censored beyond that.
3. ζ uses the same settings as the campaign (`../campaign_config.env`: SCIP for residual problems with at least
   11 free items, residual θ, no inheritance, passive OpenMP waiting), evaluated with OpenMP over the scenarios
   (24 threads) in both phases.
4. **Time of one GNN call** = graph construction + forward pass (phase 1, GPU) + evaluation of L(λ_GNN)
   (phase 2, on the same CPU as the subgradient). The evaluation dominates this time. Measuring it on the
   subgradient's CPU avoids comparing two different processors.

## Running (from the repository root)

```bash
# instances generated in test_paper/ccpo/instances/ (or INSTANCES_DIR=...); CUDA build
sbatch -o "$SCRATCH/slurm_tstar_gnn_%j.out" test_paper/ccpo/gnn_vs_subgrad/run_gnn.sh                   # GPU login node
sbatch --array=1-15 -o "$SCRATCH/slurm_tstar_sub_%A_%a.out" test_paper/ccpo/gnn_vs_subgrad/run_subgrad_nodepack.sh   # CPU login node, afterwards
python3 test_paper/ccpo/gnn_vs_subgrad/aggregate.py $SCRATCH/test_paper_ccpo_tstar/results_gnn $SCRATCH/test_paper_ccpo_tstar/results_subgrad
```

- Outputs go to `$OUT_DIR`, by default `$SCRATCH/test_paper_ccpo_tstar/`:
  - `results_gnn/`, `results_subgrad/`: one CSV per instance;
  - `logs_subgrad/`: the full trace `SUBGRAD_TRACE,<t>,<best bound>,<elapsed s>`;
  - `lambda/`.
- Instances already done are skipped.
- The GNN phase fits in Trillium's `debug` partition (under 1 h). The subgradient phase takes under 10 min per node.
- `aggregate.py` writes:
  - `results.csv`, next to `results_subgrad/`;
  - `latex_data/time_vs_tstar_ccpo.csv`, with medians per (n, N);
  - `fig_tstar_ccpo.tex`, a pgfplots figure;
  - a list of censored instances, with their remaining gap to the GNN bound.

  The preview `fig_tstar_ccpo_preview.tex` compiles with `pdflatex`.

## Results

`results.csv` holds the 120 instances of the paper, and `fig_tstar_ccpo.tex` / `fig_tstar_ccpo_preview.pdf` show
the figure.

- 111 instances reach the GNN bound. The other 9, all with N = 100, are stopped at 1000 iterations while still
  improving, with a remaining gap of 0.1 to 6.7 %. They count as +∞ in the medians, which remain exact.
- For n ≤ 12, median t* ranges from 37 to 784 iterations, and one GNN call is 60 to 1600 times faster than
  those iterations.
- At n = 15, the GNN root bound is only about 1 % below L(0). The subgradient reaches it in a median of 4 to 6
  iterations, so one GNN call is only 5 to 13 times faster.
