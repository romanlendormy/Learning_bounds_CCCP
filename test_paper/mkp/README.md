# CCMKP experiments

All CCMKP experiments of the paper share one set of 120 test instances and one result
format. Scripts are SLURM job scripts written for Trillium (Alliance); adapt `--account`
and the GPU request to another cluster.

```
test_paper/mkp/
├── generate_test_instances.py   # the 120 test instances (deterministic)
├── instances_list.txt           # one line per instance: n N m seed file (array task i = line i)
├── instances_md5.txt            # md5 of the instances used in the paper
├── parse_result.py              # solver log -> one CSV line (all campaigns)
├── aggregate_results.sh         # per-instance CSVs -> one CSV with header
├── sanity_check.sh              # 4 modes on one instance, compared with the paper results
├── sld/                         # REF and LD with the SLD bound
├── dld/                         # LD with the DLD bound (full solve and root gap only)
├── lambda0/                     # LD0: SLD bound evaluated at λ = 0
├── sensitivity/                 # LD with 4 vs 24 CPU cores
└── gnn_vs_subgrad/              # root bound: one GNN forward vs t subgradient iterations
```

## Test instances

n ∈ {15, 20, 25, 30} items × N ∈ {100, 500, 1000} scenarios, m = 5 constraints,
seeds 43–52 (10 instances per cell). Same generator as the training data
(`generators/mkp/pisinger.py`: Pisinger weakly correlated, R = 1000,
α = 0.5, Song et al. scenarios). ε = 0.1 is a solver parameter.

```bash
python3 test_paper/mkp/generate_test_instances.py          # -> test_paper/mkp/instances/ (not versioned)
(cd test_paper/mkp/instances && md5sum -c --quiet ../instances_md5.txt)
```

## Before running

1. Build with CUDA (see the top-level README) and generate the instances.
2. From the repository root, on a login node, with the Python venv:
   ```bash
   bash test_paper/mkp/sanity_check.sh
   ```
   This solves n = 15, N = 100, seed 43 with REF, LD, LD0 and DLD (about 1.5 min on CPU) and checks
   status, optimum and node count against the paper results. The first GNN call also exports
   each model to TorchScript next to its checkpoint (`models/mkp/*/*scripted*.pt`): this needs
   torch and write access to the repository, which compute nodes on Trillium do not have
   (`/project` is read-only there).
3. Submit from the repository root (`SLURM_SUBMIT_DIR` locates the repository; binaries must
   run from there). Outputs go to `$OUT_DIR`, by default `$SCRATCH/test_paper_mkp/<campaign>/`
   (`logs/` with full solver output, `results/` with one CSV per instance). Copy `results/`
   into the matching folder below to update the versioned results.

## Campaigns

Common B&B settings: ε = 0.1, depth-first search, no solver time limit (the 24 h SLURM wall
is the only limit, except LD0: 23 h 50 in the solver), 4 OpenMP threads. LD runs (`gnn_batched`)
evaluate nodes in buckets of K = 64 on one GPU, with surrogate ζ, T_root = 200, T_node = 50.

| Folder | Script | Method | Hardware (paper) | Results |
|---|---|---|---|---|
| `sld/` | `run_ref_array.sh` | REF: Gecode B&B without Lagrangian bound | 1/4 GPU node (4 cores) | `ref_*.csv`: 105/120 solved; n = 30 cells N = 500 (5) and N = 1000 (10) exceed 24 h |
| `sld/` | `run_gnn_array.sh` | LD, SLD bound | 4 cores + H100 | `gnn_*.csv`: 120/120 optimal |
| `dld/` | `run_dld_array.sh` | LD, DLD bound | 4 cores + H100 / H200 | `results/`: 95/120 optimal; n = 25 N = 1000 (10), n = 30 N = 500 (5) and N = 1000 (10) did not finish (24 h wall or GPU memory) |
| `dld/` | `run_dld_array.sh` with `FIRST_LEVEL_ONLY=1` | DLD bound at the first level of the tree only | as above | `results_firstlevel/`: 89/120 |
| `lambda0/` | `run_lambda0_nodepack.sh` | LD0: same as LD (SLD), λ = 0 at every node (`GNN_ZERO_LAMBDA=1`, no GNN call) | full CPU nodes, 48 instances × 4 cores | 116 optimal, 4 feasible at the time limit (n = 30, N = 1000) |
| `sensitivity/` | `run_sensitivity_array.sh` | LD (SLD) with 4 and 24 cores on 15 instances | as LD | `results/c4`, `results/c24`: 15 + 15 optimal |
| `gnn_vs_subgrad/` | `run_gnn_vs_subgrad_n15.sh`, `run_gnn_vs_subgrad_multi_n.sh` | root bound from one GNN forward vs cold-start subgradient (t*) | 4 cores + H100 | `results_gnn_vs_subgrad_*.csv`, `latex_data/`, figures |

Notes:
- DLD: from n = 20, N = 1000 onwards the batched DLD graphs need more than 80 GB of GPU memory;
  these instances ran on H200 (column `gpu_model`).
- Sensitivity: the 9 DLD tasks of `tasks.txt` failed in the paper campaign (the binary used then
  could not find its export script after a renaming). Only the 15 SLD tasks have results.
  The script itself runs both.
- LD0 runs on CPU only. Node counts and gaps compare directly with LD; times compare up to the
  CPU generation (EPYC 9655 vs EPYC 9654 on GPU nodes).

### Result format

One CSV line per instance (`parse_result.py --header` for the column names): instance
(n, N, m, seed), mode, status, objective, greedy value, nodes, fails, incumbent improvements,
total and B&B times, bound-time breakdown (graph build, GNN forward, ζ), first-level bound and root
gap, bucket statistics, GPU model and K. Aggregate a folder with:

```bash
bash test_paper/mkp/aggregate_results.sh test_paper/mkp/sld/results gnn_ > ld_sld.csv
```

The versioned CSVs were re-parsed from the original logs with the current `parse_result.py`
(identical values; `gpu_model` and `K` filled in everywhere).

### GNN vs subgradient (t*)

- n = 15: the subgradient runs to convergence (T_root = 3000) and logs its trace;
  `parse_gnn_vs_subgrad.py` computes t* = min{t : bound(t) ≤ GNN bound}.
- n = 20, 25, 30: the subgradient stops as soon as it beats the GNN bound
  (`GNN_COMPARE_BOUND`, cap 300 iterations), so t* is the number of iterations run.
- Post-processing, from the repository root:
  ```bash
  P=test_paper/mkp/gnn_vs_subgrad
  python3 $P/parse_gnn_vs_subgrad.py $SCRATCH/test_paper_mkp/gnn_vs_subgrad_n15 15 \
      --csv=$P/results_gnn_vs_subgrad_summary.csv --detail-csv=$P/results_gnn_vs_subgrad_detail.csv
  for n in 20 25 30; do
    python3 $P/parse_gnn_vs_subgrad.py $SCRATCH/test_paper_mkp/gnn_vs_subgrad_multin $n \
        --csv=$P/results_gnn_vs_subgrad_summary.csv --detail-csv=$P/results_gnn_vs_subgrad_detail.csv
  done
  python3 $P/extract_time_vs_tstar.py      # -> latex_data/time_vs_tstar_*.csv
  ```
  `parse_gnn_vs_subgrad.py` appends to its CSVs: remove them first to rebuild from scratch.
  Gaps use the optimum proven by REF, or by LD (SLD) for the 15 instances REF did not solve.
- `fig_time_vs_tstar.tex` and `fig_tstar_sld.tex` (pgfplots) read `latex_data/`;
  `fig_tstar_sld_preview.tex` compiles the latter standalone.
