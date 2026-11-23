# CCPO experiments

The CCPO campaign of the paper compares three arms of the same Gecode B&B on 120 feasible
test instances:

- **REF**: no Lagrangian bound; each node only gets the exact residual feasibility test (some
  scenario must still admit a feasible completion), solver mode `feas`;
- **LD**: Lagrangian bound with λ predicted by the GNN (`models/ccpo/`, epoch 57), solver mode `gnn`;
- **LD0**: the same bound evaluated at λ = 0, no network, solver mode `gnn_zero`.

Scripts are SLURM job scripts written for Trillium (Alliance); adapt `--account` and the GPU
request to another cluster.

```
test_paper/ccpo/
├── generate_test_instances.py   # the 120 test instances (deterministic)
├── instances_list.txt           # one line per instance: n N idx seed certificate (array task i = line i)
├── instances_md5.txt            # md5 of instance.txt and scenarios.txt of the instances used in the paper
├── campaign_config.env          # settings shared by the three arms
├── run_ld_array.sh              # LD or LD0 (MODE=gnn or MODE=gnn_zero), one array task per instance
├── run_ref_nodepack.sh          # REF, 8 instances per CPU node
├── parse_result.py              # solver log -> one CSV line
├── aggregate_results.py         # per-instance CSVs -> one CSV + REF / LD0 / LD table per cell
├── sanity_check.sh              # REF, LD0 and LD on one instance, compared with the paper results
├── selection/                   # how the 10 feasible instances per cell were chosen
├── gnn_vs_subgrad/              # root bound: one GNN call vs t* subgradient iterations (figure t*)
└── results/{ref,ld0,ld}/        # one CSV per instance (paper results)
```

## Test instances

n ∈ {8, 10, 12, 15} items × N ∈ {100, 500, 1000} scenarios, 10 feasible instances per cell.
Each instance is produced by `generators/ccpo/gen_test_episode.py`, the generator of the training
data with disjoint seeds (seed = 7·10⁹ + 100 000 n + 1000 N + idx; Q = round(2.5 n),
k = max(3, ⌊n/2⌋), L = 2, U = 8, γ = 1.25, ε = 0.1).

```bash
python3 test_paper/ccpo/generate_test_instances.py          # -> test_paper/ccpo/instances/ (not versioned)
(cd test_paper/ccpo/instances && md5sum -c --quiet ../instances_md5.txt)
```

**Selection** (`selection/`). A random instance may be infeasible (no portfolio meets the chance
constraint), so the indices were chosen as follows. For each cell, 40 candidates
(idx = 100, 110, …, 490) were screened by a 120 s REF run. A candidate is accepted when REF finds an
incumbent or the greedy solution is feasible, and rejected when infeasibility is proven. Undecided
candidates were run again with REF and FEAS for 1800 s each. The first 10 accepted candidates by
index form the cell. The last column of `instances_list.txt` gives the certificate: REF incumbent
within 120 s (90), REF within 1800 s (22), FEAS within 1800 s (5), greedy (3).
`selection/candidates.csv` has every candidate with its verdict, and `selection/select_instances.sh`
reruns the procedure. A rerun may select differently: verdicts depend on time limits, and the
current binary has the exact presolve, which the selection binary did not.

## Before running

1. Build with CUDA (top-level README) and generate the instances.
2. From the repository root:
   ```bash
   bash test_paper/ccpo/sanity_check.sh
   ```
   This solves n = 8, N = 100, idx 100 with REF, LD0 and LD (a few seconds, GNN forward on CPU)
   and checks status, optimum and node count against `results/`. REF and LD0 must match exactly. LD with a
   CPU forward may explore a different tree than with the GPU forward of the paper (λ can differ
   in the last bit), so only its optimum is checked strictly.
3. Submit from the repository root (`SLURM_SUBMIT_DIR` locates the repository; binaries run from
   there):
   ```bash
   sbatch --array=1-120 --export=ALL,MODE=gnn      test_paper/ccpo/run_ld_array.sh   # LD
   sbatch --array=1-120 --export=ALL,MODE=gnn_zero test_paper/ccpo/run_ld_array.sh   # LD0
   sbatch --array=1-15 test_paper/ccpo/run_ref_nodepack.sh                            # REF (CPU nodes)
   python3 test_paper/ccpo/aggregate_results.py "$SCRATCH/test_paper_ccpo/results"
   ```
   Outputs go to `$OUT_DIR`, by default `$SCRATCH/test_paper_ccpo/` (`logs/` with the full solver
   output, `results/{ref,ld0,ld}/` with one CSV per instance). An instance whose CSV exists is
   skipped, so a failed array can be resubmitted as is.

## Settings (`campaign_config.env`)

The three arms run the same binary (`build/solver_bab/bab_ccpo`) on the same instances, with the
same stopping rule. LD and LD0 run on a quarter of a Trillium GPU node (1 H100, 24 cores).
REF uses no GPU and runs on full Trillium CPU nodes, 8 instances at a time, each pinned to
24 physical cores. These nodes are about 10 % faster than the CPU part of a GPU node (LD0 on
8 instances of n = 8 to 12, same trees), so the comparison slightly favors REF. The feasibility
test of REF loops over the scenarios sequentially, so REF effectively uses one core.

- Depth-first B&B with exact presolve; nodes are evaluated in buckets of K = 32 (one GNN forward
  per bucket for LD, on the GPU).
- L(λ) is evaluated with OpenMP over the scenarios (24 threads). ζ is solved exactly, with SCIP for
  residual problems with at least 11 free items (budget 20 000 nodes). The residual θ of the node
  is included in ζ. Infeasibility is not inherited from parents.
- `OMP_WAIT_POLICY=passive`: busy-waiting OpenMP threads made times vary by up to 10× on shared
  GPU nodes; passive waiting makes them stable and equal to those on an idle node.
- REF uses the same ζ solver for its feasibility test, without SCIP (`CCPO_ZETA_SCIP=0`): ζ is
  only solved for feasibility there, with a zero cost.
- No solver time limit; runs are stopped at 23 h 50 (status `KILLED_WALL`), within the 24 h wall.

## Results

`results/ref/`, `results/ld0/` and `results/ld/` (120 runs each): one CSV line per run, columns
given by `python3 test_paper/ccpo/parse_result.py --header`; the `mode` column holds the solver mode
(`feas` for REF, `gnn_zero` for LD0, `gnn` for LD). LD and LD0 solve all 120 instances to
optimality; REF solves 116 and is stopped at 23 h 50 on four instances of n = 15, N = 1000
(idx 100, 110, 120, 140). All arms agree on the optimum of every instance they solve.
`aggregate_results.py` adds the first-level gap (bound at the children of the root vs the
optimum, LD and LD0) and prints, per cell, the time, nodes and gap of each arm (median by
default, `--stat mean` for the paper table), and the per-instance time ratios REF/LD and LD0/LD.

`gnn_vs_subgrad/` compares, at the root of the same 120 instances, one GNN bound call with the
subgradient method (iterations t* needed to reach the GNN bound, and their time); see its README.
