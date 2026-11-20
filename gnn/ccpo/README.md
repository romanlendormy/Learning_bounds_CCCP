# gnn/ccpo — GNN for the CCPO Lagrangian bound

The GNN predicts the multipliers λ of the scenario Lagrangian bound at a node of the CCPO
branch-and-bound. Training minimises −L(λ): the bound is evaluated by the C++ code of the solver
(`ccpo_lagrangian.so`, through ctypes), and its subgradient serves as the gradient (envelope theorem).

```
ccpo_dataset.py              # parse collected nodes, rebuild the residual subproblem at a node
ccpo_graph.py                # GNN input graph (node / context features, edges, CCMKP-style edge weights)
ccpo_gnn_model.py            # model (eager)
ccpo_lagrangian_autograd.py  # L(λ) and subgradient via the C++ library, as a torch autograd function
train_ccpo_gnn.py            # training (stratified selection, supervised warm-up, checkpoints, VAL metrics)
run_train_ccpo.sh            # SLURM launcher with the settings of the paper model
find_infeasible_nodes.py     # exact labelling of nodes whose subtree has no solution
ccpo_bruteforce.py           # enumeration of the free-variable assignments of a node
ccpo_gnn_scriptable_batched.py  # TorchScript export used by the B&B (bab_ccpo, mode gnn)
validate_export_ckpt.py      # export vs eager model on real nodes
presolve_box_exact.py        # Python reference of the certified exact presolve
ccpo_graph_cpp_bind.py, ccpo_libs.py  # ctypes bindings / location of the C++ libraries
test_*.py                    # Python / C++ / TorchScript correspondence, presolve, bound validity
```

The C++ libraries (`ccpo_lagrangian.so`, `presolve_exact.so`, `ccpo_graph.so`) are built with the
rest of the repository (`cmake --build build`) and loaded from `$BUILD_DIR` (default `build/`).

## Data and training pipeline (paper model `models/ccpo/`)

All steps run from the repository root, with the Python venv; CPU only.

1. **Collection** — B&B REF nodes on n ∈ {6, 8, 10} × N ∈ {20, 50, 100}, up to 300 nodes per
   instance solved to optimality (10 000 instances per cell for n = 6, 8; 3 000 / 3 000 / 1 500 for n = 10):
   ```bash
   sbatch --export=ALL,OUT_DIR=$SCRATCH/ccpo/collect collect/ccpo/run_collect_ccpo.sh
   ```
2. **Curation** — keep nodes with at least ⌈(1 − ε) N⌉ scenarios feasible at λ = 0, per cell:
   ```bash
   python3 collect/ccpo/curate_feasible_examples.py --data_dir $SCRATCH/ccpo/collect \
       --target_per_cell 1056 --root_cap_per_cell 56 --seed 0 --only_cell 6,20   # one call per cell
   ```
3. **Exact scenario flags** — recompute the presolve flags with the certified exact presolve
   (copy, the source is untouched), then check the copy:
   ```bash
   python3 collect/ccpo/recompute_presolve_flags.py --src $SCRATCH/ccpo/collect --dst $SCRATCH/ccpo/data --procs 96
   python3 collect/ccpo/verify_presolve_flags.py --src $SCRATCH/ccpo/collect --dst $SCRATCH/ccpo/data
   ```
4. **Infeasible subtrees** — label the 9 999 selected nodes by brute force (1 784 have no solution;
   they are excluded from the training loss):
   ```bash
   python3 gnn/ccpo/find_infeasible_nodes.py --data_dir $SCRATCH/ccpo/data \
       --out $SCRATCH/ccpo/node_labels.jsonl --procs 96 --cap 5000000
   ```
5. **Extra feasible nodes** — top the training set up to ~10 000 feasible nodes (3 431 added),
   VAL unchanged:
   ```bash
   python3 collect/ccpo/build_extra_feasible_nodes.py --data_dir $SCRATCH/ccpo/data \
       --pool_dir $SCRATCH/ccpo/collect --labels $SCRATCH/ccpo/node_labels.jsonl \
       --out_dir $SCRATCH/ccpo/extra --target_train_per_cell 1112 --procs 96
   ```
6. **Training** — 200 epochs, ~9 h on one CPU node:
   ```bash
   sbatch --export=ALL,DATA_DIR=$SCRATCH/ccpo/data,LABELS=$SCRATCH/ccpo/node_labels.jsonl,EXTRA=$SCRATCH/ccpo/extra,OUT_DIR=$SCRATCH/ccpo/train \
       gnn/ccpo/run_train_ccpo.sh
   ```
   The paper model is epoch 57, the best mean VAL improvement over L(0) on feasible nodes
   (10.45 % mean, 6.81 % median).
7. **Export** — TorchScript module read by `bab_ccpo`, then check it against the eager model:
   ```bash
   python3 gnn/ccpo/ccpo_gnn_scriptable_batched.py --checkpoint <ckpt> --out <out.scripted_batched.pt> --device cpu
   python3 gnn/ccpo/validate_export_ckpt.py <ckpt> <out.scripted_batched.pt> --data_dir $SCRATCH/ccpo/data
   ```

## Tests

```bash
python3 gnn/ccpo/test_presolve_box_exact.py            # Python exact presolve vs enumeration of active sets
python3 gnn/ccpo/test_presolve_exact_cpp.py $SCRATCH/ccpo/data 200   # C++ vs Python presolve (random cases + collected data)
python3 gnn/ccpo/test_lagrangian_majorant_validity.py  # bound validity with ζ upper / lower estimates (brute force)
python3 gnn/ccpo/test_correspondence.py --data_dir $SCRATCH/ccpo/data   # graph Python vs C++, eager vs TorchScript
```
