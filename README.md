# Learning Lagrangian bounds for chance-constrained combinatorial problems

Code for learning Lagrangian dual bounds with graph neural networks (GNNs) and using them
inside an exact branch-and-bound (Gecode), on two chance-constrained problems:

- **CCMKP**: chance-constrained multidimensional knapsack (scenario-based, linear constraints);
- **CCPO**: chance-constrained portfolio optimisation with quadratic (variance) risk constraints per scenario.

For each problem, the bound is a scenario Lagrangian decomposition. At every node of the
B&B, the GNN predicts the multipliers λ, which warm-start (or replace) the subgradient method.
The baseline **LD0** runs the same B&B with λ initialised at 0, and **REF** is the B&B without
Lagrangian bound.

## Repository layout

```
.
├── env/                  # environment: module list, GPU runtime paths (Trillium / Alliance clusters)
├── solver_generic/       # SLD / DLD subgradient solvers (C++), MKP subproblems,
│                         #   CCPO Lagrangian bound (θ / ζ subproblems, exact presolve, SCIP for ζ)
├── solver_bab/           # Gecode branch-and-bound with Lagrangian bounds
│   ├── mkp/              #   CCMKP: SLD / DLD bounds, GNN inference (LibTorch), node collector
│   ├── ccpo/             #   CCPO: REF / SLD / FEAS B&B, batched LD / LD0, GNN inference, node collector
│   └── nvshmem_stub/     #   stub library needed by LibTorch 2.13 CUDA on clusters without NVSHMEM
├── generators/           # instance generators
│   ├── mkp/              #   CCMKP: Pisinger instances + Song et al. scenarios
│   └── ccpo/             #   CCPO: mother instance, scenarios, discretisation, writers + tests
├── collect/              # B&B nodes used as GNN training data (C++ collectors in solver_bab/)
│   ├── mkp/              #   CCMKP: collection with 10 % node sampling
│   └── ccpo/             #   CCPO: collection, curation, exact presolve flags, extra feasible nodes
├── gnn/                  # GNNs predicting λ: training and TorchScript export
│   ├── mkp/              #   CCMKP (SLD / DLD). B&B binaries must run from the repository root:
│   │                     #   they call gnn/mkp/*_scriptable*.py to export a checkpoint the first time
│   └── ccpo/             #   CCPO, with export validation and tests
├── models/               # trained models used in the paper (mkp/sld, mkp/dld, ccpo)
└── test_paper/           # experimental campaigns of the paper (scripts + aggregated results)
    ├── mkp/
    └── ccpo/
```

## Dependencies

| Component | Version used | Used by |
|---|---|---|
| GCC | 12.3 | all C++ |
| CMake | ≥ 3.27 | build |
| Gecode | 6.2.0 | B&B solvers |
| OR-Tools | 9.6.2534 | `solver_generic` |
| SCIP | 9.0.0 (shared library, module `scipoptsuite/9.0.0`) | CCPO ζ subproblems |
| LibTorch | 2.13.0 (CPU, or CUDA 12.6) | GNN inference inside the B&B |
| CUDA / cuDNN | 12.6 / 9.10 | GPU inference |
| Python | 3.11 | training, data collection, analysis |

Python packages are listed in `requirements.txt`.

Gecode, OR-Tools and LibTorch are expected under `external/` (not versioned):

```
external/gecode/           # Gecode 6.2.0 install prefix
external/ortools/          # OR-Tools 9.6.2534 install prefix
external/libtorch/         # LibTorch 2.13.0 CPU
external/libtorch-cuda/    # LibTorch 2.13.0 + CUDA 12.6
```

## Building

```bash
source env/modules_trillium.sh
module load cuda/12.6 cudnn/9.10.0.56      # only for a CUDA build
cmake -S . -B build -DGNN_DEVICE=cuda      # or cpu (default)
cmake --build build -j
```

## Environment (Trillium)

```bash
source env/modules_trillium.sh            # compilers, SCIP, Python
python -m venv ~/venvs/cccp && source ~/venvs/cccp/bin/activate
pip install --no-index -r requirements.txt
source env/setup_gpu_env.sh               # before running a CUDA-linked B&B binary
```

## Reproducing the experiments

Each campaign directory under `test_paper/` has its own README describing the instance grid,
the SLURM scripts, the exact settings and how results are aggregated into the paper tables.
