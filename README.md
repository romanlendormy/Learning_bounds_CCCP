# Learning bounds for CCCP

Research project on learning Lagrangian dual bounds for Chance-Constrained Combinatorial Problems (CCCP), with a focus on the Multidimensional Knapsack (MKP), Set Cover (covset), and Stochastic Shortest Path (SSP) problems.

## Project structure

```
.
├── data/
│   ├── covset/          # CCSP instances (covering set)
│   ├── knapsack/        # CCMKP instances (OR-Library + Song et al.)
│   └── ssp/             # CCSSP instances (Pesant automata)
│
├── solver_generic/      # Generic v1LD / v1CLD solver (C++ + Python bindings)
│   ├── v1ld_solver.*    # Core subgradient loop for v1LD
│   ├── v1cld_solver.*   # Core subgradient loop for v1CLD
│   ├── mkp/             # MKP subproblems (B&B exact, LP relaxation)
│   ├── ssp/             # SSP subproblems (DP automata) + Python loader
│   └── covset/          # Covset subproblems (greedy + bitmask DP) + Python loader
│
└── gecode_base/         # Gecode CP solver bindings (Python)
```

## Solvers

The project implements two Lagrangian decomposition bounds for chance-constrained problems:

- **v1LD** (`V1LDProblem`): one multiplier λ_i per scenario, ζ_i solves the full feasible region per scenario.
- **v1CLD** (`V1CLDProblem`): one multiplier λ_{i,j} per (scenario, constraint) pair, ζ_{i,j} solves a single-constraint subproblem.

Both bounds are upper bounds on the true optimal value v*.

### Building the C++ module

```bash
cd solver_generic/build
cmake .. && make -j4
```

The compiled module `v1ld_generic_cpp.so` is copied to the project root.

### Usage (Python)

```python
import v1ld_generic_cpp as cpp
from solver_generic.mkp import ...   # load MKP instances
from solver_generic.ssp import load_ssp_instances, ssp_to_cpp
from solver_generic.covset import load_covset_instances, covset_to_cpp_v1ld

# MKP — v1LD (B&B exact subproblem)
prob = cpp.MKPProblem(c, p, A, b, epsilon)
res  = cpp.solve_v1ld(prob, T=500)

# Covset — v1CLD (greedy 1-constraint subproblem)
prob = cpp.CovsetCLDProblem(c, p, A, b, epsilon)
res  = cpp.solve_v1cld(prob, T=500)
```

## References

- Ahmed, S. et al. (2017). *Nonanticipative duality, relaxations, and formulations for chance-constrained stochastic programs.*
- Song, Y. et al. Chance-constrained binary packing problems.
- Pesant, G. (2004). A regular language membership constraint for finite sequences of variables.
