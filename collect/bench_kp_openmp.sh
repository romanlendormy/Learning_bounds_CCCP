#!/bin/bash
# collect/bench_kp_openmp.sh — Build isolé (build_omp) + validation + benchmark
# OpenMP sur la boucle N-scénarios de v1ld_solver.cpp (KPProblem, cas m=1).
#
# Arbre de build dédié (solver_generic/build_omp, solver_bab/build_omp),
# séparé de solver_generic/build (défaut, utilisé par solver_bab/build =
# binaires GPU de prod) et de solver_generic/build_cc / solver_bab/
# build_collect_cpu (utilisés par les jobs collect_kp_s10 en cours) — même
# logique d'isolation que collect/compile_collect_cpu.sh.
#
# Étapes : (1) build test_kp + bab_kp_gecode, (2) test_kp — valide
# solve_theta/solve_zeta (KPProblem/KPProblemDP, non touchés par ce patch),
# PAS solve_v1ld lui-même ; (3) une instance représentative (n=20,N=100,
# comme dans sweep_kp_gecode_grid du 10 juillet, où v1ld prenait ~108s à 1
# thread), résolue avec OMP_NUM_THREADS=1 puis =N_CPUS — c'est cette
# comparaison qui valide solve_v1ld après parallélisation : mêmes
# statut/obj/nœuds attendus (déterministe, pas de réduction flottante entre
# threads), seul le temps doit différer.
#
# Soumission :
#   sbatch collect/bench_kp_openmp.sh

#SBATCH --account=def-qcappart_cpu
#SBATCH --partition=cpubase_bycore_b4
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=bench_kp_openmp
#SBATCH --output=logs/bench_kp_openmp_%j.out
#SBATCH --error=logs/bench_kp_openmp_%j.err
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXT="$REPO/external"

module load StdEnv/2023
module load gcc/12.3
module load cmake/3.27.7
module load python/3.11
module load scipy-stack

GENERIC_BUILD="$REPO/solver_generic/build_omp"
BAB_BUILD="$REPO/solver_bab/build_omp"
mkdir -p "$GENERIC_BUILD" "$BAB_BUILD"

echo "======================================================================"
echo "  1. Build solver_generic (v1ld_generic_core + test_kp), arbre isolé"
echo "======================================================================"
# find_package(ortools) parcourt les deps vendorisées (abseil/protobuf/...)
# sur Lustre : ~10 min même en configure incrémental (cf. compile_cc.sh) —
# ici arbre neuf, donc ce coût est payé en entier, d'où --time=01:30:00.
cmake -S "$REPO/solver_generic" -B "$GENERIC_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$EXT/ortools"
cmake --build "$GENERIC_BUILD" --target test_kp -j"${SLURM_CPUS_PER_TASK:-16}"

echo ""
echo "======================================================================"
echo "  2. test_kp — correction/déterminisme (BB vs DP, exactitude zeta)"
echo "======================================================================"
"$GENERIC_BUILD/test_kp"

echo ""
echo "======================================================================"
echo "  3. Build solver_bab (bab_kp_gecode), pointant vers build_omp"
echo "======================================================================"
cmake -S "$REPO/solver_bab" -B "$BAB_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$EXT/ortools" \
    -DGECODE_PREFIX="$EXT/gecode" \
    -DSOLVER_GENERIC_BUILD="$GENERIC_BUILD"
cmake --build "$BAB_BUILD" --target bab_kp_gecode -j"${SLURM_CPUS_PER_TASK:-16}"

echo ""
echo "======================================================================"
echo "  4. Génération d'une instance représentative (n=20, N=100, seed=1,"
echo "     weakly_correlated — mêmes paramètres que sweep_kp_gecode_grid)"
echo "======================================================================"
INSTANCE_FILE=$(python3 - <<'PYEOF'
import sys
sys.path.insert(0, "collect")
from collect_kp import generate_song_tempfile
path = generate_song_tempfile(20, 1, 100, 1, alpha=0.5, R=1000,
                               correlation="weakly_correlated")
print(path)
PYEOF
)
echo "Instance : $INSTANCE_FILE"

BINARY="$BAB_BUILD/bab_kp_gecode"
export LD_LIBRARY_PATH="$EXT/gecode/lib:$EXT/ortools/lib:${LD_LIBRARY_PATH:-}"
# args: dataset eps T_node time_limit replication T_root mode
ARGS=(0.1 30 120 0 200 v1ld)

echo ""
echo "======================================================================"
echo "  5a. OMP_NUM_THREADS=1 (référence séquentielle)"
echo "======================================================================"
OMP_NUM_THREADS=1 "$BINARY" "$INSTANCE_FILE" "${ARGS[@]}"

echo ""
echo "======================================================================"
echo "  5b. OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} (parallèle)"
echo "======================================================================"
OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" "$BINARY" "$INSTANCE_FILE" "${ARGS[@]}"

rm -f "$INSTANCE_FILE"

echo ""
echo "======================================================================"
echo "  Terminé"
echo "======================================================================"
