#!/bin/bash
# build_bench_v1cld.sh — Rebuild incrémental de v1ld_generic_core (contient
# le fix de scaling de mkp_cld_problem_dp.cpp) puis build de la nouvelle
# cible bench_v1cld_subproblems (solver_bab/build). CPU seul, pas de GPU
# nécessaire (aucune dépendance Gecode/OR-Tools/CUDA pour cette cible).
#
# Soumettre :
#   sbatch test_paper/build_bench_v1cld.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=build_bench_v1cld
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )}"
cd "$REPO"

module load StdEnv/2023
module load gcc/12.3
module load cmake/3.27.7

GENERIC_BUILD="$REPO/solver_generic/build_cc"
BAB_BUILD="$REPO/solver_bab/build"

echo "======================================================================"
echo "  build_bench_v1cld — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "======================================================================"

echo ""
echo "=== [1/2] v1ld_generic_core (contient le fix de scaling mkp_cld_problem_dp) ==="
cmake --build "$GENERIC_BUILD" --target v1ld_generic_core -j4 2>&1

echo ""
echo "=== [2/2] bench_v1cld_subproblems (nouvelle cible) ==="
# Reconfigure explicite requis : ajouter une cible dans CMakeLists.txt ne
# suffit pas, gmake ne connaît le nom de la cible qu'après régénération du
# Makefile (le check auto de CMake ne se déclenche que pour des cibles déjà
# CONNUES de gmake — une cible totalement nouvelle échoue avant même d'y
# arriver : "No rule to make target"). CMakeCache.txt garde toutes les
# options déjà choisies (GECODE_PREFIX, GNN_DEVICE=cuda, etc.), donc un
# simple `cmake -S ... -B ...` sans -D suffit à les réappliquer.
cmake -S "$REPO/solver_bab" -B "$BAB_BUILD" 2>&1 | tail -10
cmake --build "$BAB_BUILD" --target bench_v1cld_subproblems -j4 2>&1

[[ -x "$BAB_BUILD/bench_v1cld_subproblems" ]] || { echo "ERREUR : binaire non généré"; exit 1; }

echo ""
echo "[OK] $BAB_BUILD/bench_v1cld_subproblems ($(date -r "$BAB_BUILD/bench_v1cld_subproblems"))"
echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
