#!/bin/bash
# build_test_gnn_v1cld.sh — Build de test_gnn_v1cld_native_batched (nouvelle
# cible). Reconfigure explicite requise (nouvelle cible jamais vue par gmake
# — cf. build_bench_v1cld.sh pour l'explication complète du même piège).

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=build_test_gnn_v1cld
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

BAB_BUILD="$REPO/solver_bab/build"

echo "======================================================================"
echo "  build_test_gnn_v1cld — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "======================================================================"

cmake -S "$REPO/solver_bab" -B "$BAB_BUILD" 2>&1 | tail -10
cmake --build "$BAB_BUILD" --target test_gnn_v1cld_native_batched -j4 2>&1

[[ -x "$BAB_BUILD/test_gnn_v1cld_native_batched" ]] || { echo "ERREUR : binaire non généré"; exit 1; }

echo ""
echo "[OK] $BAB_BUILD/test_gnn_v1cld_native_batched"
echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
