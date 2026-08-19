#!/bin/bash
#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=rebuild_gnn_batched_orig
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )}"
cd "$REPO"

module load StdEnv/2023
module load gcc/12.3
module load cmake/3.27.7

cmake -S "$REPO/solver_bab" -B "$REPO/solver_bab/build" 2>&1 | tail -10
cmake --build "$REPO/solver_bab/build" --target bab_mkp_gecode_gnn_batched -j4 2>&1

[[ -x "$REPO/solver_bab/build/bab_mkp_gecode_gnn_batched" ]] || { echo "ERREUR : binaire non généré"; exit 1; }
echo "[OK] bab_mkp_gecode_gnn_batched reconstruit"
