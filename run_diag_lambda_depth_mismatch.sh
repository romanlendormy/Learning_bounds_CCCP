#!/bin/bash
# run_diag_lambda_depth_mismatch.sh — Test de l'hypothèse du centrage
# invalide quand N_z<N : gnn_ssp/diag_lambda_depth_mismatch.py.
#
# Soumettre :
#   sbatch run_diag_lambda_depth_mismatch.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=diag_lambda_depth_mismatch
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
VENV_PATH="$HOME/envs/learning_bounds"

echo "=============================="
echo "  diag_lambda_depth_mismatch — $(date)"
echo "  Job : ${SLURM_JOB_ID}   Nœud : $(hostname)"
echo "=============================="

module load StdEnv/2023 gcc/12.3.0 python/3.11.5 2>/dev/null || \
    module load python/3.11 2>/dev/null || \
    echo "  (aucun module python chargé)"

source "$VENV_PATH/bin/activate"

export LD_LIBRARY_PATH="$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1

python "$REPO/gnn_ssp/diag_lambda_depth_mismatch.py" \
    --combo E4_T12_N20 \
    --n_graphs 150 \
    --n_epochs 50

echo ""
echo "Terminé — $(date)"
