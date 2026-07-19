#!/bin/bash
# run_quality_fig.sh — Génère fig_quality_*_<SPLIT>.png depuis un checkpoint.
#
# Soumettre :
#   sbatch run_quality_fig.sh
#
# Surcharger :
#   sbatch --export=ALL,SPLIT=test run_quality_fig.sh

#SBATCH --account=def-qcappart_gpu
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=quality_fig
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

CKPT="${CKPT:-gnn/runs/train_scaled/ckpt_scaled_n10_ep100_lr0.0003_bs32_h128_seed42_lam25_ep0100.pt}"
SPLIT="${SPLIT:-train}"

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
VENV_PATH="$HOME/envs/learning_bounds"

echo "======================================================================"
echo "  quality_fig — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  CKPT  : $CKPT"
echo "  SPLIT : $SPLIT"
echo "======================================================================"

module load StdEnv/2023 gcc/12.3.0 python/3.11.5 2>/dev/null || \
    module load python/3.11 2>/dev/null || \
    echo "  (aucun module python chargé)"

source "$VENV_PATH/bin/activate"

export LD_LIBRARY_PATH="$REPO/external/gecode/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python "$REPO/gnn/gen_quality_fig.py" \
    "$CKPT" \
    --split "$SPLIT" \
    --device auto

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
