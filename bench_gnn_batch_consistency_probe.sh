#!/bin/bash
# bench_gnn_batch_consistency_probe.sh — Étape 2d du plan de batching :
# vérifie que GNNSdEditNativeBatched::query_batch() reproduit EXACTEMENT
# GNNSdEditNative::query() (production, inchangé) sur une vraie instance MKP
# (cf. solver_bab/mkp/gnn_batch_consistency_probe.cpp).
#
# Soumettre :
#   sbatch bench_gnn_batch_consistency_probe.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=gnn_batch_consistency
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

module load cuda/12.6
VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export LD_LIBRARY_PATH="$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

CKPT="gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n20-m5-a50-we-r42-S50.txt"
PROBE="$REPO/solver_bab/build/gnn_batch_consistency_probe"

echo "======================================================================"
echo "  bench_gnn_batch_consistency_probe — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "=== n_g_target=20 (racine), B=8 ==="
"$PROBE" "$DATASET" "$CKPT" 0 0.10 cuda 20 8 0

echo ""
echo "=== n_g_target=10 (mi-profondeur), B=8 ==="
"$PROBE" "$DATASET" "$CKPT" 0 0.10 cuda 10 8 1

echo ""
echo "=== n_g_target=5 (profond), B=16 ==="
"$PROBE" "$DATASET" "$CKPT" 0 0.10 cuda 5 16 2

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
