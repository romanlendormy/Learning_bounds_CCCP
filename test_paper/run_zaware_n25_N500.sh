#!/bin/bash
# run_zaware_n25_N500.sh — bab_mkp_gecode_gnn_batched_zaware (ragged) sur
# seed=43, n=25/N=500, K=64/cuda — à comparer à la ligne déjà collectée dans
# results.csv (origine) : nodes=278576, solve_time_s=1390.898, objectif=6121.9157.
# n=25 est en extrapolation pour le checkpoint (entraîné sur n∈{10,20}) —
# comme pour l'origine, à documenter, pas bloquant.

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=zaware_n25_N500
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )}"
cd "$REPO"
source solver_bab/setup_gpu_env.sh

BUILD="$REPO/solver_bab/build"
DATASET="$REPO/test_paper/instance_n25_m5_N500_seed43.txt"
CKPT="gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"

echo "======================================================================"
echo "  [Z-AWARE ragged] n=25 N=500 seed=43 — K=64, cuda"
echo "  Référence (origine, déjà collectée) : nodes=278576 t=1390.898s obj=6121.9157"
echo "======================================================================"
"$BUILD/bab_mkp_gecode_gnn_batched_zaware" "$DATASET" "$CKPT" 0.10 -1 0 cuda 64
