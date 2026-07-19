#!/bin/bash
# bench_gnn_forward_breakdown_sdedit.sh — Décompose le temps du forward GNN
# sd_edit (message passing / linéaires / pooling / transferts H2D-D2H) via
# gnn_sd_edit/profile_forward_breakdown.py, sur 3 formes représentatives des
# profondeurs B&B (n_g=20 racine, n_g=10 mi-profondeur, n_g=2 profond),
# N=50, même checkpoint que bench_sdedit_vs_homogene_n20_N50.sh.
#
# Soumettre : sbatch bench_gnn_forward_breakdown_sdedit.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=gnn_fwd_breakdown_sdedit
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

CKPT="gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"

VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "======================================================================"
echo "  bench_gnn_forward_breakdown_sdedit — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Noeud : $(hostname)"
echo "  Checkpoint=$CKPT"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python3 gnn_sd_edit/profile_forward_breakdown.py \
    --checkpoint "$CKPT" --device cuda --N 50 --n_g 20 10 2 \
    --n_warmup 30 --n_iters 200

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
