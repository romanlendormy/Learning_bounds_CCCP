#!/bin/bash
# bench_gnn_light_vs_sdedit.sh — Compare le temps de forward de gnn_light
# (GNNLightScriptable : 1 couche de message passing, hidden=64, poids
# aléatoires) au modèle sd_edit de production (GNNScriptable : 2 couches,
# hidden=128, checkpoint entraîné 500 epochs) via
# gnn_sd_edit/benchmark_gnn_light.py, sur les mêmes formes que
# bench_gnn_forward_breakdown_sdedit.sh (n_g=20/10/2, N=50).
#
# Soumettre : sbatch bench_gnn_light_vs_sdedit.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=gnn_light_vs_sdedit
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
echo "  bench_gnn_light_vs_sdedit — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Noeud : $(hostname)"
echo "  Checkpoint sd_edit=$CKPT"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python3 gnn_sd_edit/benchmark_gnn_light.py \
    --checkpoint "$CKPT" --device cuda --N 50 --n_g 20 10 2 \
    --n_warmup 30 --n_iters 200

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
