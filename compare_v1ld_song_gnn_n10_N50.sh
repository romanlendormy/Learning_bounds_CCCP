#!/bin/bash
# compare_v1ld_song_gnn_n10_N50.sh — Tableau comparatif de la borne v1LD sur
# une instance n=10/m=5/N=50 (data/train_base_s10/instances_n10_m5_N50.txt,
# split TEST du checkpoint) : sous-gradient "maison" (T=500) vs inférence GNN
# (checkpoint ~10k graphes, device=cuda) vs V1LD façon SONG et al.
# (reproduction du plan coupant de Kelley de external/CCdual/ccdual.cpp,
# GLOP à la place de CPLEX — indisponible sur ce cluster).
#
# Soumettre :
#   sbatch compare_v1ld_song_gnn_n10_N50.sh

#SBATCH --account=def-qcappart
#SBATCH --job-name=cmp_v1ld_song_gnn_n10N50
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

CKPT="gnn/runs/train_scaled/ckpt_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_10k_ep0200.pt"

VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "======================================================================"
echo "  compare_v1ld_song_gnn_n10_N50 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  Checkpoint=$CKPT"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python3 -u compare_v1ld_song_gnn_n10_N50.py \
    --ckpt "$CKPT" --device cuda \
    --subgrad_T 500 --kelley_max_iters 2000 --kelley_time_limit 300 --kelley_lam_bound 50

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
