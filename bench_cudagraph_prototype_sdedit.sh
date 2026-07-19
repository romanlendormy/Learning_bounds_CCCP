#!/bin/bash
# bench_cudagraph_prototype_sdedit.sh — Valide (Python, faible risque) si la
# capture CUDA Graph fonctionne sur le forward GNN sd_edit et quel gain elle
# apporte, avant tout portage dans le backend natif C++.
#
# Soumettre : sbatch bench_cudagraph_prototype_sdedit.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=cudagraph_proto_sdedit
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:10:00
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
echo "  bench_cudagraph_prototype_sdedit — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Noeud : $(hostname)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python3 gnn_sd_edit/prototype_cudagraph.py --checkpoint "$CKPT" --n_g 20 --N 50

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
