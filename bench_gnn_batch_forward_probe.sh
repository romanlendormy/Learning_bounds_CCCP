#!/bin/bash
# bench_gnn_batch_forward_probe.sh — Sonde isolée : le batching de B nœuds
# B&B en un seul appel GNN amortit-il le plancher d'exécution/dispatch déjà
# identifié ? Go/no-go le moins cher de l'Étape 2 du plan de batching, avant
# de toucher à Gecode (cf. solver_bab/mkp/gnn_batch_forward_probe.cpp,
# gnn_sd_edit/gnn_sd_edit_scriptable_batched.py).
#
# Soumettre :
#   sbatch bench_gnn_batch_forward_probe.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=gnn_batch_fwd_probe
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:45:00
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
SINGLE="${CKPT}.scripted.pt"
BATCHED="${CKPT}.scripted_batched.pt"
PROBE="$REPO/solver_bab/build/gnn_batch_forward_probe"

echo "======================================================================"
echo "  bench_gnn_batch_forward_probe — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "=== n_g=20, N_z=50, m=5, B=1..128 ==="
"$PROBE" "$SINGLE" "$BATCHED" cuda 20 50 5 200 30 "1,2,4,8,16,32,64,128"

echo ""
echo "=== n_g=10, N_z=50, m=5 (profondeur moyenne) ==="
"$PROBE" "$SINGLE" "$BATCHED" cuda 10 50 5 200 30 "1,2,4,8,16,32,64,128"

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
