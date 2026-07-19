#!/bin/bash
# run_check_batched_n20_N500.sh — REF vs GNN batché (K=128) sur n=20/N=500,
# cas OOD en N connu cassé en mono-nœud (project_compare_ref_gnn_n20_N500 :
# GNN mono-nœud OOM-killed, mais avec device=cpu et seulement 4G mem — pas
# concluant). Ici : device=cuda, mémoire généreuse, pour voir si le batching
# (qui a déjà réduit drastiquement le coût du forward à N=100, cf.
# project_gnn_batched_bab_integration_result) rend N=500 exploitable.
#
# Référence connue : REF seul a terminé en 62.0s / 66456 nœuds / optimal
# 5221.9920 (mesuré précédemment sur ce même dataset).
#
# Soumettre :
#   sbatch run_check_batched_n20_N500.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=check_batched_n20_N500
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -uo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

source solver_bab/setup_gpu_env.sh
VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

CKPT="gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
PROD="$REPO/solver_bab/build/bab_mkp_gecode"
BATCHED="$REPO/solver_bab/build/bab_mkp_gecode_gnn_batched"
DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n20-m5-a50-we-r42-S500.txt"
EPS=0.10
T_ROOT=200
T_NODE=50
TIME_LIMIT=1500

echo "======================================================================"
echo "  check_batched_n20_N500 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Noeud : $(hostname)"
echo "  n=20  N=500 (OOD en N, hors distribution d'entraînement {20,50,100})"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "--- REF (squelette nu) ---"
"$PROD" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIMIT" 0 "$T_ROOT" ref \
  || echo "[ERREUR] REF a échoué (code $?)"

echo ""
echo "--- BATCHÉ K=128 ---"
"$BATCHED" "$DATASET" "$CKPT" "$EPS" "$TIME_LIMIT" 0 cuda 128 \
  || echo "[ERREUR] BATCHÉ K=128 a échoué (code $?)"

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
