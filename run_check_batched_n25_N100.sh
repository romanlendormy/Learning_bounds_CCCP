#!/bin/bash
# run_check_batched_n25_N100.sh — REF vs GNN batché (K=128) sur n=25/N=100,
# SANS timeout interne (time_limit=-1 = illimité côté solveur, cf.
# main_gecode.cpp/main_bab_gnn_batched.cpp) : chaque run tourne jusqu'à
# OPTIMAL réel. Seul le mur SLURM (12h) plafonne l'ensemble.
# n=25 hors distribution d'entraînement (n_list={10,20}), N=100 in-distribution
# (n_scens={20,50,100}) — simple OOD en n, contrairement à
# run_check_batched_n25_N500.sh (double OOD n ET N).
#
# Soumettre :
#   sbatch run_check_batched_n25_N100.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=check_batched_n25_N100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
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
DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n25-m5-a50-we-r42-S100.txt"
EPS=0.10
T_ROOT=200
T_NODE=50
TIME_LIMIT=-1

echo "======================================================================"
echo "  check_batched_n25_N100 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Noeud : $(hostname)"
echo "  n=25  N=100 (OOD en n seulement), time_limit=illimité, mur SLURM=12h"
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
