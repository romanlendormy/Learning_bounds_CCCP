#!/bin/bash
# compare_ref_gnn_batched_n20_N500.sh — REF vs gnn_batched sur n=20/N=500
# (OOD en N par rapport à l'entraînement N∈{20,50,100}), avec le cumul
# complet des optimisations validées le 2026-07-17/18 : cache active_x,
# héritage incrémental de b_res, cap_frac/scen_* vectorisés sur tout le
# batch, torch::jit::freeze/optimize_for_inference, zeta_method=surrogate,
# K=64, stride=1.
#
# N=500 est 5x plus gros que le N=100 déjà testé — temps/mémoire généreux
# par précaution (pas encore mesuré avec ce cumul d'optimisations).
#
# Soumettre :
#   sbatch compare_ref_gnn_batched_n20_N500.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=cmp_ref_gnn_batched_n20_N500
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
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

BINARY="$REPO/solver_bab/build/bab_mkp_gecode"
CKPT="gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n20-m5-a50-we-r42-S500.txt"
EPS=0.10
T_ROOT=200
T_NODE=50
TIME_LIMIT=600
K=64

echo "======================================================================"
echo "  compare_ref_gnn_batched_n20_N500 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Noeud : $(hostname)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "--- REF (squelette nu, sans borne) ---"
"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIMIT" 0 "$T_ROOT" ref \
  || echo "[ERREUR] REF a échoué (code $?)"

echo ""
echo "--- gnn_batched (K=$K, zeta=surrogate, stride=1, cumul complet) ---"
"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIMIT" 0 "$T_ROOT" gnn_batched \
  non 0.1 "$CKPT" cuda python3 ipc sd_edit 0.0 0.0 1.0 "$K" 1.0 surrogate 1 native \
  || echo "[ERREUR] gnn_batched a échoué (code $?)"

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
