#!/bin/bash
# check_bres_inherit_n20_N100.sh — Validation de l'héritage incrémental de
# b_res (MKPSpaceGNNBatched::b_res/apply_item_fixed, cf. échange du
# 2026-07-18) : optimisation EXACTE (pas une approximation) — nœuds/objectif
# DOIVENT rester identiques à la référence AVANT ce changement.
#
# Référence (job 16658298, backend=native, cache active_x déjà actif,
# AVANT héritage b_res) : 11494 nœuds, obj=5221.9914, temps borne 11.66s
# (graphe 2.316/fwd 4.957/zeta 4.387).
#
# Soumettre :
#   sbatch check_bres_inherit_n20_N100.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=check_bres_inherit_n20_N100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:20:00
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
DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n20-m5-a50-we-r42-S100.txt"
EPS=0.10
T_ROOT=200
T_NODE=50
TIME_LIMIT=300
K=64

echo "======================================================================"
echo "  check_bres_inherit_n20_N100 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Noeud : $(hostname)"
echo "  Réf. AVANT héritage b_res (job 16658298) : 11494 nœuds, obj=5221.9914,"
echo "  borne 11.66s (graphe 2.316/fwd 4.957/zeta 4.387)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "--- gnn_batched backend=native (K=$K, zeta=bnb, stride=1) ---"
"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIMIT" 0 "$T_ROOT" gnn_batched \
  non 0.1 "$CKPT" cuda python3 ipc sd_edit 0.0 0.0 1.0 "$K" 1.0 bnb 1 native \
  || echo "[ERREUR] backend=native a échoué (code $?)"

echo ""
echo "--- gnn_batched backend=native, zeta=surrogate (cumul avec les gains précédents) ---"
"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIMIT" 0 "$T_ROOT" gnn_batched \
  non 0.1 "$CKPT" cuda python3 ipc sd_edit 0.0 0.0 1.0 "$K" 1.0 surrogate 1 native \
  || echo "[ERREUR] backend=native+surrogate a échoué (code $?)"

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
