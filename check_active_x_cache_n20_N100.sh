#!/bin/bash
# check_active_x_cache_n20_N100.sh — Validation du cache de features
# active_x-invariantes (cf. Impl::active_x_cache, gnn_sd_edit_native_batched.cpp)
# : mêmes nœuds/objectif qu'AVANT le cache (référence : job 16633631/16637529,
# stride=1, zeta_method=bnb, K=64 : 11494 nœuds, obj=5221.9914, temps borne
# 14.50-15.34s dont construction graphe 5.43-5.53s) — le cache est une
# optimisation EXACTE (pas une approximation comme stride/surrogate côté
# recherche), donc tout écart de nœuds/objectif signale un bug d'implémentation,
# pas un compromis attendu.
#
# Soumettre :
#   sbatch check_active_x_cache_n20_N100.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=check_active_x_cache_n20_N100
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
echo "  check_active_x_cache_n20_N100 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Noeud : $(hostname)"
echo "  Référence AVANT cache (job 16633631) : 11494 nœuds, obj=5221.9914,"
echo "  temps borne ~14.5-15.3s dont construction graphe ~5.4-5.5s"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "--- gnn_batched (K=$K, zeta_method=bnb, stride=1) AVEC cache active_x ---"
"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIMIT" 0 "$T_ROOT" gnn_batched \
  non 0.1 "$CKPT" cuda python3 ipc sd_edit 0.0 0.0 1.0 "$K" 1.0 bnb 1 \
  || echo "[ERREUR] gnn_batched a échoué (code $?)"

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
