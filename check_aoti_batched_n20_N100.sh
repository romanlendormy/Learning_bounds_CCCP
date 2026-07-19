#!/bin/bash
# check_aoti_batched_n20_N100.sh — Validation du nouveau backend AOTInductor
# batché (GNNSdEditNativeAOTIBatched, gnn_backend=native_aoti) vs le backend
# TorchScript batché (gnn_backend=native, déjà validé) sur n=20/N=100, K=64,
# stride=1, zeta_method=bnb (isole l'effet du backend seul, sans mélanger
# avec surrogate/cache déjà validés séparément).
#
# Référence connue (job 16640950, backend=native + cache actif) : 11494
# nœuds, obj=5221.9914, temps borne 12.49s (graphe 2.30/fwd 5.82/zeta 4.38).
#
# Contrairement à bnb/surrogate (bit-identiques), AOTInductor recompile les
# kernels — écart numérique attendu ~1e-5 sur le forward (cf. mémoire
# aoti-native-integration-hang), donc nœuds/objectif DEVRAIENT rester
# identiques mais pas garantis bit-exacts si un nœud est exactement à la
# frontière d'un fail.
#
# Premier run : le backend déclenche automatiquement l'export
# export_aoti_sdedit_batched.py pour n_g=1..20 (~15-20s/forme la première
# fois, cf. gnn_sd_edit_native_aoti.hpp) — d'où le temps limite généreux.
#
# Soumettre :
#   sbatch check_aoti_batched_n20_N100.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=check_aoti_batched_n20_N100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00
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
echo "  check_aoti_batched_n20_N100 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Noeud : $(hostname)"
echo "  Réf. backend=native (cache actif, job 16640950) : 11494 nœuds, "
echo "  obj=5221.9914, borne 12.49s (graphe 2.30/fwd 5.82/zeta 4.38)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "--- gnn_batched backend=native (contrôle, K=$K, zeta=bnb) ---"
"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIMIT" 0 "$T_ROOT" gnn_batched \
  non 0.1 "$CKPT" cuda python3 ipc sd_edit 0.0 0.0 1.0 "$K" 1.0 bnb 1 native \
  || echo "[ERREUR] backend=native a échoué (code $?)"

echo ""
echo "--- gnn_batched backend=native_aoti (K=$K, zeta=bnb) — 1er run : export .pt2 ---"
time "$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIMIT" 0 "$T_ROOT" gnn_batched \
  non 0.1 "$CKPT" cuda python3 ipc sd_edit 0.0 0.0 1.0 "$K" 1.0 bnb 1 native_aoti \
  || echo "[ERREUR] backend=native_aoti a échoué (code $?)"

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
