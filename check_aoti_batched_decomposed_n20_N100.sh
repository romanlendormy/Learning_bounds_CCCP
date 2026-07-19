#!/bin/bash
# check_aoti_batched_decomposed_n20_N100.sh — Test de production du backend
# AOTInductor batché (gnn_backend=native_aoti) après décomposition du GRUCell
# (débloque torch.export/aoti_compile_and_package sur CUDA avec B dynamique,
# cf. probe_gru_decomposed_aoti_cuda.py, validé 2026-07-18).
#
# 1er appel native_aoti déclenche l'export complet (20 artefacts .pt2,
# n_g=1..20, ~17.5s/forme la première fois d'après la sonde) — d'où le temps
# limite généreux.
#
# Repères connus :
#   REF (job 16678921) : 99302 nœuds, obj=5221.9914, 8.43s
#   gnn_batched backend=native (job 16678921) : 11494 nœuds, obj=5221.9914,
#     12.09s (borne 8.39s : graphe 2.63/fwd 4.06/zeta 1.70) — run bruité,
#     cf. aussi job 16677150 : 10.08s (borne 7.14s : graphe 2.05/fwd 3.39/zeta 1.70)
#
# Soumettre :
#   sbatch check_aoti_batched_decomposed_n20_N100.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=check_aoti_decomposed_n20_N100
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
echo "  check_aoti_batched_decomposed_n20_N100 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Noeud : $(hostname)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "--- REF (squelette nu, sans borne) ---"
"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIMIT" 0 "$T_ROOT" ref \
  || echo "[ERREUR] REF a échoué (code $?)"

echo ""
echo "--- gnn_batched backend=native (contrôle, K=$K, zeta=surrogate) ---"
"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIMIT" 0 "$T_ROOT" gnn_batched \
  non 0.1 "$CKPT" cuda python3 ipc sd_edit 0.0 0.0 1.0 "$K" 1.0 surrogate 1 native \
  || echo "[ERREUR] backend=native a échoué (code $?)"

echo ""
echo "--- gnn_batched backend=native_aoti (K=$K, zeta=surrogate) — 1er run : export .pt2 x20 ---"
time "$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIMIT" 0 "$T_ROOT" gnn_batched \
  non 0.1 "$CKPT" cuda python3 ipc sd_edit 0.0 0.0 1.0 "$K" 1.0 surrogate 1 native_aoti \
  || echo "[ERREUR] backend=native_aoti a échoué (code $?)"

echo ""
echo "--- gnn_batched backend=native_aoti (K=$K, zeta=surrogate) — 2e run : cache chaud ---"
"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIMIT" 0 "$T_ROOT" gnn_batched \
  non 0.1 "$CKPT" cuda python3 ipc sd_edit 0.0 0.0 1.0 "$K" 1.0 surrogate 1 native_aoti \
  || echo "[ERREUR] backend=native_aoti (2e run) a échoué (code $?)"

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
