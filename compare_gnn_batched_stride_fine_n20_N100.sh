#!/bin/bash
# compare_gnn_batched_stride_fine_n20_N100.sh — Balayage fin de
# gnn_node_stride (1,2,3,4,5) avec le cumul complet des optimisations
# (cache active_x, héritage b_res, zeta_method=surrogate) — le stride=5/10
# testé précédemment (job 16637529, AVANT ces optimisations, zeta=bnb)
# faisait exploser les nœuds (×5.3 à ×6.5) pour un gain temps modeste.
# Maintenant que forward GNN domine le coût de borne (pas zeta), un stride
# FIN (2-5) pourrait avoir un meilleur compromis : moins d'appels forward,
# sans payer le même prix en nœuds qu'un stride agressif.
#
# Repères connus (job 16674669) :
#   REF : 99302 nœuds, 8.43s
#   gnn_batched stride=1 (cumul complet) : 11494 nœuds, 11.86s (borne 8.51s :
#     graphe 2.18/fwd 4.64/zeta 1.70)
#
# Soumettre :
#   sbatch compare_gnn_batched_stride_fine_n20_N100.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=cmp_gnnbatch_stride_fine_n20_N100
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
echo "  compare_gnn_batched_stride_fine_n20_N100 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Noeud : $(hostname)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "--- REF (squelette nu, sans borne) ---"
"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIMIT" 0 "$T_ROOT" ref \
  || echo "[ERREUR] REF a échoué (code $?)"

for STRIDE in 1 2 3 4 5; do
  echo ""
  echo "--- gnn_batched gnn_node_stride=$STRIDE (K=$K, zeta=surrogate) ---"
  "$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIMIT" 0 "$T_ROOT" gnn_batched \
    non 0.1 "$CKPT" cuda python3 ipc sd_edit 0.0 0.0 1.0 "$K" 1.0 surrogate "$STRIDE" native \
    || echo "[ERREUR] gnn_batched stride=$STRIDE a échoué (code $?)"
done

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
