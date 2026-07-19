#!/bin/bash
# compare_gnn_batched_node_stride_n20_N100.sh — A/B test demandé : throttle
# gnn_node_stride (mode gnn_batched) — 1 candidat GNN sur `stride` obtient
# réellement sa borne, les autres sont bypassés (poussés dans `open` avec
# obj.max() post-CP, aucune borne) — cf. doc GNNBatchedBabResult::
# nodes_bypassed_stride / gnn_node_stride (gecode_bab_gnn_batched.hpp).
#
# Motivation (diagnostic terrain, cf. discussion) : beaucoup de ub_tightened
# mais peu de fails_from_bound — la borne GNN élague rarement alors qu'elle
# coûte cher à chaque nœud. Teste si un sous-échantillonnage PUR (indépendant
# de la profondeur) récupère plus de temps que le gating par profondeur déjà
# mesuré comme modeste (cf. mémoire gnn_max_depth_frac_batched_result).
#
# Référence connue (run_check_batched_n20_N100.sh) : REF=8.4s/99302 nœuds,
# batché K=128=26.0s/25399 nœuds, optimal 5221.992. zeta_method=bnb partout
# ici (isole l'effet du stride ; le gain surrogate déjà validé séparément,
# cf. compare_gnn_batched_zeta_bnb_surrogate_n20_N100.sh).
#
# Soumettre :
#   sbatch compare_gnn_batched_node_stride_n20_N100.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=cmp_gnnbatch_stride_n20_N100
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
echo "  compare_gnn_batched_node_stride_n20_N100 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Noeud : $(hostname)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "--- REF (squelette nu, sans borne) ---"
"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIMIT" 0 "$T_ROOT" ref \
  || echo "[ERREUR] REF a échoué (code $?)"

for STRIDE in 1 5 10; do
  echo ""
  echo "--- gnn_batched gnn_node_stride=$STRIDE (K=$K, zeta_method=bnb) ---"
  "$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIMIT" 0 "$T_ROOT" gnn_batched \
    non 0.1 "$CKPT" cuda python3 ipc sd_edit 0.0 0.0 1.0 "$K" 1.0 bnb "$STRIDE" \
    || echo "[ERREUR] gnn_batched stride=$STRIDE a échoué (code $?)"
done

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
