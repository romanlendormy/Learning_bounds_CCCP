#!/bin/bash
# bench_gnn_depth_gated_threshold.sh — compare GNN plein (gnn_min_depth_frac=0.0,
# comportement historique) vs GNN gated par profondeur relative
# (gnn_min_depth_frac=0.5) vs REF, sur 4 instances in-distribution
# (n in {10,20} x N in {20,100}).
#
# Motivation : sweep de robustesse job 16186510 montre fails_from_bound_by_depth
# == 0 pour depth/n < ~0.5 dans 6/8 configs (n in {10,20}, N in {20,50,100,500})
# — cf. project_gnn_batched_robustness_sweep_launch. Ce script teste si sauter
# l'appel GNN sous ce seuil (gnn_min_depth_frac=0.5, cf. solver_bab/mkp/gecode_bab.hpp
# MKPSharedData::gnn_min_depth_frac) préserve le nombre de nœuds/la qualité de
# la borne finale tout en réduisant t_bound_total_s, sur les MÊMES instances que
# le sweep précédent (comparaison directe possible).
#
# Pas de `set -e` : une instance qui échoue ne doit pas faire perdre les autres.
#
# Soumettre :
#   sbatch bench_gnn_depth_gated_threshold.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=gnn_depth_gated
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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

CKPT="gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
PROD="$REPO/solver_bab/build/bab_mkp_gecode"
DATADIR="data/knapsack/datasets-pisinger"
EPS=0.10
T_ROOT=200
T_NODE=50
THRESHOLD=0.5   # cf. onset moyen observé (~50% de n) dans le sweep 16186510

echo "======================================================================"
echo "  bench_gnn_depth_gated_threshold — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Noeud : $(hostname)"
echo "  Seuil testé : gnn_min_depth_frac=$THRESHOLD"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

run_one() {
  local n="$1" N="$2" time_limit="$3"
  local dataset="$DATADIR/ccmk-song-pisinger-n${n}-m5-a50-we-r42-S${N}.txt"

  echo ""
  echo "######################################################################"
  echo "  n=$n  N=$N  time_limit=${time_limit}s"
  echo "######################################################################"

  echo "--- REF (squelette nu) ---"
  "$PROD" "$dataset" "$EPS" "$T_NODE" "$time_limit" 0 "$T_ROOT" ref \
    || echo "[ERREUR] REF n=$n N=$N a échoué (code $?)"

  echo ""
  echo "--- GNN plein (gnn_min_depth_frac=0.0, comportement historique) ---"
  "$PROD" "$dataset" "$EPS" "$T_NODE" "$time_limit" 0 "$T_ROOT" gnn non 0.1 "$CKPT" cuda python3 native sd_edit 0.0 \
    || echo "[ERREUR] GNN plein n=$n N=$N a échoué (code $?)"

  echo ""
  echo "--- GNN gated (gnn_min_depth_frac=$THRESHOLD) ---"
  "$PROD" "$dataset" "$EPS" "$T_NODE" "$time_limit" 0 "$T_ROOT" gnn non 0.1 "$CKPT" cuda python3 native sd_edit "$THRESHOLD" \
    || echo "[ERREUR] GNN gated n=$n N=$N a échoué (code $?)"
}

# --- in-distribution : n in {10,20} x N in {20,100} ---
run_one 10 20  60
run_one 10 100 300
run_one 20 20  60
run_one 20 100 300

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
