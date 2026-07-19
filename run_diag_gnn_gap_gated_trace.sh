#!/bin/bash
# run_diag_gnn_gap_gated_trace.sh — trace nœud par nœud (GNN_TRACE_DEBUG) sur
# n=20/N=100, mode GNN gap-gated (gnn_min_gap_frac=0.10), pour vérifier
# directement l'hypothèse de boucle de rétroaction : les nœuds dont l'appel
# GNN est SAUTÉ ne resserrent jamais obj.max() pour leurs descendants, qui
# héritent donc d'un domaine plus large que dans le run "plein" (job 16437043)
# — d'où le proxy gap_rel qui reste élevé bien plus longtemps que prévu par
# l'analyse offline (cf. discussion "gnn_local_proxy_gate").
#
# Sortie : logs/slurm_diag_gnn_gap_gated_trace_<job>.err — mêmes lignes TRACE
# que job 16437043, mais SEULEMENT pour les nœuds où l'appel a réellement eu
# lieu (gap_rel <= 0.10 à ce nœud) — comparer à la trace "plein" à profondeur
# égale pour quantifier le décalage.
#
# Soumettre :
#   sbatch run_diag_gnn_gap_gated_trace.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=diag_gnn_gap_gated_trace
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
export GNN_TRACE_DEBUG=1

CKPT="gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
PROD="$REPO/solver_bab/build/bab_mkp_gecode"
DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n20-m5-a50-we-r42-S100.txt"
EPS=0.10
T_ROOT=200
T_NODE=50
TIME_LIMIT=300

echo "======================================================================"
echo "  diag_gnn_gap_gated_trace — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Noeud : $(hostname)"
echo "  n=20  N=100  mode=gnn gap-gated (gnn_min_gap_frac=0.10)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$PROD" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIMIT" 0 "$T_ROOT" gnn non 0.1 "$CKPT" cuda python3 native sd_edit 0.0 0.10

echo ""
echo "Terminé — $(date)"
