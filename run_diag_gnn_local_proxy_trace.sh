#!/bin/bash
# run_diag_gnn_local_proxy_trace.sh — trace nœud par nœud (GNN_TRACE_DEBUG) sur
# n=20/N=100, mode GNN plein, pour tester si gap = obj_max - obj_min (domaine
# APRÈS propagation CP héritée, AVANT le rel(obj<=ub_int) de ce nœud) sépare
# les nœuds où la borne GNN cause un fail (ub_int < obj_min) de ceux où non.
#
# cf. discussion "proxy local" — alternative au gating par profondeur
# (gnn_min_depth_frac), qui n'a donné qu'un gain modeste (project_gnn_depth_gated_threshold_result)
# car coût et utilité de la borne sont concentrés dans la même bande de profondeur.
#
# Sortie : logs/slurm_diag_gnn_local_proxy_trace_<job>.err contient les lignes
# TRACE,<x_mask>,<depth>,<fp>,<ub_relative>,<ub>,<obj_min>,<obj_max>
#
# Soumettre :
#   sbatch run_diag_gnn_local_proxy_trace.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=diag_gnn_local_proxy_trace
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
echo "  diag_gnn_local_proxy_trace — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Noeud : $(hostname)"
echo "  n=20  N=100  mode=gnn plein (gnn_min_depth_frac=0.0)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$PROD" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIMIT" 0 "$T_ROOT" gnn non 0.1 "$CKPT" cuda python3 native sd_edit 0.0

echo ""
echo "Terminé — $(date)"
