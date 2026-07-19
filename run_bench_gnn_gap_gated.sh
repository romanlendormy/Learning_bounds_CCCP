#!/bin/bash
# run_bench_gnn_gap_gated.sh — compare REF vs GNN plein (gnn_min_gap_frac=0.0)
# vs GNN gap-gated (gnn_min_gap_frac=0.10), sur n=20/N=100.
#
# Prototype du proxy local gap_rel = (obj.max()-obj.min())/obj.max() comme
# alternative au gating par profondeur (gnn_min_depth_frac), qui n'avait
# donné qu'un gain modeste (project_gnn_depth_gated_threshold_result).
# T=10% choisi sur l'analyse offline de la trace GNN_TRACE_DEBUG (job
# 16437043) : ~43% d'appels GNN en moins pour ~1.4% de fails perdus,
# cf. discussion "gnn_local_proxy_gate".
#
# Soumettre :
#   sbatch run_bench_gnn_gap_gated.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=bench_gnn_gap_gated
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

CKPT="gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
PROD="$REPO/solver_bab/build/bab_mkp_gecode"
DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n20-m5-a50-we-r42-S100.txt"
EPS=0.10
T_ROOT=200
T_NODE=50
TIME_LIMIT=300

echo "======================================================================"
echo "  bench_gnn_gap_gated — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Noeud : $(hostname)"
echo "  n=20  N=100  time_limit=${TIME_LIMIT}s"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "--- REF (squelette nu) ---"
"$PROD" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIMIT" 0 "$T_ROOT" ref \
  || echo "[ERREUR] REF a échoué (code $?)"

echo ""
echo "--- GNN plein (gnn_min_gap_frac=0.0, comportement historique) ---"
"$PROD" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIMIT" 0 "$T_ROOT" gnn non 0.1 "$CKPT" cuda python3 native sd_edit 0.0 0.0 \
  || echo "[ERREUR] GNN plein a échoué (code $?)"

echo ""
echo "--- GNN gap-gated (gnn_min_gap_frac=0.10) ---"
"$PROD" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIMIT" 0 "$T_ROOT" gnn non 0.1 "$CKPT" cuda python3 native sd_edit 0.0 0.10 \
  || echo "[ERREUR] GNN gap-gated a échoué (code $?)"

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
