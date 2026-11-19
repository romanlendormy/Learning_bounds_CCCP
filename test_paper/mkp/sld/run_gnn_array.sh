#!/bin/bash
# test_paper/mkp/sld/run_gnn_array.sh — LD (borne SLD, λ prédit par le GNN) sur
# les 120 instances de test : moteur gnn_batched (DFS + buckets de K = 64 nœuds
# évalués en un forward GPU), ζ surrogate, backend LibTorch natif, une tâche
# d'array par ligne de test_paper/mkp/instances_list.txt.
#
# Prérequis : export TorchScript du modèle déjà généré (cf. test_paper/mkp/README.md,
# sanity_check.sh), car le dépôt peut être en lecture seule sur les nœuds de calcul.
# Soumettre depuis la racine du dépôt :
#   sbatch --array=1-120 test_paper/mkp/sld/run_gnn_array.sh

#SBATCH --account=def-qcappart
#SBATCH --job-name=mkp_sld_gnn
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%x_%A_%a.out
#SBATCH --error=logs/slurm_%x_%A_%a.err

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO"   # les binaires B&B doivent tourner depuis la racine du dépôt
source env/modules_trillium.sh
source env/setup_gpu_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

TD="$REPO/test_paper/mkp"
INSTANCES_DIR="${INSTANCES_DIR:-$TD/instances}"
OUT="${OUT_DIR:-${SCRATCH:-$REPO/runs}/test_paper_mkp/sld}"
mkdir -p "$OUT/logs" "$OUT/results"

read -r N_ITEMS N_SCEN M SEED FILE <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TD/instances_list.txt")"
DATASET="$INSTANCES_DIR/$FILE"
[[ -f "$DATASET" ]] || { echo "[ERREUR] instance introuvable : $DATASET" >&2; exit 1; }

CKPT="models/mkp/sld/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
EPS=0.10; T_ROOT=200; T_NODE=50; K=64
TIME_LIM_S=0   # pas de limite côté solveur : le mur SLURM (24 h) est la seule limite

TAG="n${N_ITEMS}_N${N_SCEN}_seed${SEED}"
LOG="$OUT/logs/gnn_$TAG.log"
{
echo "# sld gnn_batched — n=$N_ITEMS N=$N_SCEN m=$M seed=$SEED — K=$K zeta=surrogate — $(date)"
echo "# job ${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0} — nœud $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
} > "$LOG"

# Arguments positionnels de bab_mkp_gecode (cf. solver_bab/mkp/main_gecode.cpp) :
#   dataset eps T_node time_limit rep T_root mode
#   patience patience_tol ckpt device python backend_mono_nœud(ignoré ici) variant
#   min_depth_frac min_gap_frac max_depth_frac K batched_max_depth_frac zeta_method stride backend_batché
"$REPO/build/solver_bab/bab_mkp_gecode" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" gnn_batched \
  non 0.1 "$CKPT" cuda python3 ipc sd_edit 0.0 0.0 1.0 "$K" 1.0 surrogate 1 native \
  >> "$LOG" 2>&1

python3 "$TD/parse_result.py" gnn_batched "$N_ITEMS" "$N_SCEN" "$M" "$SEED" "$LOG" > "$OUT/results/gnn_$TAG.csv"
