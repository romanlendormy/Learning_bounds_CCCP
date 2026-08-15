#!/bin/bash
# run_gnn_array.sh — mode gnn_batched (K=64, zeta=surrogate, stride=1, cumul
# complet, backend native, variant sd_edit) sur les 120 instances de
# test_paper/, une tâche d'array par instance (test_paper/instances_list.txt,
# format : n N m seed path). GPU h100 requis.
#
# Même checkpoint et mêmes paramètres que les jobs compare_ref_gnn_batched_*_gnn.sh
# actuellement en queue/en cours sur ce projet (décision utilisateur 2026-08-13) :
#   CKPT=ckpt_sdedit_scaled_n10_20_..._v1_sdedit_500ep_ep0500.pt
#   K=64, zeta_method=surrogate, gnn_node_stride=1 (borne calculée à CHAQUE
#   nœud, aucun bypass), gnn_batched_backend=native.
#
# n=15 et n=30 ne sont couverts par AUCUN entraînement GNN (checkpoint entraîné
# sur n∈{10,20} seulement) — n=15 en interpolation, n=30 en extrapolation.
# Décision utilisateur : garder tel quel, documenter la généralisation.
#
# time_limit=0 (illimité) — mur SLURM (24h) comme seul filet de sécurité.
#
# Soumettre :
#   sbatch --array=1-120 test_paper/run_gnn_array.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=test_paper_gnn
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%x_%A_%a.out
#SBATCH --error=logs/slurm_%x_%A_%a.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )}"
cd "$REPO"

source solver_bab/setup_gpu_env.sh
VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

TESTDIR="$REPO/test_paper"
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TESTDIR/instances_list.txt")
read -r N_ITEMS N_SCEN M SEED RELPATH <<< "$LINE"
DATASET="$TESTDIR/$RELPATH"

if [[ ! -f "$DATASET" ]]; then
    echo "[ERREUR] instance introuvable : $DATASET" >&2
    exit 1
fi

BINARY="$REPO/solver_bab/build/bab_mkp_gecode"
CKPT="gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
EPS=0.10
T_ROOT=200
T_NODE=50
TIME_LIM_S=0   # illimité — cf. en-tête
K=64

LOGFILE="$TESTDIR/logs/gnn_n${N_ITEMS}_N${N_SCEN}_seed${SEED}.log"
RESULT_CSV="$TESTDIR/results/gnn_n${N_ITEMS}_N${N_SCEN}_seed${SEED}.csv"

{
echo "======================================================================"
echo "  test_paper gnn_batched — n=$N_ITEMS N=$N_SCEN m=$M seed=$SEED — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}   Nœud : $(hostname)"
echo "  K=$K  zeta_method=surrogate  gnn_node_stride=1  backend=native"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
} | tee "$LOGFILE"

"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" gnn_batched \
  non 0.1 "$CKPT" cuda python3 ipc sd_edit 0.0 0.0 1.0 "$K" 1.0 surrogate 1 native \
  2>&1 | tee -a "$LOGFILE"

python3 "$TESTDIR/parse_result.py" gnn_batched "$N_ITEMS" "$N_SCEN" "$M" "$SEED" "$LOGFILE" > "$RESULT_CSV"

echo "" | tee -a "$LOGFILE"
echo "Terminé — $(date)" | tee -a "$LOGFILE"
