#!/bin/bash
# run_test_inference_sweep_rorqual.sh — Balaie (N, depth) pour un checkpoint GNN V1LD
# et produit un CSV brut, un CSV agrégé et une figure récapitulative.
#
# Soumettre :
#   sbatch run_test_inference_sweep_rorqual.sh
#
# Surcharger des paramètres :
#   sbatch --export=ALL,CKPT=<autre_ckpt.pt>,N_LIST="50 100",DEPTHS="0 5 9" run_test_inference_sweep_rorqual.sh

#SBATCH --account=def-qcappart_gpu
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=test_inference_sweep
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
VENV_PATH="$HOME/envs/learning_bounds"

CKPT="${CKPT:-$REPO/gnn/runs/train_scaled/ckpt_scaled_n10_ep500_lr0.0003_bs32_h128_seed42_v6_n10_500ep_ep0500.pt}"
N_VAL_ARG="${N_VAL:-10}"
N_LIST="${N_LIST:-50 100 200 500}"
DEPTHS="${DEPTHS:-0 1 2 3 4 5 6 7 8 9}"
NUM_INSTANCES="${NUM_INSTANCES:-10}"
PICK_INDEX="${PICK_INDEX:-0}"
T_MAX="${T_MAX:-500}"
OUT_DIR="${OUT_DIR:-$REPO/gnn/runs/test_inference_sweep}"

echo "======================================================================"
echo "  test_inference_sweep — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  CKPT           : $CKPT"
echo "  n=$N_VAL_ARG  N_LIST=($N_LIST)  DEPTHS=($DEPTHS)"
echo "  NUM_INSTANCES  : $NUM_INSTANCES  (à partir de pick_index=$PICK_INDEX)"
echo "  T_MAX          : $T_MAX"
echo "  OUT_DIR        : $OUT_DIR"
echo "======================================================================"

module load StdEnv/2023 gcc/12.3.0 python/3.11.5 2>/dev/null || \
    module load python/3.11 2>/dev/null || \
    echo "  (aucun module python chargé)"

source "$VENV_PATH/bin/activate"

export LD_LIBRARY_PATH="$REPO/external/gecode/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python "$REPO/gnn/test_inference_sweep.py" \
    --ckpt          "$CKPT"          \
    --n             "$N_VAL_ARG"     \
    --N_list        $N_LIST          \
    --depths        $DEPTHS          \
    --num_instances "$NUM_INSTANCES" \
    --pick_index    "$PICK_INDEX"    \
    --T_max         "$T_MAX"         \
    --out_dir       "$OUT_DIR"       \
    --device        cpu

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "  Résultats dans : $OUT_DIR"
echo "======================================================================"
