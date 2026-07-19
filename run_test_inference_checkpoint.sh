#!/bin/bash
# run_test_inference_checkpoint.sh — Inférence GNN V1LD sur des instances test
# (racines, jamais vues à l'entraînement) + comparaison sous-gradient.
#
# Soumettre :
#   sbatch run_test_inference_checkpoint.sh
#
# Surcharger des paramètres :
#   sbatch --export=ALL,CKPT=<autre_ckpt.pt>,N_VAL=100 run_test_inference_checkpoint.sh

#SBATCH --account=def-qcappart_gpu
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=test_inference_ckpt
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
VENV_PATH="$HOME/envs/learning_bounds"

CKPT="${CKPT:-$REPO/gnn/runs/train_scaled/ckpt_scaled_n10_ep500_lr0.0003_bs32_h128_seed42_v6_n10_500ep_ep0500.pt}"
N_VAL_ARG="${N_VAL:-10}"
N_SCEN_ARG="${N_SCEN:-500}"
DEPTH_ARG="${DEPTH:-0}"
NUM_INSTANCES="${NUM_INSTANCES:-10}"
PICK_INDEX="${PICK_INDEX:-0}"
SUBGRAD50_T="${SUBGRAD50_T:-50}"
SUBGRAD_CONV_T_MAX="${SUBGRAD_CONV_T_MAX:-500}"
SUBGRAD_CONV_PRECISION="${SUBGRAD_CONV_PRECISION:-0.1}"
SUBGRAD_CONV_PATIENCE="${SUBGRAD_CONV_PATIENCE:-30}"

echo "======================================================================"
echo "  test_inference_checkpoint — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  CKPT           : $CKPT"
echo "  n=$N_VAL_ARG  N=$N_SCEN_ARG  depth=$DEPTH_ARG"
echo "  NUM_INSTANCES  : $NUM_INSTANCES  (à partir de pick_index=$PICK_INDEX)"
echo "  SG@50 it.      : $SUBGRAD50_T"
echo "  SG convergé    : T_max=$SUBGRAD_CONV_T_MAX prec=$SUBGRAD_CONV_PRECISION patience=$SUBGRAD_CONV_PATIENCE"
echo "======================================================================"

module load StdEnv/2023 gcc/12.3.0 python/3.11.5 2>/dev/null || \
    module load python/3.11 2>/dev/null || \
    echo "  (aucun module python chargé)"

source "$VENV_PATH/bin/activate"

export LD_LIBRARY_PATH="$REPO/external/gecode/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python "$REPO/gnn/test_inference_checkpoint.py" \
    --ckpt                   "$CKPT"                    \
    --n                      "$N_VAL_ARG"                \
    --N                      "$N_SCEN_ARG"               \
    --depth                  "$DEPTH_ARG"                \
    --num_instances          "$NUM_INSTANCES"            \
    --pick_index             "$PICK_INDEX"               \
    --subgrad50_T            "$SUBGRAD50_T"              \
    --subgrad_conv_T_max     "$SUBGRAD_CONV_T_MAX"       \
    --subgrad_conv_precision "$SUBGRAD_CONV_PRECISION"   \
    --subgrad_conv_patience  "$SUBGRAD_CONV_PATIENCE"    \
    --device                 cpu

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
