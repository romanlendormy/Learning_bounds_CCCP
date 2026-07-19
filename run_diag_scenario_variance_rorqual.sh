#!/bin/bash
# run_diag_scenario_variance_rorqual.sh — Localise l'étage du GNN V1LD où la
# différenciation inter-scénarios de λ s'effondre (gnn/diag_scenario_variance.py).
#
# Job CPU léger, même ordre de grandeur que run_overfit_single_rorqual.sh
# (500 epochs sur une seule instance) + calcul ICC par étage (négligeable).
#
# Soumettre :
#   sbatch --export=ALL,ARCH=homogene,DATA_DIR=data/train_base_s10,N_VAL=10,N_SCEN=20 \
#       run_diag_scenario_variance_rorqual.sh
#   sbatch --export=ALL,ARCH=sd_edit,DATA_DIR=data/train_base_s10,N_VAL=10,N_SCEN=100 \
#       run_diag_scenario_variance_rorqual.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=diag_scen_var
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

ARCH="${ARCH:-homogene}"      # homogene | sd_edit
N_SCEN="${N_SCEN:-100}"
N_VAL="${N_VAL:-10}"
M_VAL="${M_VAL:-5}"
MAX_DEPTH="${MAX_DEPTH:-0}"
MAX_LOAD="${MAX_LOAD:-5000}"
GRAPH_IDX="${GRAPH_IDX:-0}"

N_EPOCHS="${N_EPOCHS:-500}"
LR="${LR:-3e-4}"
ETA_MIN="${ETA_MIN:-1e-5}"
EVAL_EVERY="${EVAL_EVERY:-20}"
SEED="${SEED:-42}"
HIDDEN="${HIDDEN:-64}"
MAX_NORM="${MAX_NORM:-2.0}"

SUBGRAD_FULL_T_MAX="${SUBGRAD_FULL_T_MAX:-500}"
SUBGRAD_FULL_PATIENCE="${SUBGRAD_FULL_PATIENCE:-30}"
SUBGRAD_FULL_PRECISION="${SUBGRAD_FULL_PRECISION:-1}"

DEVICE="${DEVICE:-cpu}"
OUT_DIR="${OUT_DIR:-}"
DATA_DIR="${DATA_DIR:-}"

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
VENV_PATH="$HOME/envs/learning_bounds"

echo "======================================================================"
echo "  Diagnostic scenario variance (ICC par étage) — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  ARCH=$ARCH  N_VAL=$N_VAL  N_SCEN=$N_SCEN  N_EPOCHS=$N_EPOCHS  DATA_DIR=$DATA_DIR"
echo "======================================================================"

module load StdEnv/2023 gcc/12.3.0 python/3.11.5 2>/dev/null || \
    module load python/3.11 2>/dev/null || \
    echo "  (aucun module python chargé)"

if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "ERREUR : venv introuvable à $VENV_PATH"
    exit 1
fi

export LD_LIBRARY_PATH="$REPO/external/gecode/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

mkdir -p "$REPO/logs"

OUT_DIR_ARG=""
[ -n "$OUT_DIR" ] && OUT_DIR_ARG="--out_dir $OUT_DIR"

DATA_DIR_ARG=""
[ -n "$DATA_DIR" ] && DATA_DIR_ARG="--data_dir $DATA_DIR"

python "$REPO/gnn/diag_scenario_variance.py" \
    --arch       "$ARCH"       \
    --n_scen     "$N_SCEN"     \
    --n_val      "$N_VAL"      \
    --m_val      "$M_VAL"      \
    --max_depth  "$MAX_DEPTH"  \
    --max_load   "$MAX_LOAD"   \
    --graph_idx  "$GRAPH_IDX"  \
    --n_epochs   "$N_EPOCHS"   \
    --lr         "$LR"         \
    --eta_min    "$ETA_MIN"    \
    --eval_every "$EVAL_EVERY" \
    --seed       "$SEED"       \
    --hidden     "$HIDDEN"     \
    --max_norm   "$MAX_NORM"   \
    --subgrad_full_T_max     "$SUBGRAD_FULL_T_MAX"     \
    --subgrad_full_patience  "$SUBGRAD_FULL_PATIENCE"  \
    --subgrad_full_precision "$SUBGRAD_FULL_PRECISION" \
    --device     "$DEVICE"     \
    $OUT_DIR_ARG                \
    $DATA_DIR_ARG

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
