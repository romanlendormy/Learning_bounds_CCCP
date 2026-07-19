#!/bin/bash
# run_overfit_single_rorqual.sh — Overfit du GNN V1LD (config ablation N) sur
# une seule instance (gnn/diag_overfit_single.py).
#
# Job CPU léger : ~1s/epoch pour N=50, ~2.5s/epoch pour N=500 (mesuré sur ce
# cluster, 4 CPUs) — un run à 500 epochs prend donc quelques minutes (N=50)
# à ~25 min (N=500). Pas besoin de GPU pour une seule instance.
#
# Soumettre :
#   sbatch --export=ALL,N_SCEN=50  run_overfit_single_rorqual.sh
#   sbatch --export=ALL,N_SCEN=500 run_overfit_single_rorqual.sh
#
# Autres overrides utiles :
#   sbatch --export=ALL,N_SCEN=500,N_EPOCHS=1000,GRAPH_IDX=1 run_overfit_single_rorqual.sh
#   sbatch --export=ALL,N_SCEN=500,SKIP_SUBGRAD_FULL=1 run_overfit_single_rorqual.sh  # sans la baseline de convergence

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=overfit_single
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

# ── Paramètres ────────────────────────────────────────────────────────────────
N_SCEN="${N_SCEN:-500}"       # nombre de scénarios de l'instance : 50, 200 ou 500
N_VAL="${N_VAL:-10}"
M_VAL="${M_VAL:-5}"
MAX_DEPTH="${MAX_DEPTH:-0}"    # 0 = racine uniquement (première instance)
MAX_LOAD="${MAX_LOAD:-5000}"
GRAPH_IDX="${GRAPH_IDX:-0}"    # 0 = première instance parmi les nœuds filtrés

N_EPOCHS="${N_EPOCHS:-500}"
LR="${LR:-3e-4}"
ETA_MIN="${ETA_MIN:-1e-5}"
EVAL_EVERY="${EVAL_EVERY:-20}"
SEED="${SEED:-42}"
HIDDEN="${HIDDEN:-64}"
MAX_NORM="${MAX_NORM:-2.0}"

# Baseline de convergence sous-gradient (patience/précision) — calculée par
# défaut, affichée sur la figure. SKIP_SUBGRAD_FULL=1 pour la sauter (rapide
# de toute façon : ~1-2s même pour N=500, cf. mesures).
SKIP_SUBGRAD_FULL="${SKIP_SUBGRAD_FULL:-0}"
SUBGRAD_FULL_T_MAX="${SUBGRAD_FULL_T_MAX:-500}"
SUBGRAD_FULL_PATIENCE="${SUBGRAD_FULL_PATIENCE:-30}"
SUBGRAD_FULL_PRECISION="${SUBGRAD_FULL_PRECISION:-0.1}"

DEVICE="${DEVICE:-cpu}"
OUT_DIR="${OUT_DIR:-}"   # vide = défaut du script (gnn/runs/overfit_single/N{..}_inst{..}_..)
DATA_DIR="${DATA_DIR:-}" # ex. data/train_base_s10 — vide = résolution via src_dir_for

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
VENV_PATH="$HOME/envs/learning_bounds"

echo "======================================================================"
echo "  Overfit single-instance — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  N_SCEN=$N_SCEN  N_EPOCHS=$N_EPOCHS  MAX_DEPTH=$MAX_DEPTH  GRAPH_IDX=$GRAPH_IDX"
echo "  LR=$LR  ETA_MIN=$ETA_MIN  EVAL_EVERY=$EVAL_EVERY  SEED=$SEED  HIDDEN=$HIDDEN  MAX_NORM=$MAX_NORM"
echo "  SKIP_SUBGRAD_FULL=$SKIP_SUBGRAD_FULL  DEVICE=$DEVICE"
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

SKIP_SUBGRAD_FLAG=""
[ "$SKIP_SUBGRAD_FULL" = "1" ] && SKIP_SUBGRAD_FLAG="--skip_subgrad_full"

OUT_DIR_ARG=""
[ -n "$OUT_DIR" ] && OUT_DIR_ARG="--out_dir $OUT_DIR"

DATA_DIR_ARG=""
[ -n "$DATA_DIR" ] && DATA_DIR_ARG="--data_dir $DATA_DIR"

RICH_POOL="${RICH_POOL:-0}"
RICH_POOL_FLAG=""
[ "$RICH_POOL" = "1" ] && RICH_POOL_FLAG="--use_rich_pool"

python "$REPO/gnn/diag_overfit_single.py" \
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
    $SKIP_SUBGRAD_FLAG          \
    $RICH_POOL_FLAG             \
    $OUT_DIR_ARG                \
    $DATA_DIR_ARG

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
