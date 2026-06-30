#!/bin/bash
# run_train_scaled_n10_500ep.sh — Entraînement GNN V1LD sur instances n=10 uniquement, 500 epochs.
#
# Soumettre :
#   sbatch run_train_scaled_n10_500ep.sh
#
# Surcharger des paramètres :
#   sbatch --export=ALL,LR=1e-4,HIDDEN=256 run_train_scaled_n10_500ep.sh

# ── Ressources SLURM ──────────────────────────────────────────────────────────
#SBATCH --account=def-qcappart_gpu
#SBATCH --chdir=/home/romanlen/projects/def-qcappart/romanl/Learning_bounds_CCCP
#SBATCH --job-name=train_scaled_n10_500ep
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

# ── Paramètres ────────────────────────────────────────────────────────────────
N_EPOCHS="${N_EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-3e-4}"
ETA_MIN="${ETA_MIN:-1e-5}"
HIDDEN="${HIDDEN:-128}"
MAX_NORM="${MAX_NORM:-0.5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
SEED="${SEED:-42}"

# Dataset — n=10 uniquement
TARGET_GRAPHS="${TARGET_GRAPHS:-10000}"
N_SCENS_N10="${N_SCENS_N10:-50 100 200 500}"

# Architecture v21
USE_MAX_POOL="${USE_MAX_POOL:-1}"
USE_THETA_ATTN="${USE_THETA_ATTN:-1}"
USE_FILM="${USE_FILM:-1}"
# Scheduler LR
LR_WARMUP_EP="${LR_WARMUP_EP:-10}"
LR_T0="${LR_T0:-0}"
LR_T_MULT="${LR_T_MULT:-1}"
NO_SCHEDULER="${NO_SCHEDULER:-0}"
CENTER_LAM="${CENTER_LAM:-1}"

# Pondération depth
DEPTH_W0="${DEPTH_W0:-1.0}"
ALPHA_L2="${ALPHA_L2:-0.0}"
LAM_MAX="${LAM_MAX:-0}"

# Warmup supervisé
N_WARMUP="${N_WARMUP:-10}"
WARMUP_LR="${WARMUP_LR:-1e-3}"
WARMUP_T="${WARMUP_T:-50}"
WARMUP_MAX_N="${WARMUP_MAX_N:-10}"

# Figure qualité
EVAL_MIXED="${EVAL_MIXED:-1}"
SKIP_QUALITY="${SKIP_QUALITY:-0}"
QUALITY_T="${QUALITY_T:-100}"
QUALITY_MAX_N="${QUALITY_MAX_N:-10}"

# Baseline sous-gradient pour pct_vs_sg
EVAL_SG_T="${EVAL_SG_T:-60}"

# Évaluation & infra
EVAL_EVERY="${EVAL_EVERY:-10}"
EVAL_SAMPLE="${EVAL_SAMPLE:-400}"
KEEP_CKPTS="${KEEP_CKPTS:-3}"
NUM_WORKERS="${NUM_WORKERS:-4}"
RUN_TAG="${RUN_TAG:-v6_n10_500ep}"

# ── Chemins ───────────────────────────────────────────────────────────────────
REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
VENV_PATH="$HOME/envs/learning_bounds"
OUT_DIR="${OUT_DIR:-$REPO/gnn/runs/train_scaled}"

echo "======================================================================"
echo "  train_scaled_n10_500ep — n=10 only — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  N_EPOCHS=$N_EPOCHS  BATCH_SIZE=$BATCH_SIZE  LR=$LR  HIDDEN=$HIDDEN"
echo "  TARGET_GRAPHS=$TARGET_GRAPHS (n=10)  N_SCENS_N10=$N_SCENS_N10"
echo "  USE_MAX_POOL=$USE_MAX_POOL  USE_THETA_ATTN=$USE_THETA_ATTN  USE_FILM=$USE_FILM"
echo "  LR_WARMUP_EP=$LR_WARMUP_EP  LR_T0=$LR_T0  CENTER_LAM=$CENTER_LAM"
echo "  DEPTH_W0=$DEPTH_W0  N_WARMUP=$N_WARMUP  WARMUP_T=$WARMUP_T"
echo "  EVAL_EVERY=$EVAL_EVERY  RUN_TAG=$RUN_TAG"
echo "  OUT_DIR : $OUT_DIR"
echo "======================================================================"

# ── Modules & venv ────────────────────────────────────────────────────────────
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

# ── Données → SLURM_TMPDIR ────────────────────────────────────────────────────
echo ""
echo "=== Copie des données vers SLURM_TMPDIR ==="
DATA_LOCAL="${SLURM_TMPDIR:-/tmp}/data"
mkdir -p "$DATA_LOCAL"

S30="$REPO/data/train_base_s30"
cp "$S30/instances_n10_m5.txt" "$DATA_LOCAL/"
cp "$S30/nodes_n10_m5.txt"     "$DATA_LOCAL/"
echo "  OK train_base_s30 (N=500 pour n=10)"

for N_VAL in $N_SCENS_N10; do
    if [ "$N_VAL" = "500" ]; then
        continue
    fi
    SRC_DIR="$REPO/data/train_N${N_VAL}"
    if [ -f "$SRC_DIR/instances_n10_m5.txt" ] && [ -f "$SRC_DIR/nodes_n10_m5.txt" ]; then
        mkdir -p "$DATA_LOCAL/train_N${N_VAL}"
        cp "$SRC_DIR/instances_n10_m5.txt" "$DATA_LOCAL/train_N${N_VAL}/"
        cp "$SRC_DIR/nodes_n10_m5.txt"     "$DATA_LOCAL/train_N${N_VAL}/"
        echo "  OK train_N${N_VAL}"
    else
        echo "  WARN: train_N${N_VAL} introuvable ou incomplet — ignoré"
    fi
done

mkdir -p "$DATA_LOCAL/train_base_s30"
cp "$S30/instances_n10_m5.txt" "$DATA_LOCAL/train_base_s30/"
cp "$S30/nodes_n10_m5.txt"     "$DATA_LOCAL/train_base_s30/"

mkdir -p "$OUT_DIR" "$REPO/logs"

# ── Flags optionnels ──────────────────────────────────────────────────────────
MAX_POOL_FLAG=""
[ "${USE_MAX_POOL:-1}" = "1" ] && MAX_POOL_FLAG="--use_max_pool"
THETA_ATTN_FLAG=""
[ "${USE_THETA_ATTN:-1}" = "1" ] && THETA_ATTN_FLAG="--use_theta_attn"
FILM_FLAG=""
[ "${USE_FILM:-1}" = "1" ] && FILM_FLAG="--use_film"
LR_WARMUP_ARG=""
[ "${LR_WARMUP_EP:-0}" != "0" ] && LR_WARMUP_ARG="--lr_warmup_ep ${LR_WARMUP_EP}"
LR_T0_ARG=""
[ "${LR_T0:-0}" != "0" ] && LR_T0_ARG="--lr_T0 ${LR_T0} --lr_T_mult ${LR_T_MULT}"
NO_SCHEDULER_FLAG=""
[ "${NO_SCHEDULER:-0}" = "1" ] && NO_SCHEDULER_FLAG="--no_scheduler"
CENTER_LAM_FLAG=""
[ "${CENTER_LAM:-1}" = "1" ] && CENTER_LAM_FLAG="--center_lam"
EVAL_MIXED_FLAG=""
[ "${EVAL_MIXED:-1}" = "1" ] && EVAL_MIXED_FLAG="--eval_mixed"
SKIP_QUALITY_FLAG=""
[ "${SKIP_QUALITY:-0}" = "1" ] && SKIP_QUALITY_FLAG="--skip_quality"

# ── Lancement ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Lancement — $(date) ==="

python "$REPO/gnn/train_scaled_v1ld.py" \
    --data_dir      "$DATA_LOCAL"        \
    --out_dir       "$OUT_DIR"           \
    --n_list        10                   \
    --n_scens_n10   $N_SCENS_N10        \
    --m             5                    \
    --target_graphs "$TARGET_GRAPHS"     \
    --n_epochs      "$N_EPOCHS"          \
    --batch_size    "$BATCH_SIZE"        \
    --lr            "$LR"               \
    --eta_min       "$ETA_MIN"           \
    --hidden        "$HIDDEN"            \
    --max_norm      "$MAX_NORM"          \
    --weight_decay  "$WEIGHT_DECAY"      \
    --depth_w0      "$DEPTH_W0"          \
    --alpha_l2      "$ALPHA_L2"          \
    --lam_max       "$LAM_MAX"           \
    --n_warmup      "$N_WARMUP"          \
    --warmup_lr     "$WARMUP_LR"         \
    --warmup_T      "$WARMUP_T"          \
    --warmup_max_n  "$WARMUP_MAX_N"      \
    --eval_every    "$EVAL_EVERY"        \
    --eval_sample   "$EVAL_SAMPLE"       \
    --keep_ckpts    "$KEEP_CKPTS"        \
    --num_workers   "$NUM_WORKERS"       \
    --seed          "$SEED"              \
    --run_tag       "$RUN_TAG"           \
    --device        auto                 \
    --quality_T     "$QUALITY_T"          \
    --quality_max_n "$QUALITY_MAX_N"      \
    --eval_sg_T     "$EVAL_SG_T"          \
    $MAX_POOL_FLAG                        \
    $THETA_ATTN_FLAG                      \
    $FILM_FLAG                            \
    $LR_WARMUP_ARG                        \
    $LR_T0_ARG                            \
    $NO_SCHEDULER_FLAG                    \
    $CENTER_LAM_FLAG                      \
    $EVAL_MIXED_FLAG                      \
    $SKIP_QUALITY_FLAG

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "  Résultats dans : $OUT_DIR"
echo "======================================================================"
