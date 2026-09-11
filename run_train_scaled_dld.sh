#!/bin/bash
# run_train_scaled_dld.sh — Entraînement GNN DLD (SSP) sur ~10k graphes,
# 6 combos (E,T,N), stratifié par depth (cf. gnn_ssp/train_scaled_dld.py).
#
# Soumettre :
#   sbatch run_train_scaled_dld.sh
#
# Surcharger des paramètres :
#   sbatch --export=ALL,N_EPOCHS=200,HIDDEN=128 run_train_scaled_dld.sh
#
# Reprise automatique : si le job est coupé par SLURM avant la fin des
# N_EPOCHS, relancer la même commande reprend depuis le dernier checkpoint
# (vérifié en calibrage — cf. find_latest_checkpoint dans le script Python).

# ── Ressources SLURM ──────────────────────────────────────────────────────────
#SBATCH --account=def-qcappart
#SBATCH --chdir=/home/romanlen/projects/def-qcappart/romanl/Learning_bounds_CCCP
#SBATCH --job-name=train_scaled_dld
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

# ── Paramètres ────────────────────────────────────────────────────────────────
N_EPOCHS="${N_EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-3e-4}"
ETA_MIN="${ETA_MIN:-1e-5}"
HIDDEN="${HIDDEN:-64}"
MAX_NORM="${MAX_NORM:-0.5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
SEED="${SEED:-42}"

TARGET_GRAPHS="${TARGET_GRAPHS:-10000}"
ROOT_FRAC="${ROOT_FRAC:-0.05}"
RESERVOIR_K="${RESERVOIR_K:-50}"
COMBOS="${COMBOS:-E3_T12_N20 E3_T12_N50 E4_T12_N20 E4_T12_N50 E4_T8_N20 E4_T8_N50}"

CENTER_LAM="${CENTER_LAM:-1}"

# Warmup supervisé (racines train, λ* sous-gradient Polyak)
N_WARMUP="${N_WARMUP:-10}"
WARMUP_LR="${WARMUP_LR:-1e-3}"
WARMUP_T="${WARMUP_T:-50}"

# Évaluation
EVAL_EVERY="${EVAL_EVERY:-5}"
EVAL_SAMPLE="${EVAL_SAMPLE:-200}"
EVAL_SG_T="${EVAL_SG_T:-50}"
EVAL_MIXED="${EVAL_MIXED:-0}"

KEEP_CKPTS="${KEEP_CKPTS:-3}"
NUM_WORKERS="${NUM_WORKERS:-0}"
RUN_TAG="${RUN_TAG:-v1_10k}"

# ── Chemins ───────────────────────────────────────────────────────────────────
REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
VENV_PATH="$HOME/envs/learning_bounds"
OUT_DIR="${OUT_DIR:-$REPO/gnn_ssp/runs/train_scaled}"

echo "======================================================================"
echo "  train_scaled_dld (SSP) — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  N_EPOCHS=$N_EPOCHS  BATCH_SIZE=$BATCH_SIZE  LR=$LR  HIDDEN=$HIDDEN"
echo "  TARGET_GRAPHS=$TARGET_GRAPHS  ROOT_FRAC=$ROOT_FRAC  RESERVOIR_K=$RESERVOIR_K"
echo "  COMBOS=$COMBOS"
echo "  N_WARMUP=$N_WARMUP  WARMUP_T=$WARMUP_T  EVAL_SG_T=$EVAL_SG_T  CENTER_LAM=$CENTER_LAM"
echo "  RUN_TAG=$RUN_TAG"
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

export LD_LIBRARY_PATH="$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# ── Données → SLURM_TMPDIR (évite la contention Lustre pendant le run) ────────
echo ""
echo "=== Copie des données vers SLURM_TMPDIR ==="
DATA_LOCAL="${SLURM_TMPDIR:-/tmp}/data"
mkdir -p "$DATA_LOCAL/train_ssp_base_s10"
for COMBO in $COMBOS; do
    cp "$REPO/data/train_ssp_base_s10/instances_${COMBO}.txt" "$DATA_LOCAL/train_ssp_base_s10/"
    cp "$REPO/data/train_ssp_base_s10/nodes_${COMBO}.txt"     "$DATA_LOCAL/train_ssp_base_s10/"
    echo "  OK $COMBO"
done

mkdir -p "$OUT_DIR" "$REPO/logs"

CENTER_LAM_FLAG=""
[ "${CENTER_LAM:-1}" = "1" ] && CENTER_LAM_FLAG="--center_lam"
EVAL_MIXED_FLAG=""
[ "${EVAL_MIXED:-0}" = "1" ] && EVAL_MIXED_FLAG="--eval_mixed"

# ── Lancement ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Lancement — $(date) ==="

python "$REPO/gnn_ssp/train_scaled_dld.py" \
    --data_dir      "$DATA_LOCAL"        \
    --out_dir       "$OUT_DIR"           \
    --combos        $COMBOS              \
    --target_graphs "$TARGET_GRAPHS"     \
    --root_frac     "$ROOT_FRAC"         \
    --reservoir_k   "$RESERVOIR_K"       \
    --n_epochs      "$N_EPOCHS"          \
    --batch_size    "$BATCH_SIZE"        \
    --lr            "$LR"               \
    --eta_min       "$ETA_MIN"           \
    --hidden        "$HIDDEN"            \
    --max_norm      "$MAX_NORM"          \
    --weight_decay  "$WEIGHT_DECAY"      \
    --n_warmup      "$N_WARMUP"          \
    --warmup_lr     "$WARMUP_LR"         \
    --warmup_T      "$WARMUP_T"          \
    --eval_every    "$EVAL_EVERY"        \
    --eval_sample   "$EVAL_SAMPLE"       \
    --eval_sg_T     "$EVAL_SG_T"         \
    --keep_ckpts    "$KEEP_CKPTS"        \
    --num_workers   "$NUM_WORKERS"       \
    --seed          "$SEED"              \
    --run_tag       "$RUN_TAG"           \
    --device        cpu                  \
    $CENTER_LAM_FLAG                      \
    $EVAL_MIXED_FLAG

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "  Résultats dans : $OUT_DIR"
echo "======================================================================"
