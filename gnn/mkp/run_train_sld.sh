#!/bin/bash
# gnn/mkp/run_train_sld.sh — Entraînement du GNN SLD (graphe sd_edit) sur
# ~10 000 graphes n=10/20, N=20/50/100 (data/train_base_s10, cf. collect/mkp/run_collect_mkp.sh).
# Valeurs par défaut = réglages du modèle de l'article
# (models/mkp/sld/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt).
# Reprise automatique depuis le dernier checkpoint si le job est coupé.
#
# Soumettre depuis la racine du dépôt (adapter --account au cluster) :
#   sbatch gnn/mkp/run_train_sld.sh
# Surcharger des paramètres :
#   sbatch --export=ALL,N_EPOCHS=200,DEPTH_W0=2.0 gnn/mkp/run_train_sld.sh

# ── Ressources SLURM ──────────────────────────────────────────────────────────
#SBATCH --account=def-qcappart
#SBATCH --job-name=train_scaled_sdedit
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

# Dataset — n=10/20, N=20/50/100 (data/train_base_s10)
TARGET_GRAPHS="${TARGET_GRAPHS:-10000}"    # ~1667/combo réparti également sur les 6 (n,N)
ROOT_FRAC="${ROOT_FRAC:-0.05}"
RESERVOIR_K="${RESERVOIR_K:-2}"
N_SCENS="${N_SCENS:-20 50 100}"
N_LIST="${N_LIST:-10 20}"

# Architecture (tête h_item_k+h_scen_i+g_pool, cf. train_SLD_sd_edit.py::GNN)
USE_MAX_POOL="${USE_MAX_POOL:-1}"
# Scheduler LR :
#   LR_WARMUP_EP>0  → LinearWarmup+CosineDecay (recommandé, pas de restarts)
#   LR_T0>0         → CosineAnnealingWarmRestarts T_0
#   sinon           → CosineAnnealingLR
LR_WARMUP_EP="${LR_WARMUP_EP:-10}"
LR_T0="${LR_T0:-0}"
LR_T_MULT="${LR_T_MULT:-1}"
NO_SCHEDULER="${NO_SCHEDULER:-0}"
# Centrage des λ sur les scénarios (élimine dérive de jauge) : 1 = activé
CENTER_LAM="${CENTER_LAM:-1}"

# Pondération depth
DEPTH_W0="${DEPTH_W0:-1.0}"
ALPHA_L2="${ALPHA_L2:-0.0}"

# Warmup supervisé — n<=20 (couvre n=10 ET n=20, cf. warmup_max_n)
N_WARMUP="${N_WARMUP:-10}"
WARMUP_LR="${WARMUP_LR:-1e-3}"
WARMUP_T="${WARMUP_T:-50}"
WARMUP_MAX_N="${WARMUP_MAX_N:-20}"

# Éval set stratifié par depth (comparable à loss_train)
EVAL_MIXED="${EVAL_MIXED:-1}"

# Figure qualité GNN vs sous-gradient (en fin d'entraînement)
SKIP_QUALITY="${SKIP_QUALITY:-0}"
QUALITY_T="${QUALITY_T:-100}"
QUALITY_MAX_N="${QUALITY_MAX_N:-20}"

# Baseline sous-gradient pour pct_vs_sg durant l'éval (itérations par graphe)
EVAL_SG_T="${EVAL_SG_T:-50}"

# Évaluation & infra — EVAL_EVERY=10 : log la généralisation par n tous les 10 epochs
EVAL_EVERY="${EVAL_EVERY:-10}"
EVAL_SAMPLE="${EVAL_SAMPLE:-400}"
KEEP_CKPTS="${KEEP_CKPTS:-3}"
NUM_WORKERS="${NUM_WORKERS:-4}"
RUN_TAG="${RUN_TAG:-v1_sdedit_500ep}"

# ── Chemins ───────────────────────────────────────────────────────────────────
REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )}"   # soumettre depuis la racine du dépôt
VENV_PATH="${VENV_PATH:-$HOME/venvs/cccp}"   # cf. README, « Environment »
OUT_DIR="${OUT_DIR:-$REPO/runs/mkp_sld}"

echo "======================================================================"
echo "  train_scaled_sd_edit — n=10/20, N=20/50/100 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  N_EPOCHS=$N_EPOCHS  BATCH_SIZE=$BATCH_SIZE  LR=$LR  HIDDEN=$HIDDEN"
echo "  TARGET_GRAPHS=$TARGET_GRAPHS  ROOT_FRAC=$ROOT_FRAC  RESERVOIR_K=$RESERVOIR_K  N_SCENS=$N_SCENS  N_LIST=$N_LIST"
echo "  USE_MAX_POOL=$USE_MAX_POOL  CENTER_LAM=$CENTER_LAM"
echo "  LR_WARMUP_EP=$LR_WARMUP_EP  LR_T0=$LR_T0  LR_T_MULT=$LR_T_MULT"
echo "  DEPTH_W0=$DEPTH_W0  N_WARMUP=$N_WARMUP  WARMUP_T=$WARMUP_T  WARMUP_MAX_N=$WARMUP_MAX_N"
echo "  EVAL_EVERY=$EVAL_EVERY  EVAL_MIXED=$EVAL_MIXED  EVAL_SG_T=$EVAL_SG_T  RUN_TAG=$RUN_TAG"
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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

# ── Données → SLURM_TMPDIR ────────────────────────────────────────────────────
# train_base_s10 : un fichier d'instances et un fichier de nœuds par (n, N).
echo ""
echo "=== Copie des données vers SLURM_TMPDIR ==="
DATA_LOCAL="${SLURM_TMPDIR:-/tmp}/data"
mkdir -p "$DATA_LOCAL/train_base_s10"

S10="$REPO/data/train_base_s10"
for N_VAL in $N_SCENS; do
    for N_VAL_ITEM in $N_LIST; do
        INST="instances_n${N_VAL_ITEM}_m5_N${N_VAL}.txt"
        NODES="nodes_n${N_VAL_ITEM}_m5_N${N_VAL}.txt"
        if [ -f "$S10/$INST" ] && [ -f "$S10/$NODES" ]; then
            cp "$S10/$INST" "$S10/$NODES" "$DATA_LOCAL/train_base_s10/"
            echo "  OK n=${N_VAL_ITEM} N=${N_VAL}"
        else
            echo "  WARN: n=${N_VAL_ITEM} N=${N_VAL} introuvable dans $S10 — ignoré"
        fi
    done
done

mkdir -p "$OUT_DIR" "$REPO/logs"

# ── Flags optionnels ──────────────────────────────────────────────────────────
MAX_POOL_FLAG=""
[ "${USE_MAX_POOL:-1}" = "1" ] && MAX_POOL_FLAG="--use_max_pool"
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

python "$REPO/gnn/mkp/train_scaled_sd_edit.py" \
    --data_dir      "$DATA_LOCAL"        \
    --out_dir       "$OUT_DIR"           \
    --n_list        $N_LIST              \
    --n_scens       $N_SCENS             \
    --m             5                    \
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
    --depth_w0      "$DEPTH_W0"          \
    --alpha_l2      "$ALPHA_L2"          \
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
