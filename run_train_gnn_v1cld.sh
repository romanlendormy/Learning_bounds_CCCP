#!/bin/bash
# run_train_gnn_v1cld.sh — Soumission SLURM (Compute Canada / Narval) pour l'entraînement GNN V1CLD.
#
# Soumettre :
#   sbatch run_train_gnn_v1cld.sh
#
# Surcharger des paramètres sans éditer le script :
#   sbatch --export=ALL,DATASET=train_v1CLD,N_EPOCHS=50 run_train_gnn_v1cld.sh

# ── Ressources SLURM ──────────────────────────────────────────────────────────
#SBATCH --account=def-qcappart_gpu
#SBATCH --partition=gpubase_bygpu_b3
#SBATCH --chdir=/home/romanlen/projects/def-qcappart/romanl/Learning_bounds_CCCP
#SBATCH --job-name=gnn_v1cld
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

# ── Paramètres d'entraînement ─────────────────────────────────────────────────
DATASET="${DATASET:-train_base}"
M_FILTER="${M_FILTER:-5}"
MAX_N="${MAX_N:-500}"
N_VAL="${N_VAL:-10}"
N_EPOCHS="${N_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-3}"
ETA_MIN="${ETA_MIN:-1e-5}"
TRAIN_FRAC="${TRAIN_FRAC:-0.8}"
SEED="${SEED:-42}"
EVAL_EVERY="${EVAL_EVERY:-10}"
EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-200}"
FINAL_EVAL_N_TRAIN="${FINAL_EVAL_N_TRAIN:-1000}"
NUM_WORKERS="${NUM_WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-4} - 1 ))}"

# ── Chemins ────────────────────────────────────────────────────────────────────
REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
DATA_SRC="$REPO/data"
OUT_DIR="$REPO/results/${DATASET}_n${N_VAL}_ep${N_EPOCHS}_lr${LR}"

VENV_PATH="$HOME/envs/learning_bounds"

# ── Environnement ──────────────────────────────────────────────────────────────
echo "=== Environnement ==="
echo "  REPO      : $REPO"
echo "  DATASET   : $DATASET"
echo "  N_EPOCHS  : $N_EPOCHS"
echo "  OUT_DIR   : $OUT_DIR"
echo "  SLURM_JOB_ID : ${SLURM_JOB_ID:-local}"
echo "  CPUs      : ${SLURM_CPUS_PER_TASK:-?}"

module load StdEnv/2023 gcc/12.3.0 python/3.11.5 2>/dev/null || \
    module load python/3.11 2>/dev/null || \
    echo "  (aucun module python chargé — utilisation du python du venv)"

if [ -f "$VENV_PATH/bin/activate" ]; then
    echo "  Activation venv : $VENV_PATH"
    source "$VENV_PATH/bin/activate"
else
    echo "  ERREUR : venv introuvable à $VENV_PATH"
    exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# ── Copie des données vers le scratch local ($SLURM_TMPDIR) ───────────────────
echo ""
echo "=== Copie des données vers SLURM_TMPDIR ==="
DATA_LOCAL="$SLURM_TMPDIR/data"
mkdir -p "$DATA_LOCAL/$DATASET"
cp "$DATA_SRC/$DATASET/"*.txt "$DATA_LOCAL/$DATASET/"
echo "  Copié : $(ls $DATA_LOCAL/$DATASET/ | wc -l) fichiers"

# ── Création des dossiers de sortie ───────────────────────────────────────────
mkdir -p "$OUT_DIR"
mkdir -p "$REPO/logs"

# ── Lancement ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Lancement de l'entraînement V1CLD ==="
echo "  Démarré à : $(date)"

M_ARG=""
[[ -n "$M_FILTER" ]] && M_ARG="--m $M_FILTER"

MAX_N_ARG=""
[[ -n "$MAX_N" ]] && MAX_N_ARG="--max_N $MAX_N"

python "$REPO/gnn/train_full_v1cld.py" \
    --dataset          "$DATASET"             \
    --n_val            "$N_VAL"               \
    ${M_ARG}                                  \
    ${MAX_N_ARG}                              \
    --data_dir         "$DATA_LOCAL"          \
    --out_dir          "$OUT_DIR"             \
    --n_epochs         "$N_EPOCHS"            \
    --batch_size       "$BATCH_SIZE"          \
    --lr               "$LR"                  \
    --eta_min          "$ETA_MIN"             \
    --train_frac       "$TRAIN_FRAC"          \
    --seed             "$SEED"                \
    --eval_every       "$EVAL_EVERY"          \
    --eval_sample_size "$EVAL_SAMPLE_SIZE"    \
    --final_eval_n_train "$FINAL_EVAL_N_TRAIN" \
    --num_workers      "$NUM_WORKERS"         \
    --log_every        1                      \
    --save_every       1                      \
    --device           auto

echo ""
echo "=== Terminé à : $(date) ==="
echo "  Résultats dans : $OUT_DIR"
