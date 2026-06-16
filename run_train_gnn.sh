#!/bin/bash
# run_train_gnn.sh — Soumission SLURM (Compute Canada / Narval) pour l'entraînement GNN V1LD.
#
# Soumettre :
#   sbatch run_train_gnn.sh
#
# Surcharger des paramètres sans éditer le script :
#   sbatch --export=ALL,DATASET=train_base,N_EPOCHS=200 run_train_gnn.sh
#
# Lancer plusieurs configs en parallèle :
#   for ds in train_v1LD train_base; do
#       sbatch --export=ALL,DATASET=$ds,N_EPOCHS=200 run_train_gnn.sh
#   done

# ── Ressources SLURM ──────────────────────────────────────────────────────────
#SBATCH --account=def-qcappart
#SBATCH --job-name=gnn_v1ld
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

# ── Paramètres d'entraînement ─────────────────────────────────────────────────
# Valeurs par défaut ; surchargeable via --export depuis sbatch
DATASET="${DATASET:-train_v1LD}"     # train_v1LD | train_base
N_VAL="${N_VAL:-10}"
N_EPOCHS="${N_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-3}"
ETA_MIN="${ETA_MIN:-1e-5}"
HIDDEN="${HIDDEN:-64}"
TRAIN_FRAC="${TRAIN_FRAC:-0.8}"
SEED="${SEED:-42}"
EVAL_EVERY="${EVAL_EVERY:-10}"
EVAL_SAMPLE_SIZE="${EVAL_SAMPLE_SIZE:-200}"
FINAL_EVAL_N_TRAIN="${FINAL_EVAL_N_TRAIN:-1000}"

# ── Chemins ────────────────────────────────────────────────────────────────────
REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_SRC="$REPO/data"
OUT_DIR="$SCRATCH/gnn_results/${DATASET}_n${N_VAL}_ep${N_EPOCHS}_lr${LR}_h${HIDDEN}"

# Virtualenv Python (à adapter si ton env s'appelle autrement)
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
    echo "  Créer avec : python -m venv $VENV_PATH && pip install torch torch-geometric"
    exit 1
fi

# Limiter le parallélisme interne de numpy/torch au nombre de CPUs alloués
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# ── Copie des données vers le scratch local ($SLURM_TMPDIR) ───────────────────
# $SLURM_TMPDIR est un disque local NVMe ultra-rapide sur chaque nœud de calcul.
# Copier les données là évite les accès répétés sur le réseau Lustre (lent).
echo ""
echo "=== Copie des données vers SLURM_TMPDIR ==="
DATA_LOCAL="$SLURM_TMPDIR/data"
mkdir -p "$DATA_LOCAL/$DATASET"
cp "$DATA_SRC/$DATASET/"*.txt "$DATA_LOCAL/$DATASET/"
echo "  Copié : $(ls $DATA_LOCAL/$DATASET/ | wc -l) fichiers"

# ── Création des dossiers de sortie ───────────────────────────────────────────
mkdir -p "$OUT_DIR"
mkdir -p "$REPO/logs"
echo "  OUT_DIR : $OUT_DIR"

# ── Lancement ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Lancement de l'entraînement ==="
echo "  Démarré à : $(date)"

python "$REPO/gnn/train_full_v1ld.py" \
    --dataset          "$DATASET"           \
    --n_val            "$N_VAL"             \
    --data_dir         "$DATA_LOCAL"        \
    --out_dir          "$OUT_DIR"           \
    --n_epochs         "$N_EPOCHS"          \
    --batch_size       "$BATCH_SIZE"        \
    --lr               "$LR"               \
    --eta_min          "$ETA_MIN"           \
    --hidden           "$HIDDEN"            \
    --train_frac       "$TRAIN_FRAC"        \
    --seed             "$SEED"              \
    --eval_every       "$EVAL_EVERY"        \
    --eval_sample_size "$EVAL_SAMPLE_SIZE"  \
    --final_eval_n_train "$FINAL_EVAL_N_TRAIN" \
    --log_every        5                    \
    --save_every       10                   \
    --device           cpu

echo ""
echo "=== Terminé à : $(date) ==="
echo "  Résultats dans : $OUT_DIR"
