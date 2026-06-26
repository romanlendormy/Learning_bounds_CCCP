#!/bin/bash
# run_train_gnn_multi.sh — Entraînement GNN V1LD multi-dataset (n=10,15,20).
#
# Soumettre :
#   sbatch run_train_gnn_multi.sh
#
# Relancer depuis le dernier checkpoint (même run_name = reprise auto) :
#   sbatch run_train_gnn_multi.sh
#
# Surcharger un paramètre :
#   sbatch --export=ALL,N_EPOCHS=200 run_train_gnn_multi.sh

# ── Ressources SLURM ──────────────────────────────────────────────────────────
#SBATCH --account=def-qcappart_gpu
#SBATCH --chdir=/home/romanlen/projects/def-qcappart/romanl/Learning_bounds_CCCP
#SBATCH --job-name=gnn_multi
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=76:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

# ── Paramètres d'entraînement ─────────────────────────────────────────────────
N_EPOCHS="${N_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-3e-4}"           # réduit (1e-3 → 3e-4) ; loss normalisée par opt
ETA_MIN="${ETA_MIN:-1e-5}"  # réduit (1e-4 → 1e-5) ; décroissance plus prononcée
HIDDEN="${HIDDEN:-64}"
TRAIN_FRAC="${TRAIN_FRAC:-0.8}"
SEED="${SEED:-42}"
EVAL_SAMPLE="${EVAL_SAMPLE:-300}"
EVAL_EVERY="${EVAL_EVERY:-5}"
KEEP_CKPTS="${KEEP_CKPTS:-10}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"  # désactivé (était 1e-4 hard-codé)
MAX_NORM="${MAX_NORM:-0.5}"           # clipping gradient (était 1.0 hard-codé)
# 1 CPU pour le process principal, 1 de marge → workers = CPUs - 2
NUM_WORKERS="${NUM_WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-6} - 2 ))}"

# ── Chemins ───────────────────────────────────────────────────────────────────
REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
VENV_PATH="$HOME/envs/learning_bounds"
OUT_DIR="$REPO/gnn/runs"
S30="$REPO/data/train_base_s30"

echo "======================================================================"
echo "  GNN V1LD multi-dataset — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  N_EPOCHS   : $N_EPOCHS  BATCH_SIZE : $BATCH_SIZE"
echo "  LR / eta_min : $LR / $ETA_MIN  HIDDEN : $HIDDEN"
echo "  WD / max_norm : $WEIGHT_DECAY / $MAX_NORM"
echo "  NUM_WORKERS : $NUM_WORKERS  CPUs : ${SLURM_CPUS_PER_TASK:-?}"
echo "  OUT_DIR    : $OUT_DIR"
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

# Bibliothèques Gecode (nécessaires pour la loss V1LD via C++)
export LD_LIBRARY_PATH="$REPO/external/gecode/lib:${LD_LIBRARY_PATH:-}"

# Limiter le parallélisme interne numpy/torch/BLAS aux CPUs alloués
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

# ── Copie des données vers SLURM_TMPDIR (NVMe local) ─────────────────────────
echo ""
echo "=== Copie des données vers SLURM_TMPDIR ==="
DATA_LOCAL="$SLURM_TMPDIR/data/train_base_s30"
mkdir -p "$DATA_LOCAL"

# Fichiers instances (complets)
cp "$S30/instances_n10_m5.txt" "$DATA_LOCAL/"
cp "$S30/instances_n15_m5.txt" "$DATA_LOCAL/"
cp "$S30/instances_n20_m5.txt" "$DATA_LOCAL/"

# Fichiers nodes sous-échantillonnés
cp "$S30/nodes_n10_m5.txt"     "$DATA_LOCAL/"
cp "$S30/nodes_n15_m5_30k.txt" "$DATA_LOCAL/"
cp "$S30/nodes_n20_m5_20k.txt" "$DATA_LOCAL/"

echo "  Copié : $(ls "$DATA_LOCAL/" | wc -l) fichiers dans $DATA_LOCAL"

# Checkpoints dans OUT_DIR (NFS) pour persistance inter-jobs
mkdir -p "$OUT_DIR"
mkdir -p "$REPO/logs"

# ── Lancement ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Lancement — $(date) ==="

python "$REPO/gnn/train_full_v1ld_multi.py" \
    --inst_files  "$DATA_LOCAL/instances_n10_m5.txt" \
                  "$DATA_LOCAL/instances_n15_m5.txt" \
                  "$DATA_LOCAL/instances_n20_m5.txt" \
    --nodes_files "$DATA_LOCAL/nodes_n10_m5.txt"     \
                  "$DATA_LOCAL/nodes_n15_m5_30k.txt" \
                  "$DATA_LOCAL/nodes_n20_m5_20k.txt" \
    --m           5                  \
    --out_dir     "$OUT_DIR"         \
    --n_epochs    "$N_EPOCHS"        \
    --batch_size  "$BATCH_SIZE"      \
    --lr          "$LR"              \
    --eta_min     "$ETA_MIN"         \
    --hidden      "$HIDDEN"          \
    --train_frac  "$TRAIN_FRAC"      \
    --seed        "$SEED"            \
    --eval_sample "$EVAL_SAMPLE"     \
    --eval_every  "$EVAL_EVERY"      \
    --keep_ckpts   "$KEEP_CKPTS"      \
    --weight_decay "$WEIGHT_DECAY"   \
    --max_norm     "$MAX_NORM"       \
    --num_workers  "$NUM_WORKERS"    \
    --run_tag      "${RUN_TAG:-v2}" \
    --device       auto

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "  Résultats dans : $OUT_DIR"
echo "======================================================================"
