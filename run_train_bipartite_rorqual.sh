#!/bin/bash
# run_train_bipartite_rorqual.sh — Run réel (borné) du GNN bipartite V1LD,
# validation avant de démarrer le Tier 3 (warmup, multi-N, diagnostics).
#
# Job CPU, données N=50 réelles (data/train_N50), <=300 epochs.
#
# Soumettre :
#   sbatch run_train_bipartite_rorqual.sh
#
# Ajuster rapidement :
#   sbatch --export=ALL,N_EPOCHS=100,MAX_RECORDS=500 run_train_bipartite_rorqual.sh

# ── Ressources SLURM ──────────────────────────────────────────────────────────
#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=bipartite_val
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

# ── Paramètres ────────────────────────────────────────────────────────────────
N_EPOCHS="${N_EPOCHS:-300}"
MAX_RECORDS="${MAX_RECORDS:-1000}"    # n_graphs train (round-robin stratifié par depth), meme echelle que le baseline homogene single-N50 (job 15080032)
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-3}"
HIDDEN="${HIDDEN:-64}"
MAX_NORM="${MAX_NORM:-2.0}"
EVAL_EVERY="${EVAL_EVERY:-10}"
SEED="${SEED:-42}"

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
VENV_PATH="$HOME/envs/learning_bounds"
OUT_DIR="${OUT_DIR:-$REPO/gnn_bipartite/runs/diag_bipartite_N50}"

echo "======================================================================"
echo "  GNN bipartite — validation Tier 0/1/2 sur N=50 reel — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  N_EPOCHS=$N_EPOCHS  MAX_RECORDS=$MAX_RECORDS  BATCH_SIZE=$BATCH_SIZE"
echo "  LR=$LR  HIDDEN=$HIDDEN  MAX_NORM=$MAX_NORM  SEED=$SEED"
echo "  OUT_DIR : $OUT_DIR"
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

mkdir -p "$OUT_DIR" "$REPO/logs"

python "$REPO/gnn_bipartite/diag_bipartite.py" \
    --data_dir "$REPO/data/train_N50" --n_list 10 --m 5 \
    --n_graphs "$MAX_RECORDS" \
    --n_epochs "$N_EPOCHS" --batch_size "$BATCH_SIZE" \
    --lr "$LR" --hidden "$HIDDEN" \
    --max_norm "$MAX_NORM" --center_lam \
    --use_film --use_theta_attn --use_max_pool \
    --eval_every "$EVAL_EVERY" --eval_set_every "$EVAL_EVERY" \
    --n_cosine 20 --subgrad_max_n 10 \
    --device cpu --seed "$SEED" \
    --out_dir "$OUT_DIR"

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
