#!/bin/bash
# run_sweep_benchmark.sh — SLURM (Compute Canada / Narval) pour le sweep timing GNN V1LD.
#
# Soumettre :
#   sbatch run_sweep_benchmark.sh
#
# Surcharger des paramètres :
#   sbatch --export=ALL,TIME_LIMIT=30,MAX_DEPTH=6 run_sweep_benchmark.sh

# ── Ressources SLURM ──────────────────────────────────────────────────────────
#SBATCH --account=def-qcappart_gpu
#SBATCH --partition=gpubase_bygpu_b3
#SBATCH --chdir=/home/romanlen/projects/def-qcappart/romanl/Learning_bounds_CCCP
#SBATCH --job-name=sweep_gnn
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

# ── Paramètres (surchargeables via --export) ──────────────────────────────────
# Limite en secondes par instance B&B (-1 = illimité, 10 = recommandé)
TIME_LIMIT="${TIME_LIMIT:-10}"
# Profondeur max de l'arbre B&B lors de la collecte
MAX_DEPTH="${MAX_DEPTH:-5}"
# Workers DataLoader (= CPUs alloués - 1)
NUM_WORKERS="${NUM_WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-4} - 1 ))}"
SEED="${SEED:-42}"

# Dossier de sortie : $SCRATCH pour éviter la latence Lustre sur les écritures JSON
OUT_DIR="${OUT_DIR:-$SCRATCH/sweep_benchmark}"

# ── Chemins ───────────────────────────────────────────────────────────────────
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
VENV_PATH="$HOME/envs/learning_bounds"

# ── Environnement ─────────────────────────────────────────────────────────────
echo "=== Sweep Benchmark GNN V1LD ==="
echo "  REPO        : $REPO"
echo "  OUT_DIR     : $OUT_DIR"
echo "  TIME_LIMIT  : ${TIME_LIMIT}s/instance"
echo "  MAX_DEPTH   : $MAX_DEPTH"
echo "  NUM_WORKERS : $NUM_WORKERS"
echo "  SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"
echo "  GPU         : ${SLURM_JOB_GPUS:-?}"
echo "  CPUs        : ${SLURM_CPUS_PER_TASK:-?}"

module load StdEnv/2023 gcc/12.3.0 python/3.11.5 2>/dev/null || \
    module load python/3.11 2>/dev/null || \
    echo "  (aucun module python chargé — utilisation du venv)"

if [ -f "$VENV_PATH/bin/activate" ]; then
    echo "  Activation venv : $VENV_PATH"
    source "$VENV_PATH/bin/activate"
else
    echo "ERREUR : venv introuvable à $VENV_PATH"
    echo "Créer avec : python -m venv $VENV_PATH && pip install -r requirements.txt"
    exit 1
fi

# Bibliothèques Gecode (nécessaires pour le binaire C++ et bnb_solver.so)
export LD_LIBRARY_PATH="$REPO/external/gecode/lib:${LD_LIBRARY_PATH:-}"

# Limiter le parallélisme interne de numpy/torch au CPUs alloués
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Les fichiers temporaires B&B iront dans $SLURM_TMPDIR (NVMe local, ultra-rapide)
export TMPDIR="${SLURM_TMPDIR:-/tmp}"

mkdir -p "$REPO/logs"
mkdir -p "$OUT_DIR"

echo ""
echo "=== Démarré à : $(date) ==="
echo ""

python "$REPO/gnn/sweep_benchmark.py" \
    --time_limit   "$TIME_LIMIT"   \
    --max_depth    "$MAX_DEPTH"    \
    --num_workers  "$NUM_WORKERS"  \
    --seed         "$SEED"         \
    --device       auto            \
    --out_dir      "$OUT_DIR"

echo ""
echo "=== Terminé à : $(date) ==="
echo "  Résultats JSON : $OUT_DIR/sweep_results.json"

# Copier le JSON dans le repo pour archivage (optionnel)
mkdir -p "$REPO/results/sweep_benchmark"
cp "$OUT_DIR/sweep_results.json" \
   "$REPO/results/sweep_benchmark/sweep_results_${SLURM_JOB_ID:-local}.json" 2>/dev/null || true
