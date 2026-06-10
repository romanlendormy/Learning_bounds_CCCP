#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# sweep_bab/sweep_bab_base.sh — Job SLURM pour sweep_bab/sweep_bab_base.py
#
# Avant de soumettre :
#   1. Vérifier que bab_mkp_gecode_base est compilé :
#        salloc --account=def-qcappart --cpus-per-task=4 --mem=8G --time=2:00:00
#        bash compile_cc.sh
#   2. Ajuster --time si besoin.
#
# Soumission (depuis la racine du dépôt) :
#   sbatch sweep_bab/sweep_bab_base.sh
#
# Suivi :
#   squeue -u $USER
#   tail -f logs/sweep_bab_base_<JOBID>.out
# ─────────────────────────────────────────────────────────────────────────────

# ── Ressources SLURM ──────────────────────────────────────────────────────────
#SBATCH --job-name=sweep_bab_base
#SBATCH --output=logs/sweep_bab_base_%j.out
#SBATCH --error=logs/sweep_bab_base_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --account=def-qcappart

# ── Préparation ───────────────────────────────────────────────────────────────
set -euo pipefail

REPO_DIR="$HOME/projects/def-qcappart/romanl/Learning_bounds_CCCP"
cd "$REPO_DIR"

mkdir -p logs results

echo "======================================================================"
echo "  SWEEP BAB BASE — $(date)"
echo "  Job : $SLURM_JOB_ID   Nœud : $(hostname)"
echo "======================================================================"

# ── Modules ───────────────────────────────────────────────────────────────────
module load python/3.11
module load scipy-stack

# ── Bibliothèques dynamiques Gecode (compilées localement) ───────────────────
export LD_LIBRARY_PATH="$REPO_DIR/external/gecode/lib:${LD_LIBRARY_PATH:-}"

# ── Vérification du binaire ───────────────────────────────────────────────────
BAB_BIN="$REPO_DIR/solver_bab/build/bab_mkp_gecode_base"
if [[ ! -f "$BAB_BIN" ]]; then
    echo "ERREUR : binaire introuvable : $BAB_BIN"
    echo "Compiler avec : bash compile_cc.sh  (en session interactive)"
    exit 1
fi
echo "[binaire] Trouvé : $BAB_BIN"

# ── Lancement ─────────────────────────────────────────────────────────────────
echo ""
echo "Démarrage sweep_bab/sweep_bab_base.py ..."
echo ""

python sweep_bab/sweep_bab_base.py

EXIT_CODE=$?
echo ""
echo "======================================================================"
echo "  Terminé — $(date)   code retour : $EXIT_CODE"
echo "  Résultats : $REPO_DIR/results/sweep_bab_base.csv"
echo "======================================================================"

exit $EXIT_CODE
