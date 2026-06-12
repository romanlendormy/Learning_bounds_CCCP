#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# collect/collect_large.sh — Job SLURM pour collect/collect_large.py
#
# Avant de soumettre :
#   1. Compiler les binaires en session interactive :
#        salloc --account=def-qcappart --cpus-per-task=4 --mem=8G --time=2:00:00
#        bash compile_cc.sh
#   2. Ajuster les paramètres ci-dessous si besoin (n, m, N, n_instances).
#
# Soumission (depuis la racine du dépôt) :
#   sbatch collect/collect_large.sh
#
# Suivi :
#   squeue -u $USER
#   tail -f logs/collect_large_<JOBID>.out
# ─────────────────────────────────────────────────────────────────────────────

# ── Ressources SLURM ──────────────────────────────────────────────────────────
#SBATCH --job-name=collect_large
#SBATCH --output=logs/collect_large_%j.out
#SBATCH --error=logs/collect_large_%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --account=def-qcappart

# ── Préparation ───────────────────────────────────────────────────────────────
set -euo pipefail

REPO_DIR="$HOME/projects/def-qcappart/romanl/Learning_bounds_CCCP"
cd "$REPO_DIR"

mkdir -p logs

# ── Chemins (définis tôt pour la restauration) ────────────────────────────────
MODE="v1cld"
LOCAL_OUTDIR="$SLURM_TMPDIR/train"
LOCAL_RESULT="${LOCAL_OUTDIR}_${MODE}"
DEST="$REPO_DIR/data/train_${MODE}"
mkdir -p "$DEST" "$LOCAL_RESULT"

# Restaurer les données existantes depuis le stockage permanent (progress file inclus)
if [[ -n "$(ls -A "$DEST" 2>/dev/null)" ]]; then
    echo "[init] Restauration depuis $DEST ..."
    rsync -a "$DEST/" "$LOCAL_RESULT/"
fi

echo "======================================================================"
echo "  COLLECT LARGE — $(date)"
echo "  Job : $SLURM_JOB_ID   Nœud : $(hostname)"
echo "======================================================================"

# ── Modules ───────────────────────────────────────────────────────────────────
module load python/3.11
module load scipy-stack

# ── Bibliothèques dynamiques Gecode et OR-Tools (compilées localement) ────────
# Copier les .so vers $SLURM_TMPDIR (node-local) pour éviter le throttling Lustre
# lors des appels répétés au binaire C++ (recommandé par Compute Canada pour I/O intensif)
LOCAL_LIB="$SLURM_TMPDIR/lib"
mkdir -p "$LOCAL_LIB"
cp "$REPO_DIR/external/gecode/lib/"*.so* "$LOCAL_LIB/"
cp "$REPO_DIR/external/ortools/lib/"*.so* "$LOCAL_LIB/"
export LD_LIBRARY_PATH="$LOCAL_LIB:${LD_LIBRARY_PATH:-}"

# ── Vérification du binaire ───────────────────────────────────────────────────
COLLECT_BIN="$REPO_DIR/solver_bab/build/bab_mkp_gecode_collect"
if [[ ! -f "$COLLECT_BIN" ]]; then
    echo "ERREUR : binaire introuvable : $COLLECT_BIN"
    echo "Compiler en session interactive avec : bash compile_cc.sh"
    exit 1
fi
echo "[binaire] Trouvé : $COLLECT_BIN"

# ── Lancement de la collecte ──────────────────────────────────────────────────
echo ""
echo "Démarrage collect/collect_large.py ..."
echo "  sortie temporaire : ${LOCAL_RESULT}/"
echo ""

python collect/collect_large.py \
    --n 10 \
    --m 5 \
    --N 50 \
    --n_instances 1 \
    --epsilon 0.1 \
    --correlation weakly_correlated \
    --mode "$MODE" \
    --patience_mode \
    --patience_tol 1.0 \
    --outdir "$LOCAL_OUTDIR" \
    --binary "$COLLECT_BIN"

EXIT_CODE=$?

echo ""
echo "Copie des résultats vers $DEST ..."
rsync -a "$LOCAL_RESULT/" "$DEST/"

echo ""
echo "======================================================================"
echo "  Terminé — $(date)   code retour : $EXIT_CODE"
echo "  Résultats : $DEST/"
echo "======================================================================"

exit $EXIT_CODE
