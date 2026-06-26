#!/bin/bash
# collect/run_collect_base.sh — Job SLURM pour collect/collect_large_base.py
#
# Avant de soumettre :
#   Compiler le binaire en session interactive :
#     salloc --account=def-qcappart_cpu --partition=cpubase_bycore_b4 --cpus-per-task=4 --mem=8G --time=2:00:00
#     bash compile_cc.sh
#
# Soumission (depuis la racine du dépôt) :
#   sbatch collect/run_collect_base.sh
#
# Surcharger des paramètres :
#   sbatch --export=ALL,N_INSTANCES=200,N_VAL=20 collect/run_collect_base.sh
#   sbatch --export=ALL,N_SCEN=100,N_INSTANCES=200 collect/run_collect_base.sh
#
# Résultats dans data/train_N{N_SCEN}/ (ex. data/train_N100/).
# Exception : N_SCEN=500 → data/train_base_s30/ (compat. existante).

# ── Ressources SLURM ──────────────────────────────────────────────────────────
#SBATCH --account=def-qcappart_cpu
#SBATCH --partition=cpubase_bycore_b4
#SBATCH --chdir=/home/romanlen/projects/def-qcappart/romanl/Learning_bounds_CCCP
#SBATCH --job-name=collect_base
#SBATCH --output=logs/collect_base_%j.out
#SBATCH --error=logs/collect_base_%j.err
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )}"

# ── Paramètres (surchargeables via --export) ──────────────────────────────────
N_VAL="${N_VAL:-10}"           # nombre d'items
N_SCEN="${N_SCEN:-500}"        # nombre de scénarios
N_INSTANCES="${N_INSTANCES:-100}"
MAX_DEPTH="${MAX_DEPTH:-10}"
EPSILON="${EPSILON:-0.1 0.2 0.3}"

# ── Chemins ────────────────────────────────────────────────────────────────────
COLLECT_BIN="$REPO/solver_bab/build/bab_mkp_gecode_base_collect"
LOCAL_OUTDIR="$SLURM_TMPDIR/train"
LOCAL_RESULT="${LOCAL_OUTDIR}_base"
# N=500 : compat. avec le répertoire existant ; sinon dossier dédié par N
if [[ "$N_SCEN" == "500" ]]; then
    DEST="$REPO/data/train_base_s30"
else
    DEST="$REPO/data/train_N${N_SCEN}"
fi

echo "======================================================================"
echo "  COLLECT BASE — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  n=$N_VAL  N_scen=$N_SCEN  n_instances=$N_INSTANCES  max_depth=$MAX_DEPTH"
echo "  DEST : $DEST"
echo "======================================================================"

# ── Vérification du binaire ───────────────────────────────────────────────────
if [[ ! -f "$COLLECT_BIN" ]]; then
    echo "ERREUR : binaire introuvable : $COLLECT_BIN"
    echo "Compiler en session interactive avec : bash compile_cc.sh"
    exit 1
fi
echo "[binaire] Trouvé : $COLLECT_BIN"

# ── Modules ───────────────────────────────────────────────────────────────────
module load python/3.11
module load scipy-stack

# ── Bibliothèques dynamiques Gecode (node-local pour éviter le throttling Lustre)
LOCAL_LIB="$SLURM_TMPDIR/lib"
mkdir -p "$LOCAL_LIB"
cp "$REPO/external/gecode/lib/"*.so* "$LOCAL_LIB/"
export LD_LIBRARY_PATH="$LOCAL_LIB:${LD_LIBRARY_PATH:-}"

# ── Sync incrémentale vers stockage permanent ─────────────────────────────────
mkdir -p "$DEST" "$LOCAL_RESULT"

if [[ -n "$(ls -A "$DEST" 2>/dev/null)" ]]; then
    echo "[init] Restauration depuis $DEST ..."
    rsync -a "$DEST/" "$LOCAL_RESULT/"
fi

do_sync() {
    [[ -d "$LOCAL_RESULT" ]] || return 0
    rsync -a --append "$LOCAL_RESULT/" "$DEST/" 2>/dev/null || true
    echo "  [sync] $(date '+%H:%M:%S') → $DEST/"
}

trap 'echo ""; echo "[trap] Sync finale avant sortie..."; do_sync' EXIT

(while true; do sleep 600; do_sync; done) &
SYNC_PID=$!

mkdir -p "$REPO/logs"

# ── Lancement ─────────────────────────────────────────────────────────────────
echo ""
echo "Démarrage collect/collect_large_base.py ..."
echo "  sortie temporaire : $LOCAL_RESULT/"
echo "  sync toutes les 10 min vers : $DEST/"
echo ""

python "$REPO/collect/collect_large_base.py" \
    --n            "$N_VAL"         \
    --m            5                \
    --N            "$N_SCEN"        \
    --n_instances  "$N_INSTANCES"   \
    --epsilon      $EPSILON         \
    --max_depth    "$MAX_DEPTH"     \
    --binary       "$COLLECT_BIN"   \
    --outdir       "$LOCAL_OUTDIR"

EXIT_CODE=$?

kill "$SYNC_PID" 2>/dev/null || true

echo ""
echo "======================================================================"
echo "  Terminé — $(date)   code retour : $EXIT_CODE"
echo "  Résultats : $DEST/"
echo "======================================================================"

exit $EXIT_CODE
