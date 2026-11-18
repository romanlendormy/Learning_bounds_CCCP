#!/bin/bash
# collect/run_collect_mkp.sh — Collecte des nœuds d'entraînement du GNN MKP
# (collect_large_base_samples.py, 10 % des nœuds gardés) pour la grille
# n∈{10,20} × N∈{20,50,100}, m=5, epsilon=0.1 : 6 tâches d'array indépendantes,
# 10 000 instances par cellule. Sortie : data/train_base_s10/.
#
# Reprise après dépassement de --time : resoumettre le même script. Chaque cellule
# a son propre seeds_done_n{n}_m{m}_N{N}.txt dans $DEST, restauré au démarrage ;
# --n_instances est une cible TOTALE, donc une cellule complète ne refait rien.
#
# Prérequis : build CPU (cmake -S . -B build -DGNN_DEVICE=cpu && cmake --build build).
# Soumission (adapter --account / --partition au cluster) :
#   sbatch collect/run_collect_mkp.sh

#SBATCH --account=def-qcappart
#SBATCH --job-name=collect_s10
#SBATCH --output=logs/collect_s10_%A_%a.out
#SBATCH --error=logs/collect_s10_%A_%a.err
#SBATCH --time=13:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-5

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
COLLECT_BIN="$REPO/build/solver_bab/bab_mkp_gecode_base_collect"
SCRIPT="$REPO/collect/collect_large_base_samples.py"

# ── (n, N) selon l'indice du job array ────────────────────────────────────────
N_VALS=(10 10 10 20 20 20)
N_SCENS=(20 50 100 20 50 100)
N_VAL="${N_VALS[$SLURM_ARRAY_TASK_ID]}"
N_SCEN="${N_SCENS[$SLURM_ARRAY_TASK_ID]}"

# Stockage local rapide (NVMe SLURM, propre à cette tâche)
LOCAL_OUTDIR="$SLURM_TMPDIR/train"
LOCAL_STAGING="${LOCAL_OUTDIR}_base"
LOCAL_S10="${LOCAL_OUTDIR}_base_s10"
DEST="$REPO/data/train_base_s10"

echo "======================================================================"
echo "  COLLECT S10 (sampling 10 %, epsilon=0.1) — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Array task : ${SLURM_ARRAY_TASK_ID:-0}"
echo "  Nœud : $(hostname)   n=$N_VAL   N=$N_SCEN"
echo "======================================================================"

# ── Vérification du binaire ───────────────────────────────────────────────────
if [[ ! -f "$COLLECT_BIN" ]]; then
    echo "ERREUR : binaire introuvable : $COLLECT_BIN"
    echo "Compiler d'abord : cmake -S . -B build -DGNN_DEVICE=cpu && cmake --build build"
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

# ── Restauration des données existantes (reprise sur redémarrage) ────────────
mkdir -p "$DEST" "$LOCAL_STAGING" "$LOCAL_S10"
if [[ -n "$(ls -A "$DEST" 2>/dev/null)" ]]; then
    echo "[init] Restauration depuis $DEST ..."
    rsync -a "$DEST/" "$LOCAL_S10/"
fi

# ── Sync périodique + finale vers stockage permanent ─────────────────────────
do_sync() {
    [[ -d "$LOCAL_S10" ]] || return 0
    rsync -a --append "$LOCAL_S10/" "$DEST/" 2>/dev/null || true
    echo "  [sync] $(date '+%H:%M:%S') → $DEST/"
}
trap 'echo ""; echo "[trap] Sync finale avant sortie..."; do_sync' EXIT
(while true; do sleep 600; do_sync; done) &
SYNC_PID=$!

mkdir -p "$REPO/logs"

# ── Lancement ─────────────────────────────────────────────────────────────────
echo ""
echo "Démarrage collect_large_base_samples.py  n=$N_VAL  N=$N_SCEN ..."
python "$SCRIPT" \
    --n "$N_VAL" --m 5 --N "$N_SCEN" \
    --epsilon 0.1 --max_depth 10 --sample_rate 0.10 \
    --n_instances 10000 \
    --binary "$COLLECT_BIN" --outdir "$LOCAL_OUTDIR"

kill "$SYNC_PID" 2>/dev/null || true

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "  Résultats : $DEST/"
echo "======================================================================"
