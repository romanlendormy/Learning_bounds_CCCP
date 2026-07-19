#!/bin/bash
# collect/run_collect_base_n1020_N200_500_rorqual.sh — Collecte CC-MKP samplée
# à 10 % (collect_large_base_samples.py) pour la grille n∈{10,20} × N∈{200,500},
# epsilon=0.1 uniquement. 4 combinaisons = 4 tâches d'un job array, chacune
# indépendante (chaque tâche a son propre $SLURM_TMPDIR, donc son propre
# staging <outdir>_base/{instances,nodes}_n{n}.txt — pas de risque de
# collision même pour deux tâches partageant le même n).
#
# Variante de run_collect_base_samples_rorqual.sh (N∈{20,50,100}) : mêmes m,
# sample_rate, cible --n_instances (10000 = maximum, best-effort en 24h) et
# DEST partagé data/train_base_s10 (fichiers déjà distingués par n et N dans
# leur nom, donc pas de collision avec les données existantes).
#
# --time=24:00:00 : N plus grand (200/500) → sous-problèmes ζ plus coûteux
# par nœud B&B ; on laisse tourner 24h et on garde ce qui a été collecté
# (reprise possible en resoumettant si la cible de 10000 n'est pas atteinte).
# Estimation grossière (extrapolation du débit mesuré à N=20/50/100, ~N^0.8) :
# n=10 tient probablement dans 24h pour N=200 et N=500 ; n=20 nécessitera
# vraisemblablement plusieurs relances (comme n=25/30), surtout à N=500.
#
# Soumission :
#   sbatch collect/run_collect_base_n1020_N200_500_rorqual.sh

# ── Ressources SLURM ──────────────────────────────────────────────────────────
#SBATCH --account=def-qcappart_cpu
#SBATCH --partition=cpubase_bycore_b4
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=collect_s10_N200500
#SBATCH --output=logs/collect_s10_N200500_%A_%a.out
#SBATCH --error=logs/collect_s10_N200500_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-3

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
COLLECT_BIN="$REPO/solver_bab/build_collect_cpu/bab_mkp_gecode_base_collect"
SCRIPT="$REPO/collect/collect_large_base_samples.py"

# ── (n, N) selon l'indice du job array ────────────────────────────────────────
N_VALS=(10 10 20 20)
N_SCENS=(200 500 200 500)
N_VAL="${N_VALS[$SLURM_ARRAY_TASK_ID]}"
N_SCEN="${N_SCENS[$SLURM_ARRAY_TASK_ID]}"

# Stockage local rapide (NVMe SLURM, propre à cette tâche)
LOCAL_OUTDIR="$SLURM_TMPDIR/train"
LOCAL_STAGING="${LOCAL_OUTDIR}_base"
LOCAL_S10="${LOCAL_OUTDIR}_base_s10"
DEST="$REPO/data/train_base_s10"

echo "======================================================================"
echo "  COLLECT S10 n=10/20 × N=200/500 (sampling 10 %, epsilon=0.1) — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Array task : ${SLURM_ARRAY_TASK_ID:-0}"
echo "  Nœud : $(hostname)   n=$N_VAL   N=$N_SCEN"
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
