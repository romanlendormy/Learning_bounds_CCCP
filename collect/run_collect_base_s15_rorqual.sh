#!/bin/bash
# collect/run_collect_base_s15.sh — Collecte CC-MKP avec sampling 15 %.
#
# Configurations :
#   n=15  N=100   seeds   0..299
#   n=15  N=250   seeds 1000..1299
#   n=20  N=50    seeds   0..299
#   n=20  N=100   seeds 1000..1299  (time_limit=30s pour rester raisonnable)
#
# Résultats dans : data/train_base_s30/  (même répertoire que les données existantes)
# Les fichiers sont nommés nodes_n{n}_m5.txt — mélange des N possible car le
# loader résout N depuis instances_n{n}_m5.txt via inst_id.
#
# NOTE seeds : seeds_done_n{n}_m5.txt ne contient pas N → on sépare les
# plages de seeds par valeur de N pour éviter les conflits.
#
# Soumission :
#   sbatch collect/run_collect_base_s15.sh

# ── Ressources SLURM ──────────────────────────────────────────────────────────
#SBATCH --account=def-qcappart_cpu
#SBATCH --partition=cpubase_bycore_b4
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=collect_s15
#SBATCH --output=logs/collect_s15_%j.out
#SBATCH --error=logs/collect_s15_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
COLLECT_BIN="$REPO/solver_bab/build/bab_mkp_gecode_base_collect"
SCRIPT="$REPO/collect/collect_large_base_s30.py"

# Stockage local rapide (NVMe SLURM)
LOCAL_OUTDIR="$SLURM_TMPDIR/train"
LOCAL_STAGING="${LOCAL_OUTDIR}_base"
LOCAL_S30="${LOCAL_OUTDIR}_base_s30"
DEST="$REPO/data/train_base_s30"

echo "======================================================================"
echo "  COLLECT S15 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "======================================================================"

# ── Modules & bibliothèques ───────────────────────────────────────────────────
module load python/3.11
module load scipy-stack

LOCAL_LIB="$SLURM_TMPDIR/lib"
mkdir -p "$LOCAL_LIB"
cp "$REPO/external/gecode/lib/"*.so* "$LOCAL_LIB/"
export LD_LIBRARY_PATH="$LOCAL_LIB:${LD_LIBRARY_PATH:-}"

# ── Restauration des données existantes ──────────────────────────────────────
mkdir -p "$DEST" "$LOCAL_STAGING" "$LOCAL_S30"
if [[ -n "$(ls -A "$DEST" 2>/dev/null)" ]]; then
    echo "[init] Restauration depuis $DEST ..."
    rsync -a "$DEST/" "$LOCAL_S30/"
fi

# ── Sync périodique vers stockage permanent ───────────────────────────────────
do_sync() {
    [[ -d "$LOCAL_S30" ]] || return 0
    rsync -a --append "$LOCAL_S30/" "$DEST/" 2>/dev/null || true
    echo "  [sync] $(date '+%H:%M:%S') → $DEST/"
}
trap 'echo "[trap] Sync finale..."; do_sync' EXIT
(while true; do sleep 600; do_sync; done) &
SYNC_PID=$!

mkdir -p "$REPO/logs"

# ── Arguments communs ─────────────────────────────────────────────────────────
COMMON="--m 5 --epsilon 0.1 0.2 0.3 --max_depth 10 --sample_rate 0.15
        --n_instances 300 --binary $COLLECT_BIN --outdir $LOCAL_OUTDIR"

# ── 1/4  n=15  N=100  seeds 0..299 ───────────────────────────────────────────
echo ""
echo "=== [1/4]  n=15  N=100  seeds 0..299  $(date '+%H:%M:%S') ==="
python "$SCRIPT" $COMMON --n 15 --N 100 --seed_start 0

# ── 2/4  n=15  N=250  seeds 1000..1299 ───────────────────────────────────────
echo ""
echo "=== [2/4]  n=15  N=250  seeds 1000..1299  $(date '+%H:%M:%S') ==="
python "$SCRIPT" $COMMON --n 15 --N 250 --seed_start 1000

# ── 3/4  n=20  N=50   seeds 0..299 ───────────────────────────────────────────
echo ""
echo "=== [3/4]  n=20  N=50   seeds 0..299  $(date '+%H:%M:%S') ==="
python "$SCRIPT" $COMMON --n 20 --N 50 --seed_start 0

# ── 4/4  n=20  N=100  seeds 1000..1299  (time_limit=30s par B&B) ─────────────
echo ""
echo "=== [4/4]  n=20  N=100  seeds 1000..1299  $(date '+%H:%M:%S') ==="
python "$SCRIPT" $COMMON --n 20 --N 100 --seed_start 1000 --time_limit 30

kill "$SYNC_PID" 2>/dev/null || true

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "  Résultats : $DEST/"
echo "======================================================================"
