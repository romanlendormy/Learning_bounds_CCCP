#!/bin/bash
# collect/run_collect_kp_base_samples_rorqual.sh — Collecte CC-KP (m=1, CP pur)
# samplée à 10 % (collect_kp_base_samples.py) pour la grille n∈{10,20} ×
# N∈{20,50,100}, epsilon=0.1 uniquement. 6 combinaisons = 6 tâches d'un job
# array, chacune indépendante ($SLURM_TMPDIR propre par tâche).
#
# Analogue direct de run_collect_base_samples_rorqual.sh (MKP, m=5). Calibrage
# effectué le 13 juillet 2026 : n=20/N=100 mesuré à ~1-1.5s/instance sur 3
# reproductions indépendantes (un job de calibrage a montré ~18s/instance,
# mais non reproduit ensuite — traité comme anomalie, cf. session). Avec cette
# vitesse, 10000 instances/combo ≈ 3-4h — le budget 13h ci-dessous garde une
# marge large en cas de nouvelle anomalie.
#
# Reprise le 15 juillet 2026 : job initial 16001756 interrompu par un incident
# cluster (pas un bug applicatif, cf. session) — n=10 avait atteint sa cible
# de 10000 SEEDS (mais --n_instances compte les seeds tentées, pas les
# instances faisables : ~69-76 % de rendement → seulement 6912-7600
# instances faisables collectées). Cibles recalculées pour viser ~10000
# instances FAISABLES par combo (seeds nécessaires = 10000/rendement observé,
# avec marge) : n10 N20→14500(69%) N50→14200(70.5%) N100→13200(76%) ;
# n20 N20→11800(85%) N50→11900(84%) N100→12200(82.5%).
#
# Deuxième incident (job 16204939, même jour) : solver_bab/build a été
# reconfiguré en CUDA le 14 juillet (validation native_aoti), ce qui a
# relinké bab_mkp_gecode_base_collect contre libtorch-cuda — cassé sur les
# nœuds CPU (cpubase_bycore_b4, pas de CUDA). Résultat : 100% des tentatives
# de ce job ont échoué (libcupti.so.12 introuvable) MAIS le script marque
# quand même chaque seed "faite" même en cas d'échec binaire → 0 instance
# ajoutée, seeds_done pollué (cible déjà "atteinte" sans rien avoir collecté).
# Fix : binaire recompilé dans un arbre CPU-only dédié
# (solver_bab/build_collect_cpu/, via collect/compile_collect_cpu.sh, ne
# touche pas solver_bab/build qui reste en CUDA pour la prod GPU), testé
# fonctionnel (résout une vraie instance). Plutôt que de tronquer
# seeds_done (déconseillé/bloqué — action destructive sur des données
# existantes), on pousse les cibles plus loin pour compenser les seeds
# brûlées (arbitraires, les sauter ne coûte rien) :
#   n10 N20→19000 N50→18400 N100→16400 ; n20 N20→18040 N50→19343 N100→20605
# Reprise automatique via seeds_done : aucun risque de doublon.
#
# Reprise sur dépassement de --time : resoumettre le même script.
# --n_instances est une cible TOTALE (pas "combien en ajouter").
#
# Soumission :
#   sbatch collect/run_collect_kp_base_samples_rorqual.sh

# ── Ressources SLURM ──────────────────────────────────────────────────────────
#SBATCH --account=def-qcappart_cpu
#SBATCH --partition=cpubase_bycore_b4
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=collect_kp_s10
#SBATCH --output=logs/collect_kp_s10_%A_%a.out
#SBATCH --error=logs/collect_kp_s10_%A_%a.err
#SBATCH --time=13:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-5

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
COLLECT_BIN="$REPO/solver_bab/build_collect_cpu/bab_mkp_gecode_base_collect"
SCRIPT="$REPO/collect/collect_kp_base_samples.py"

# ── (n, N) selon l'indice du job array ────────────────────────────────────────
N_VALS=(10 10 10 20 20 20)
N_SCENS=(20 50 100 20 50 100)
TARGETS=(19000 18400 16400 18040 19343 20605)
N_VAL="${N_VALS[$SLURM_ARRAY_TASK_ID]}"
N_SCEN="${N_SCENS[$SLURM_ARRAY_TASK_ID]}"
TARGET="${TARGETS[$SLURM_ARRAY_TASK_ID]}"

# Stockage local rapide (NVMe SLURM, propre à cette tâche)
LOCAL_OUTDIR="$SLURM_TMPDIR/train_kp"
LOCAL_STAGING="${LOCAL_OUTDIR}_base"
LOCAL_S10="${LOCAL_OUTDIR}_base_s10"
DEST="$REPO/data/train_kp_base_s10"

echo "======================================================================"
echo "  COLLECT KP S10 (m=1, sampling 10 %, epsilon=0.1) — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Array task : ${SLURM_ARRAY_TASK_ID:-0}"
echo "  Nœud : $(hostname)   n=$N_VAL   N=$N_SCEN"
echo "======================================================================"

if [[ ! -f "$COLLECT_BIN" ]]; then
    echo "ERREUR : binaire introuvable : $COLLECT_BIN"
    echo "Compiler avec : sbatch collect/compile_collect_cpu.sh"
    exit 1
fi
echo "[binaire] Trouvé : $COLLECT_BIN"

module load python/3.11
module load scipy-stack

LOCAL_LIB="$SLURM_TMPDIR/lib"
mkdir -p "$LOCAL_LIB"
cp "$REPO/external/gecode/lib/"*.so* "$LOCAL_LIB/"
export LD_LIBRARY_PATH="$LOCAL_LIB:${LD_LIBRARY_PATH:-}"

mkdir -p "$DEST" "$LOCAL_STAGING" "$LOCAL_S10"
if [[ -n "$(ls -A "$DEST" 2>/dev/null)" ]]; then
    echo "[init] Restauration depuis $DEST ..."
    rsync -a "$DEST/" "$LOCAL_S10/"
fi

do_sync() {
    [[ -d "$LOCAL_S10" ]] || return 0
    rsync -a --append "$LOCAL_S10/" "$DEST/" 2>/dev/null || true
    echo "  [sync] $(date '+%H:%M:%S') → $DEST/"
}
trap 'echo ""; echo "[trap] Sync finale avant sortie..."; do_sync' EXIT
(while true; do sleep 600; do_sync; done) &
SYNC_PID=$!

mkdir -p "$REPO/logs"

echo ""
echo "Démarrage collect_kp_base_samples.py  n=$N_VAL  N=$N_SCEN  cible_seeds=$TARGET ..."
python3 "$SCRIPT" \
    --n "$N_VAL" --N "$N_SCEN" \
    --epsilon 0.1 --max_depth 10 --sample_rate 0.10 \
    --n_instances "$TARGET" \
    --binary "$COLLECT_BIN" --outdir "$LOCAL_OUTDIR"

kill "$SYNC_PID" 2>/dev/null || true

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "  Résultats : $DEST/"
echo "======================================================================"
