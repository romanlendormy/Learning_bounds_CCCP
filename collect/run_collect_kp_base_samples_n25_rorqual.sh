#!/bin/bash
# collect/run_collect_kp_base_samples_n25_rorqual.sh — Collecte CC-KP (m=1,
# CP pur) samplée à 10 % pour la grille STRETCH n=25 × N∈{20,50,100},
# epsilon=0.1 uniquement. 3 combinaisons = 3 tâches d'un job array.
#
# Analogue de run_collect_base_n25_30_rorqual.sh (MKP), mais SANS n=30 —
# n=30 a été explicitement exclu (même à m=1, REF timeout déjà à 120s dans
# le sweep_kp_gecode_grid.py, jusqu'à 13.5M nœuds — hors de portée).
#
# ⚠️ n=25 est un "stretch" à rendement partiel attendu, PAS une collecte
# complète : calibrage du 13 juillet 2026 mesure ~15-25s/instance à N=100
# (cohérent avec le sweep : REF met 12.5-24.4s pour résoudre à l'optimum).
# Avec --time=24:00:00, on attend ~4000-5500 instances/combo (pas 10000) —
# cohérent avec ce que MKP (m=5) a obtenu à n=25 (4333/1625/657 sur 3 combos).
# Reprise possible en resoumettant si la cible n'est pas atteinte.
#
# Reprise le 15 juillet 2026 : job initial 16001757 interrompu par un incident
# cluster (pas un bug applicatif) après 4228/3877/4149 seeds tentées (N20/N50/
# N100). --n_instances compte les seeds TENTÉES, pas les instances faisables :
# rendement observé 57.6 %/31.5 %/15.2 % (N20/N50/N100) → viser 10000
# instances FAISABLES demanderait ~17400/31800/65700 seeds, soit ~9.6h/22h/46h
# de calcul SUPPLÉMENTAIRE au rythme observé. Décision utilisateur : garder
# --time=24:00:00 (pas d'extension) et laisser tourner avec une cible haute
# (10000 faisables visées) — N20 devrait l'atteindre, N50/N100 seront coupés
# par la limite de temps avant la cible mais avec un rendement bien meilleur
# qu'avant (plafond dur = 24h, pas un nombre de seeds). Resoumettre après
# coup si N50/N100 n'ont pas assez progressé.
#
# Deuxième incident (job 16204940, même jour) : même cause que côté n=10/20
# (solver_bab/build relié à libtorch-cuda depuis le 14 juillet, cassé sur
# nœuds CPU) — 100 % des tentatives ont échoué (libcupti.so.12 introuvable),
# 0 instance ajoutée, mais seeds_done pollué (cibles "atteintes" à tort).
# Fix : binaire CPU-only dédié (solver_bab/build_collect_cpu/, cf.
# collect/compile_collect_cpu.sh), testé fonctionnel. Cibles repoussées
# (plutôt que de tronquer seeds_done) pour compenser les seeds brûlées :
#   N20→30572  N50→59723  N100→127251
# (N50/N100 seront de toute façon coupés par --time=24:00:00 bien avant ces
# cibles — c'est le plafond dur convenu, pas un problème.)
#
# Soumission :
#   sbatch collect/run_collect_kp_base_samples_n25_rorqual.sh

# ── Ressources SLURM ──────────────────────────────────────────────────────────
#SBATCH --account=def-qcappart_cpu
#SBATCH --partition=cpubase_bycore_b4
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=collect_kp_s10_n25
#SBATCH --output=logs/collect_kp_s10_n25_%A_%a.out
#SBATCH --error=logs/collect_kp_s10_n25_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-2

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
COLLECT_BIN="$REPO/solver_bab/build_collect_cpu/bab_mkp_gecode_base_collect"
SCRIPT="$REPO/collect/collect_kp_base_samples.py"

# ── N selon l'indice du job array (n=25 fixe) ─────────────────────────────────
N_VAL=25
N_SCENS=(20 50 100)
TARGETS=(30572 59723 127251)
N_SCEN="${N_SCENS[$SLURM_ARRAY_TASK_ID]}"
TARGET="${TARGETS[$SLURM_ARRAY_TASK_ID]}"

LOCAL_OUTDIR="$SLURM_TMPDIR/train_kp"
LOCAL_STAGING="${LOCAL_OUTDIR}_base"
LOCAL_S10="${LOCAL_OUTDIR}_base_s10"
DEST="$REPO/data/train_kp_base_s10"

echo "======================================================================"
echo "  COLLECT KP S10 n=25 STRETCH (sampling 10 %, epsilon=0.1) — $(date)"
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
