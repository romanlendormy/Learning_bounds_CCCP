#!/bin/bash
# collect/run_collect_ssp_grid_rorqual.sh — Collecte CC-SSP samplée à 10 %
# (collect_large_base_samples_ssp.py) sur la grille à 9 combos (E,T,N)
# retenue en session (6 tailles validées E∈{3,4}×T∈{8,12}×N∈{20,50} + 3
# tailles "un cran plus grand" E=6×T∈{8,12}×N∈{20,50} confirmées par un
# sondage rapide — cf. discussion : E≥8 laissé de côté, calibration instable
# à cette échelle, deux instances sondées se sont révélées INFEASIBLE).
#
# Calqué sur collect/run_collect_base_samples_rorqual.sh (MKP) : job array,
# une tâche par combo, chacune sur son propre $SLURM_TMPDIR (staging local,
# pas de risque de collision), rsync périodique + final vers data/.
#
# --n_instances 10000 par tâche (cible, pas garantie) ; --time=24:00:00 par
# tâche : les combos bon marché (E≤4) finiront largement avant, les combos
# E=6 pourraient ne pas atteindre 10k dans la journée (surtout E=6,T=8,N=20,
# le plus lent observé : ~7-8s/instance) — reprise possible en resoumettant
# le même script (seeds_done_E{E}_T{T}_N{N}.txt par combo, cible TOTALE).
#
# --time_limit 120 (par instance, coté B&B de collecte) : garde-fou pour
# qu'un seed anormalement dur n'accapare pas tout le budget de la tâche —
# absent côté MKP mais utile ici vu la variance observée (jusqu'à ~3x d'un
# rep à l'autre sur certains combos).
#
# Soumission :
#   sbatch collect/run_collect_ssp_grid_rorqual.sh

# ── Ressources SLURM ──────────────────────────────────────────────────────────
#SBATCH --account=def-qcappart_cpu
#SBATCH --partition=cpubase_bycore_b4
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=collect_ssp
#SBATCH --output=logs/collect_ssp_%A_%a.out
#SBATCH --error=logs/collect_ssp_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-8

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
COLLECT_BIN="$REPO/solver_bab/build/bab_ssp_gecode_base_collect"
PROBE_BIN="$REPO/solver_bab/build/ssp_probe"
SCRIPT="$REPO/collect/collect_large_base_samples_ssp.py"

# ── (E, T, N) selon l'indice du job array ─────────────────────────────────────
E_VALS=(3  3  4  4  4  4  6  6  6)
T_VALS=(12 12 8  8  12 12 8  8  12)
N_VALS=(20 50 20 50 20 50 20 50 50)
E_VAL="${E_VALS[$SLURM_ARRAY_TASK_ID]}"
T_VAL="${T_VALS[$SLURM_ARRAY_TASK_ID]}"
N_VAL="${N_VALS[$SLURM_ARRAY_TASK_ID]}"

# Stockage local rapide (NVMe SLURM, propre à cette tâche)
LOCAL_OUTDIR="$SLURM_TMPDIR/train_ssp"
LOCAL_STAGING="${LOCAL_OUTDIR}_base"
LOCAL_S10="${LOCAL_OUTDIR}_base_s10"
DEST="$REPO/data/train_ssp_base_s10"

echo "======================================================================"
echo "  COLLECT SSP S10 (sampling 10 %, epsilon=0.1) — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Array task : ${SLURM_ARRAY_TASK_ID:-0}"
echo "  Nœud : $(hostname)   E=$E_VAL   T=$T_VAL   N=$N_VAL"
echo "======================================================================"

# ── Vérification des binaires ──────────────────────────────────────────────────
for BIN in "$COLLECT_BIN" "$PROBE_BIN"; do
    if [[ ! -f "$BIN" ]]; then
        echo "ERREUR : binaire introuvable : $BIN"
        echo "Compiler avec : cmake --build solver_bab/build --target bab_ssp_gecode_base_collect ssp_probe"
        exit 1
    fi
done
echo "[binaires] OK : $COLLECT_BIN , $PROBE_BIN"

# ── Environnement Python (venv dédié — ssp_stochastic dépend de h5py/pyyaml,
# pas garantis par le seul module scipy-stack, cf. run_collect_ssp.sh) ───────
module load StdEnv/2023 gcc/12.3.0 python/3.11.5 2>/dev/null || true
VENV_PATH="$HOME/envs/learning_bounds"
if [[ -f "$VENV_PATH/bin/activate" ]]; then
    source "$VENV_PATH/bin/activate"
else
    echo "ERREUR : venv introuvable à $VENV_PATH" >&2
    exit 1
fi

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
echo "Démarrage collect_large_base_samples_ssp.py  E=$E_VAL  T=$T_VAL  N=$N_VAL ..."
python "$SCRIPT" \
    --E "$E_VAL" --T "$T_VAL" --N "$N_VAL" \
    --epsilon 0.1 --max_depth 10 --sample_rate 0.10 \
    --n_instances 10000 --time_limit 120 \
    --binary "$COLLECT_BIN" --probe_bin "$PROBE_BIN" \
    --outdir "$LOCAL_OUTDIR"

kill "$SYNC_PID" 2>/dev/null || true

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "  Résultats : $DEST/"
echo "======================================================================"
