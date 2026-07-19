#!/bin/bash
# collect/run_calibrate_kp_base_samples_rorqual.sh — Calibrage du débit réel
# de collect_kp_base_samples.py (CC-KP, m=1, CP pur) avant de dimensionner
# --time du run d'ampleur (run_collect_kp_base_samples_rorqual.sh, à écrire
# une fois ce calibrage analysé).
#
# Teste les 2 combos les plus coûteux de la grille visée (n=20/N=100 et
# n=25/N=100 — n=25 est le "stretch" de la grille, cf. discussion), avec
# n_instances=200 (pas 10000) pour rester rapide. Le débit mesuré (s/instance)
# permet d'extrapoler le budget --time nécessaire pour n_instances=10000.
#
# Soumission :
#   sbatch collect/run_calibrate_kp_base_samples_rorqual.sh

#SBATCH --account=def-qcappart_cpu
#SBATCH --partition=cpubase_bycore_b4
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=calib_kp_s10
#SBATCH --output=logs/calib_kp_s10_%j.out
#SBATCH --error=logs/calib_kp_s10_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
COLLECT_BIN="$REPO/solver_bab/build/bab_mkp_gecode_base_collect"
SCRIPT="$REPO/collect/collect_kp_base_samples.py"

echo "======================================================================"
echo "  CALIBRAGE collect_kp_base_samples — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "======================================================================"

if [[ ! -f "$COLLECT_BIN" ]]; then
    echo "ERREUR : binaire introuvable : $COLLECT_BIN"
    echo "Compiler en session interactive avec : bash compile_cc.sh"
    exit 1
fi
echo "[binaire] Trouvé : $COLLECT_BIN"

module load python/3.11
module load scipy-stack

LOCAL_LIB="$SLURM_TMPDIR/lib"
mkdir -p "$LOCAL_LIB"
cp "$REPO/external/gecode/lib/"*.so* "$LOCAL_LIB/"
export LD_LIBRARY_PATH="$LOCAL_LIB:${LD_LIBRARY_PATH:-}"

LOCAL_OUTDIR="$SLURM_TMPDIR/calib_kp"
mkdir -p "$REPO/logs"

N_INSTANCES_CALIB=200

for CFG in "20 100" "25 100"; do
    read -r N_VAL N_SCEN <<< "$CFG"
    echo ""
    echo "=== n=$N_VAL  N=$N_SCEN  n_instances=$N_INSTANCES_CALIB  $(date '+%H:%M:%S') ==="
    T0=$(date +%s)
    python3 "$SCRIPT" \
        --n "$N_VAL" --N "$N_SCEN" \
        --epsilon 0.1 --max_depth 10 --sample_rate 0.10 \
        --n_instances "$N_INSTANCES_CALIB" \
        --binary "$COLLECT_BIN" --outdir "$LOCAL_OUTDIR"
    T1=$(date +%s)
    ELAPSED=$((T1 - T0))
    PER_INST=$(awk -v e="$ELAPSED" -v n="$N_INSTANCES_CALIB" 'BEGIN{printf "%.3f", e/n}')
    EXTRAP_10K_H=$(awk -v p="$PER_INST" 'BEGIN{printf "%.2f", p*10000/3600}')
    echo "--- n=$N_VAL N=$N_SCEN : ${ELAPSED}s pour ${N_INSTANCES_CALIB} instances" \
         "(${PER_INST}s/instance, extrapolation 10000 instances : ${EXTRAP_10K_H}h) ---"
done

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
