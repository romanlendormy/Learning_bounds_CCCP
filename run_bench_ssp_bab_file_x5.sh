#!/bin/bash
# run_bench_ssp_bab_file_x5.sh — bench_ssp_bab_file sur 5 instances SSP (E=3, T=8,
# N=20, seed=42, ep0..ep4, cf. solver_bab/ssp/instances/E3_T8_N20_seed42_x5/),
# Gecode base vs B&B base_ref vs B&B v1CLD vs B&B v1LD, T_node=50, sans limite
# de temps par méthode (bornée par le --time SLURM ci-dessous).
#
# Soumettre :
#   sbatch run_bench_ssp_bab_file_x5.sh
#
# Toutes les lignes CSV_ROW des 5 instances sont dans le même .out — moyenne
# calculée après coup (cf. session) en parsant ces lignes.

# ── Ressources SLURM ──────────────────────────────────────────────────────────
#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=bench_ssp_bab_file_x5
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

T_NODE="${T_NODE:-50}"
TIME_LIM="${TIME_LIM:-0}"       # 0 = illimité (Gecode base / base_ref / v1CLD)
TIME_LIM_LD="${TIME_LIM_LD:-0}" # 0 = illimité (v1LD) ; < 0 désactiverait v1LD

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
BINARY="$REPO/solver_bab/build/bench_ssp_bab_file"
INSTANCES_ROOT="$REPO/solver_bab/ssp/instances/E3_T8_N20_seed42_x5"

module load StdEnv/2023 gcc/12.3.0 2>/dev/null || true
export LD_LIBRARY_PATH="$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

mkdir -p "$REPO/logs"

echo "======================================================================"
echo "  bench_ssp_bab_file_x5 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  T_node=$T_NODE  time_lim=${TIME_LIM}s  time_lim_v1LD=${TIME_LIM_LD}s"
echo "======================================================================"

if [ ! -x "$BINARY" ]; then
    echo "ERREUR : binaire introuvable : $BINARY"
    exit 1
fi

for ep in ep0 ep1 ep2 ep3 ep4; do
    dir="$INSTANCES_ROOT/$ep"
    if [ ! -f "$dir/instance.txt" ]; then
        echo "ERREUR : instance introuvable : $dir/instance.txt"
        exit 1
    fi
    echo ""
    echo "──────────────────────────────────────────────────────────────────────"
    echo "  Instance : $ep ($dir)"
    echo "──────────────────────────────────────────────────────────────────────"
    # ne pas laisser set -e interrompre le batch : bench_ssp_bab_file retourne 1
    # quand le sanity check échoue (résultat informatif attendu, pas une erreur
    # de script) — on veut quand même enchaîner sur les instances suivantes.
    "$BINARY" "$dir" "$T_NODE" "$TIME_LIM" "$TIME_LIM_LD" || \
        echo "  (bench_ssp_bab_file a retourné un code d'erreur pour $ep — cf. sanity check ci-dessus)"
done

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
