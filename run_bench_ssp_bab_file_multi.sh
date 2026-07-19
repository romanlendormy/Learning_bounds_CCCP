#!/bin/bash
# run_bench_ssp_bab_file_multi.sh — B&B base_ref vs B&B v1CLD vs B&B v1LD
# sur plusieurs instances SSP (comparaison E/T/N), enchaînées dans un seul job.
#
# Gecode base désactivé par défaut (RUN_GECODE_BASE=0) — coûteux/redondant sur
# ce lot, B&B base_ref sert de référence pour le sanity check et le gap
# racine (cf. bench_ssp_bab_file.cpp, option run_gecode_base). v1LD activé par
# défaut (TIME_LIM_LD=3000s) — cf. run_bench_ssp_bab_file.sh pour la
# comparaison à 1 seule instance.
#
# Génère d'abord (si absentes) les instances manquantes via
# generate_episode_fixed_E (ssp_stochastic.generator_adaptive) + export_to_txt
# — calibration par bissection sur Gecode base/ssp_probe (cf. calibrate_adaptive.py),
# PAS de garantie de convergence pour des (E,T) jamais calibrés auparavant
# (cf. commentaire calibrate_adaptive.py : T=16 ne calibre dans aucune fenêtre,
# T=15 n'a jamais été testé).
#
# Soumettre :
#   sbatch run_bench_ssp_bab_file_multi.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=bench_ssp_multi
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

T_NODE="${T_NODE:-50}"
TIME_LIM="${TIME_LIM:-0}"           # 0 = illimité par méthode (borné par --time SLURM ci-dessus)
TIME_LIM_LD="${TIME_LIM_LD:-3000}"  # v1LD activé, budget 3000s/instance
RUN_GECODE_BASE="${RUN_GECODE_BASE:-0}"  # 0 = désactivé (cf. en-tête)

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
BINARY="$REPO/solver_bab/build/bench_ssp_bab_file"
VENV_PATH="$HOME/envs/learning_bounds"

module load StdEnv/2023 gcc/12.3.0 python/3.11 2>/dev/null || true
export LD_LIBRARY_PATH="$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

mkdir -p "$REPO/logs"

INSTANCES=(
    "solver_bab/ssp/instances/E3_T8_N20_seed42"
    "solver_bab/ssp/instances/E3_T8_N50_seed42"
    "solver_bab/ssp/instances/E3_T15_N50_seed42"
    "ssp_stochastic/debug_instances/E4_T8_N20_rep2_v1cld_invalid_bound"
    "solver_bab/ssp/instances/E4_T8_N50_seed42"
    "solver_bab/ssp/instances/E4_T15_N50_seed42"
)

# ── Recompilation incrémentale (option run_gecode_base ajoutée au binaire) ────
echo "=== Compilation bench_ssp_bab_file ($(date)) ==="
cmake --build "$REPO/solver_bab/build" --target bench_ssp_bab_file ssp_probe -j"${SLURM_CPUS_PER_TASK:-4}"

if [ ! -x "$BINARY" ]; then
    echo "ERREUR : binaire introuvable : $BINARY"
    exit 1
fi

# ── Génération des instances manquantes ───────────────────────────────────────
echo ""
echo "=== Génération des instances manquantes ($(date)) ==="
source "$VENV_PATH/bin/activate"
python "$REPO/gen_missing_ssp_instances.py"
deactivate

for inst in "${INSTANCES[@]}"; do
    if [ ! -f "$REPO/$inst/instance.txt" ]; then
        echo "======================================================================"
        echo "  Instance : $inst  —  ABSENTE (calibration probablement échouée) — SKIP"
        echo "======================================================================"
        echo ""
        continue
    fi
    echo "======================================================================"
    echo "  Instance : $inst  —  $(date)"
    echo "  T_node=$T_NODE  time_lim=${TIME_LIM}s  time_lim_v1LD=${TIME_LIM_LD}s  run_gecode_base=${RUN_GECODE_BASE}"
    echo "======================================================================"
    # || true : un sanity check [FAIL] fait sortir le binaire en code 1 — ne doit
    # pas interrompre les instances suivantes (set -e ci-dessus).
    "$BINARY" "$REPO/$inst" "$T_NODE" "$TIME_LIM" "$TIME_LIM_LD" "$RUN_GECODE_BASE" || true
    echo ""
done

echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
