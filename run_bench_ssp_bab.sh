#!/bin/bash
# run_bench_ssp_bab.sh — Gecode base vs B&B v1CLD vs B&B v1LD sur SSP (solver_bab/ssp).
#
# Toutes les méthodes tournent sur Gecode::BAB<> natif (cf. gecode_bab_ssp.hpp) —
# pas de DFS maison. Résultats imprimés au fur et à mesure dans le .out, pas
# seulement à la fin du job (utile pour suivre un run long, notamment v1LD dont
# ζ exact via CP-SAT à chaque nœud peut être très coûteux).
#
# Soumettre :
#   sbatch run_bench_ssp_bab.sh
#
# Varier l'instance / le budget :
#   sbatch --export=ALL,E=4,T=8,N=50               run_bench_ssp_bab.sh
#   sbatch --export=ALL,E=4,T=8,N=50,TIME_LIM=600   run_bench_ssp_bab.sh
#   sbatch --export=ALL,T_NODE=10                   run_bench_ssp_bab.sh

# ── Ressources SLURM ──────────────────────────────────────────────────────────
#SBATCH --account=def-qcappart
#SBATCH --chdir=/home/romanlen/projects/def-qcappart/romanl/Learning_bounds_CCCP
#SBATCH --job-name=bench_ssp_bab
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

# ── Paramètres ────────────────────────────────────────────────────────────────
E="${E:-4}"
T="${T:-8}"
N="${N:-50}"
EPS="${EPS:-0.10}"
T_NODE="${T_NODE:-30}"
TIME_LIM="${TIME_LIM:-0}"    # secondes, par méthode ; 0 = illimité (borné par le --time SLURM ci-dessus)

# ── Chemins ───────────────────────────────────────────────────────────────────
REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
SOLVER_BAB_DIR="$REPO/solver_bab"
BINARY="$SOLVER_BAB_DIR/build/bench_ssp_bab"

# ── Modules ───────────────────────────────────────────────────────────────────
module load StdEnv/2023 gcc/12.3.0 2>/dev/null || true

export LD_LIBRARY_PATH="$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

mkdir -p "$REPO/logs"

echo "======================================================================"
echo "  bench_ssp_bab — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  Instance : E=$E  T=$T  N=$N  eps=$EPS  T_node=$T_NODE  time_lim=${TIME_LIM}s/méthode"
echo "======================================================================"

# ── Vérification du binaire ───────────────────────────────────────────────────
if [ ! -x "$BINARY" ]; then
    echo "ERREUR : binaire introuvable : $BINARY"
    echo "  → Compiler d'abord depuis solver_bab/ :"
    echo "       cmake -B build -DCMAKE_BUILD_TYPE=Release \\"
    echo "             -DCMAKE_PREFIX_PATH=\"\$(pwd)/../external/ortools\" \\"
    echo "             -DGECODE_PREFIX=\"\$(pwd)/../external/gecode\" \\"
    echo "             -DSOLVER_GENERIC_BUILD=\"\$(pwd)/../solver_generic/build_cc\""
    echo "       cmake --build build --target bench_ssp_bab"
    exit 1
fi

# ── Run ────────────────────────────────────────────────────────────────────────
echo ""
"$BINARY" "$E" "$T" "$N" "$EPS" "$T_NODE" "$TIME_LIM"

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
