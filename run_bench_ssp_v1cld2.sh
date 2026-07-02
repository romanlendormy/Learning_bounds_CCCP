#!/bin/bash
# run_bench_ssp_v1cld2.sh — Convergence v1CLD2 sur une instance SSP synthétique.
#
# Analogue de run_bench_ssp_v1cld.sh (v1CLD), même instance, même Gecode de
# référence. v1CLD2 décompose ζ_{i,j} exactement par employé (E) et par période
# (T) via tables avant/arrière au lieu d'une DP dédiée par période forcée —
# borne rigoureusement identique à v1CLD, seule la granularité interne change.
#
# Soumettre :
#   sbatch run_bench_ssp_v1cld2.sh
#
# Varier les paramètres de l'instance :
#   sbatch --export=ALL,E=4,T=8,N=15   run_bench_ssp_v1cld2.sh
#   sbatch --export=ALL,E=3,T=12,N=20  run_bench_ssp_v1cld2.sh
#
# Forcer un T_MAX différent :
#   sbatch --export=ALL,T_MAX=2000 run_bench_ssp_v1cld2.sh
#
# Comparer v1LD, v1CLD et v1CLD2 sur la même figure (après avoir les CSV) :
#   python solver_generic/tests/plot_bench_ssp_compare.py \
#          solver_generic/results/bench_ssp_E3_T8_N10.csv \
#          solver_generic/results/bench_ssp_v1cld_E3_T8_N10.csv \
#          solver_generic/results/bench_ssp_v1cld2_E3_T8_N10.csv \
#          --out compare.png

# ── Ressources SLURM ──────────────────────────────────────────────────────────
#SBATCH --account=def-qcappart
#SBATCH --chdir=/home/romanlen/projects/def-qcappart/romanl/Learning_bounds_CCCP
#SBATCH --job-name=bench_ssp_v1cld2
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:10:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

# ── Paramètres ────────────────────────────────────────────────────────────────
E="${E:-3}"
T="${T:-8}"
N="${N:-10}"
T_MAX="${T_MAX:-5000}"

# ── Chemins ───────────────────────────────────────────────────────────────────
REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
VENV_PATH="$HOME/envs/learning_bounds"
SOLVER_DIR="$REPO/solver_generic"
BINARY="$SOLVER_DIR/build_cc/bench_ssp_v1cld2"
OUT_DIR="$SOLVER_DIR/results"
CSV_FILE="$OUT_DIR/bench_ssp_v1cld2_E${E}_T${T}_N${N}.csv"
FIG_FILE="$OUT_DIR/bench_ssp_v1cld2_E${E}_T${T}_N${N}.png"

# ── Modules & venv ────────────────────────────────────────────────────────────
module load StdEnv/2023 gcc/12.3.0 python/3.11.5 2>/dev/null || \
    module load python/3.11 2>/dev/null || true

source "$VENV_PATH/bin/activate"

export LD_LIBRARY_PATH="$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

mkdir -p "$OUT_DIR" "$REPO/logs"

echo "======================================================================"
echo "  bench_ssp v1CLD2 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  Instance : E=$E  T=$T  N=$N  T_MAX=$T_MAX"
echo "  CSV  : $CSV_FILE"
echo "  Fig  : $FIG_FILE"
echo "======================================================================"

# ── Vérification du binaire ───────────────────────────────────────────────────
if [ ! -x "$BINARY" ]; then
    echo "ERREUR : binaire introuvable : $BINARY"
    echo "  → Compiler d'abord depuis solver_generic/ :"
    echo "       cmake -B build_cc -DCMAKE_BUILD_TYPE=Release \\"
    echo "             -DGECODE_PREFIX=\"\$(pwd)/../external/gecode\""
    echo "       cmake --build build_cc --target bench_ssp_v1cld2"
    exit 1
fi

# ── Run bench_ssp_v1cld2 ──────────────────────────────────────────────────────
echo ""
echo "=== Sous-gradient v1CLD2 + Gecode exact — $(date) ==="
"$BINARY" "$E" "$T" "$N" "$T_MAX" "$CSV_FILE"

# ── Génération de la figure ───────────────────────────────────────────────────
echo ""
echo "=== Figure convergence v1CLD2 — $(date) ==="
python "$SOLVER_DIR/tests/plot_bench_ssp.py" "$CSV_FILE" --out "$FIG_FILE"

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "  CSV     : $CSV_FILE"
echo "  Figure  : $FIG_FILE"
echo "======================================================================"
