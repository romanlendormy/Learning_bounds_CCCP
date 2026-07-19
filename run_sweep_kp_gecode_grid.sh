#!/bin/bash
# run_sweep_kp_gecode_grid.sh — Sweep REF vs V1LD (bab_kp_gecode) sur grille n×N.
#
# Grille croisée n∈{10,15,20,25,30} × N∈{50,100,250,500}, weakly_correlated,
# 2 seeds, 2 modes (ref/v1ld) — cf. collect/sweep_kp_gecode_grid.py pour le
# détail. Objectif : voir si/où V1LD finit par payer son coût par nœud face
# à REF quand n ET N grandissent (à n=15/N=100, REF gagnait encore en temps
# réel malgré plus de nœuds, cf. session).
#
# 80 runs, parallélisés (ThreadPoolExecutor, MAX_WORKERS=8) — cpus-per-task=8
# ci-dessous doit rester cohérent avec MAX_WORKERS.
#
# Soumettre :
#   sbatch run_sweep_kp_gecode_grid.sh
#
# Varier la grille / le budget :
#   sbatch --export=ALL,N_LIST="10,20",CAP_LIST="100,500" run_sweep_kp_gecode_grid.sh
#   sbatch --export=ALL,TIME_LIM_S=60 run_sweep_kp_gecode_grid.sh

# ── Ressources SLURM ──────────────────────────────────────────────────────────
#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=sweep_kp_gecode_grid
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

# ── Paramètres (surchargeables via sbatch --export) ──────────────────────────
export N_LIST="${N_LIST:-10,15,20,25,30}"
export CAP_LIST="${CAP_LIST:-50,100,250,500}"
export SEEDS="${SEEDS:-1,2}"
export CORRELATION="${CORRELATION:-weakly_correlated}"
export EPSILON="${EPSILON:-0.1}"
export ALPHA="${ALPHA:-0.5}"
export R="${R:-1000}"
export T_ROOT="${T_ROOT:-200}"
export T_NODE="${T_NODE:-30}"
export TIME_LIM_S="${TIME_LIM_S:-120}"
export MAX_WORKERS="${MAX_WORKERS:-8}"

VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export LD_LIBRARY_PATH="$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$REPO/logs"

echo "======================================================================"
echo "  sweep_kp_gecode_grid — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "======================================================================"

BINARY="$REPO/solver_bab/build/bab_kp_gecode"
if [ ! -x "$BINARY" ]; then
    echo "ERREUR : binaire introuvable : $BINARY"
    echo "  → Compiler d'abord :"
    echo "       cmake -S solver_bab -B solver_bab/build -DCMAKE_PREFIX_PATH=\"\$(pwd)/external/ortools\" \\"
    echo "             -DGECODE_PREFIX=\"\$(pwd)/external/gecode\" -DSOLVER_GENERIC_BUILD=\"\$(pwd)/solver_generic/build_cc\""
    echo "       cmake --build solver_bab/build --target bab_kp_gecode -j4"
    exit 1
fi

python3 "$REPO/collect/sweep_kp_gecode_grid.py"

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
