#!/bin/bash
# run_bench_ssp_bab_file.sh — Gecode base vs B&B base_ref vs B&B v1CLD (vs v1LD,
# désactivable) sur une instance SSP chargée depuis fichiers texte générée par
# ssp_stochastic (cf. ssp_stochastic/generator_adaptive.py + export_to_txt.py).
#
# Contrairement à run_bench_ssp_bab.sh (instance jouet synthétique, générée en
# dur dans bench_ssp_bab.cpp), l'instance ici vient du nouveau générateur
# calibré (demande bimodale AR(1), automate à contraintes réalistes).
#
# Soumettre :
#   sbatch --export=ALL,INSTANCE_DIR=solver_bab/ssp/instances/E3_T8_N20_seed42 \
#          run_bench_ssp_bab_file.sh
#
# Varier le budget :
#   sbatch --export=ALL,INSTANCE_DIR=...,T_NODE=50,TIME_LIM=0,TIME_LIM_LD=-1 \
#          run_bench_ssp_bab_file.sh
#
# TIME_LIM=0    : illimité par méthode (Gecode base / base_ref / v1CLD), borné
#                 par le --time SLURM ci-dessous.
# TIME_LIM_LD<0 : désactive B&B v1LD (ζ exact via CP-SAT à chaque nœud, coûteux
#                 — cf. gecode_bab_ssp.hpp). Défaut -1 (désactivé).

# ── Ressources SLURM ──────────────────────────────────────────────────────────
#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=bench_ssp_bab_file
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

# ── Paramètres ────────────────────────────────────────────────────────────────
: "${INSTANCE_DIR:?INSTANCE_DIR requis (répertoire avec instance.txt/automaton.txt/cost.txt/demand.txt)}"
T_NODE="${T_NODE:-30}"
TIME_LIM="${TIME_LIM:-0}"        # secondes, par méthode ; 0 = illimité (borné par le --time SLURM ci-dessus)
TIME_LIM_LD="${TIME_LIM_LD:--1}" # < 0 = B&B v1LD désactivé (cf. en-tête)

# ── Chemins ───────────────────────────────────────────────────────────────────
REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
SOLVER_BAB_DIR="$REPO/solver_bab"
BINARY="$SOLVER_BAB_DIR/build/bench_ssp_bab_file"

# ── Modules ───────────────────────────────────────────────────────────────────
module load StdEnv/2023 gcc/12.3.0 2>/dev/null || true

export LD_LIBRARY_PATH="$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

mkdir -p "$REPO/logs"

echo "======================================================================"
echo "  bench_ssp_bab_file — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  Instance : $INSTANCE_DIR"
echo "  T_node=$T_NODE  time_lim=${TIME_LIM}s/méthode  time_lim_v1LD=${TIME_LIM_LD}s"
echo "======================================================================"

# ── Vérifications ──────────────────────────────────────────────────────────────
if [ ! -x "$BINARY" ]; then
    echo "ERREUR : binaire introuvable : $BINARY"
    echo "  → Compiler d'abord depuis solver_bab/ :"
    echo "       cmake --build build --target bench_ssp_bab_file"
    exit 1
fi
if [ ! -f "$REPO/$INSTANCE_DIR/instance.txt" ]; then
    echo "ERREUR : instance introuvable : $REPO/$INSTANCE_DIR/instance.txt"
    exit 1
fi

# ── Run ────────────────────────────────────────────────────────────────────────
echo ""
"$BINARY" "$REPO/$INSTANCE_DIR" "$T_NODE" "$TIME_LIM" "$TIME_LIM_LD"

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
