#!/bin/bash
# run_table_synth_rorqual.sh — Tableau base / V1LD / V1CLD (sweep_bab/table_synth.py)
# sur une liste d'instances synthétiques Song-Pisinger.
#
# Job CPU léger (1 seul core utilisé à la fois, séquentiel) mais potentiellement
# long : chaque run individuel est borné par --time_limit, mais avec 5
# instances x 3 méthodes le total peut monter à ~40 min dans le pire cas
# (voir NOTES : n=30 sature déjà son budget de 300s sur les 3 méthodes).
#
# Soumettre :
#   sbatch sweep_bab/run_table_synth_rorqual.sh
#
# Surcharger les instances/time_limit :
#   sbatch --export=ALL,TIME_LIMIT=600 sweep_bab/run_table_synth_rorqual.sh
#   sbatch --export=ALL,INSTANCES="10,5,50 20,5,50 30,5,50" sweep_bab/run_table_synth_rorqual.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=table_synth
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

TIME_LIMIT="${TIME_LIMIT:-300}"
INSTANCES="${INSTANCES:-10,5,50 10,5,100 10,5,500 20,5,50 30,5,50}"

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )}"
VENV_PATH="$HOME/envs/learning_bounds"

echo "======================================================================"
echo "  table_synth — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  TIME_LIMIT=$TIME_LIMIT  INSTANCES=$INSTANCES"
echo "======================================================================"

module load StdEnv/2023 gcc/12.3.0 python/3.11.5 2>/dev/null || \
    module load python/3.11 2>/dev/null || \
    echo "  (aucun module python chargé)"

if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "ERREUR : venv introuvable à $VENV_PATH"
    exit 1
fi

export LD_LIBRARY_PATH="$REPO/external/gecode/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$REPO/logs" "$REPO/results"

python "$REPO/sweep_bab/table_synth.py" \
    --time_limit "$TIME_LIMIT" \
    --instances $INSTANCES \
    --out_md "$REPO/results/table_synth.md"

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
