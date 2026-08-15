#!/bin/bash
# run_ref_array.sh — mode ref (squelette nu, sans borne) sur les 120 instances
# de test_paper/, une tâche d'array par instance (test_paper/instances_list.txt,
# format : n N m seed path). CPU seul (ref n'appelle jamais le GNN), pas de GPU.
#
# time_limit=0 (illimité) — décision explicite de l'utilisateur du 2026-08-13 :
# le mur SLURM (24h) est le seul filet de sécurité, comme pour les jobs
# compare_ref_gnn_batched_*_notimeout en cours sur ce projet.
#
# Soumettre :
#   sbatch --array=1-120 test_paper/run_ref_array.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=test_paper_ref
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%x_%A_%a.out
#SBATCH --error=logs/slurm_%x_%A_%a.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )}"
cd "$REPO"

source solver_bab/setup_gpu_env.sh

TESTDIR="$REPO/test_paper"
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TESTDIR/instances_list.txt")
read -r N_ITEMS N_SCEN M SEED RELPATH <<< "$LINE"
DATASET="$TESTDIR/$RELPATH"

if [[ ! -f "$DATASET" ]]; then
    echo "[ERREUR] instance introuvable : $DATASET" >&2
    exit 1
fi

BINARY="$REPO/solver_bab/build/bab_mkp_gecode"
EPS=0.10
T_ROOT=200
T_NODE=50
TIME_LIM_S=0   # illimité — cf. en-tête

LOGFILE="$TESTDIR/logs/ref_n${N_ITEMS}_N${N_SCEN}_seed${SEED}.log"
RESULT_CSV="$TESTDIR/results/ref_n${N_ITEMS}_N${N_SCEN}_seed${SEED}.csv"

echo "======================================================================" | tee "$LOGFILE"
echo "  test_paper ref — n=$N_ITEMS N=$N_SCEN m=$M seed=$SEED — $(date)" | tee -a "$LOGFILE"
echo "  Job : ${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}   Nœud : $(hostname)" | tee -a "$LOGFILE"
echo "======================================================================" | tee -a "$LOGFILE"

"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" ref 2>&1 | tee -a "$LOGFILE"

VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
python3 "$TESTDIR/parse_result.py" ref "$N_ITEMS" "$N_SCEN" "$M" "$SEED" "$LOGFILE" > "$RESULT_CSV"

echo "" | tee -a "$LOGFILE"
echo "Terminé — $(date)" | tee -a "$LOGFILE"
