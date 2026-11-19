#!/bin/bash
# test_paper/mkp/sld/run_ref_array.sh — Référence REF : B&B Gecode sans borne
# lagrangienne, sur les 120 instances de test (une tâche d'array par ligne de
# test_paper/mkp/instances_list.txt). CPU seul (le GPU alloué n'est pas utilisé).
#
# Soumettre depuis la racine du dépôt (cf. test_paper/mkp/README.md) :
#   sbatch --array=1-120 test_paper/mkp/sld/run_ref_array.sh

#SBATCH --account=def-qcappart
#SBATCH --job-name=mkp_ref
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%x_%A_%a.out
#SBATCH --error=logs/slurm_%x_%A_%a.err

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO"
source env/modules_trillium.sh
source env/setup_gpu_env.sh   # met aussi Gecode / OR-Tools dans LD_LIBRARY_PATH

TD="$REPO/test_paper/mkp"
INSTANCES_DIR="${INSTANCES_DIR:-$TD/instances}"
OUT="${OUT_DIR:-${SCRATCH:-$REPO/runs}/test_paper_mkp/ref}"
mkdir -p "$OUT/logs" "$OUT/results"

read -r N_ITEMS N_SCEN M SEED FILE <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TD/instances_list.txt")"
DATASET="$INSTANCES_DIR/$FILE"
[[ -f "$DATASET" ]] || { echo "[ERREUR] instance introuvable : $DATASET" >&2; exit 1; }

EPS=0.10; T_ROOT=200; T_NODE=50
TIME_LIM_S=0   # pas de limite côté solveur : le mur SLURM (24 h) est la seule limite

TAG="n${N_ITEMS}_N${N_SCEN}_seed${SEED}"
LOG="$OUT/logs/ref_$TAG.log"
{
echo "# ref — n=$N_ITEMS N=$N_SCEN m=$M seed=$SEED — $(date)"
echo "# job ${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0} — nœud $(hostname)"
} > "$LOG"

"$REPO/build/solver_bab/bab_mkp_gecode" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" ref \
  >> "$LOG" 2>&1

python3 "$TD/parse_result.py" ref "$N_ITEMS" "$N_SCEN" "$M" "$SEED" "$LOG" > "$OUT/results/ref_$TAG.csv"
