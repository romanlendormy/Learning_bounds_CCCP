#!/bin/bash
# test_paper/ccpo/selection/select_instances.sh — choix des 10 instances faisables par cellule (protocole de l'article) :
#   étape 1 (gen_and_select.py) : 40 candidats par cellule (idx = 100, 110, …, 490), un run REF de 120 s chacun ;
#           accepté si REF trouve un incumbent ou si le glouton est faisable, rejeté si infaisable prouvé, sinon indéterminé ;
#   étape 2 (stage2_certify.py) : les indéterminés sont repris avec REF et FEAS, 1800 s chacun ;
#   sélection : les 10 premiers acceptés par indice, étapes 1 et 2 confondues -> instances_list.txt, candidates.csv.
# Sorties dans $OUT_DIR (défaut $SCRATCH/test_paper_ccpo/selection). La sélection de l'article est figée dans
# ../instances_list.txt et candidates.csv : un nouveau passage peut en différer (verdicts sous limite de temps, et le
# binaire actuel inclut le presolve exact, absent du binaire utilisé pour la sélection).
# Soumettre depuis la racine du dépôt (nœud CPU entier) :
#   sbatch test_paper/ccpo/selection/select_instances.sh

#SBATCH --account=def-qcappart
#SBATCH --job-name=ccpo_select
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:30:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO"
source env/modules_trillium.sh
# Build CUDA (repéré au stub NVSHMEM) : bibliothèques CUDA nécessaires même sans GPU
[[ -e "$REPO/build/solver_bab/libnvshmem_host.so.3" ]] && source env/setup_gpu_env.sh
VENV_PATH="${VENV_PATH:-$HOME/venvs/cccp}"
source "$VENV_PATH/bin/activate"
export OMP_NUM_THREADS=1
OUT="${OUT_DIR:-${SCRATCH:-$REPO/runs}/test_paper_ccpo/selection}"
BIN="$REPO/build/solver_bab/bab_ccpo"
SEL="$REPO/test_paper/ccpo/selection"

python3 -u "$SEL/gen_and_select.py" "$REPO" "$OUT" "$BIN" "$(command -v python3)" \
    --ns 8,10,12,15 --Ns 100,500,1000 --maxc 40 --keep 10 --t_screen 120 --P 100
python3 -u "$SEL/stage2_certify.py" "$OUT" "$BIN" --t2 1800 --P 90 --keep 10
