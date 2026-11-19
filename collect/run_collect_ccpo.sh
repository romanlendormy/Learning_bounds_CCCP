#!/bin/bash
# collect/run_collect_ccpo.sh — Collecte des nœuds B&B CCPO pour l'entraînement du GNN
# (collect_ccpo_samples.py : n ∈ {6, 8, 10} × N ∈ {20, 50, 100}, 300 nœuds max par instance).
# Un seul processus ; plusieurs jours au total. Reprise : resoumettre le même script, la
# progression de chaque cellule est gardée dans OUT_DIR/_progress_n*_N*.json.
#
# Prérequis : build (cmake --build build), qui fournit build/solver_bab/bab_ccpo_collect.
# Soumettre depuis la racine du dépôt :
#   sbatch --export=ALL,OUT_DIR=$SCRATCH/ccpo_collect collect/run_collect_ccpo.sh

#SBATCH --account=def-qcappart
#SBATCH --job-name=ccpo_collect
#SBATCH --nodes=1
#SBATCH --time=23:30:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -u
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
source env/modules_trillium.sh
VENV_PATH="${VENV_PATH:-$HOME/venvs/cccp}"
source "$VENV_PATH/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

OUT_DIR="${OUT_DIR:-${SCRATCH:-$REPO/runs}/ccpo_collect}"
mkdir -p "$OUT_DIR"
echo "# collecte CCPO — job ${SLURM_JOB_ID:-local} sur $(hostname) — $(date) — sortie $OUT_DIR"
python3 -u collect/collect_ccpo_samples.py --out_dir "$OUT_DIR" --cap 300
echo "# fin (code $?) — $(date)"
