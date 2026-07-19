#!/bin/bash
# probe_aoti_batched_dynamic_B.sh — Sonde de faisabilité (Python pur, PAS de
# Gecode/C++) : AOTInductor peut-il exporter GNNSdEditScriptableBatched avec
# B dynamique (un seul .pt2 par n_g) plutôt que B figé (jusqu'à K .pt2 par
# n_g) ? cf. probe_aoti_batched_dynamic_B.py pour le détail.
#
# CPU : cette question porte sur le traçage/export (torch.export +
# aoti_compile_and_package), pas sur la performance — CPU compile plus vite
# et évite la queue GPU pour une simple sonde de faisabilité.
#
# Soumettre :
#   sbatch probe_aoti_batched_dynamic_B.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=probe_aoti_batched_dynB
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -uo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "======================================================================"
echo "  probe_aoti_batched_dynamic_B — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Noeud : $(hostname)"
echo "======================================================================"

python3 gnn_sd_edit/probe_aoti_batched_dynamic_B.py cpu

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
