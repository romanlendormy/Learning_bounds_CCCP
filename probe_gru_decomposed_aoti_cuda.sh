#!/bin/bash
# probe_gru_decomposed_aoti_cuda.sh — Valide la décomposition du GRUCell
# (contourne aten::_thnn_fused_gru_cell, sans noyau fake/meta CUDA) pour
# débloquer l'export AOTInductor batché avec B dynamique sur CUDA (cf.
# probe_gru_decomposed_aoti_cuda.py).
#
# Soumettre :
#   sbatch probe_gru_decomposed_aoti_cuda.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=probe_gru_decomposed_aoti_cuda
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -uo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

source solver_bab/setup_gpu_env.sh
VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "======================================================================"
echo "  probe_gru_decomposed_aoti_cuda — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Noeud : $(hostname)"
echo "======================================================================"

python3 gnn_sd_edit/probe_gru_decomposed_aoti_cuda.py cuda

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
