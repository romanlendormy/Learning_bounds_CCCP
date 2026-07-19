#!/bin/bash
# probe_aoti_batched_dynamic_B_cuda.sh — Reprise de probe_aoti_batched_dynamic_B.sh
# mais device=cuda : le job check_aoti_batched_n20_N100 (16658298) a échoué
# sur aten._thnn_fused_gru_cell (pas de noyau fake/meta) lors de l'export
# CUDA batché avec B dynamique — jamais testé en CUDA jusqu'ici (la sonde
# initiale du 2026-07-17 tournait en CPU). Teste si c'est bien spécifique à
# CUDA (CPU avait réussi).
#
# Soumettre :
#   sbatch probe_aoti_batched_dynamic_B_cuda.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=probe_aoti_batched_dynB_cuda
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
echo "  probe_aoti_batched_dynamic_B_cuda — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Noeud : $(hostname)"
echo "======================================================================"

python3 gnn_sd_edit/probe_aoti_batched_dynamic_B.py cuda

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
