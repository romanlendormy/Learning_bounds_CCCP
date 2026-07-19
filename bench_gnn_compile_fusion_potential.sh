#!/bin/bash
# bench_gnn_compile_fusion_potential.sh — Mesure du potentiel de fusion de
# kernels (torch.compile/Inductor/Triton) sur le forward GNN dense sd_edit,
# via torch.profiler (temps GPU réel, immunisé à l'overhead d'interpréteur
# Python — cf. mémoire "cudagraph-python-probe-overestimate") — AVANT
# d'investir dans un portage AOTInductor + C++ pour le backend natif.
#
# Soumettre :
#   sbatch bench_gnn_compile_fusion_potential.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=bench_compile_fusion
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export LD_LIBRARY_PATH="$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

CKPT="gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"

echo "======================================================================"
echo "  bench_gnn_compile_fusion_potential — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python3 gnn_sd_edit/profile_compile_fusion_potential.py \
    --checkpoint "$CKPT" --device cuda --N 50 --n_g 20 5 \
    --n_warmup_eager 30 --n_warmup_compile 10 --n_iters 200

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
