#!/bin/bash
# bench_zeta_openmp_test.sh — Vérifie si activer -fopenmp pour bnb_solver.cpp
# (solve_zeta_batch, déjà écrit avec #pragma omp parallel for mais jamais
# compilé avec -fopenmp jusqu'ici, cf. mémoire projet
# "gnn_inference_forward_cost_benchmark") réduit effectivement le temps de
# calcul de la borne ζ à la racine (n_g=20, coût dominant mesuré à 41.6ms).
#
# Soumettre :
#   sbatch bench_zeta_openmp_test.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=bench_zeta_omp
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:10:00
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

CKPT="gnn/runs/train_scaled/ckpt_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_10k_ep0197.pt"
PROBE="$REPO/solver_bab/build/gnn_native_probe"
DATASET="$REPO/data/knapsack/datasets-pisinger/ccmk-song-pisinger-n20-m5-a50-we-r42-S50.txt"

echo "======================================================================"
echo "  bench_zeta_openmp_test — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)   OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "======================================================================"

echo ""
echo "=== n=20/N=50, device=cuda, 200 itérations à la racine (n_g=20, ζ dominant) ==="
"$PROBE" "$DATASET" "$CKPT" 0.10 cuda 0 200 50

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
