#!/bin/bash
# bench_gnn_cudagraph_probe.sh — Sonde de faisabilité CUDA graphs pour le
# forward GNN natif (cf. solver_bab/mkp/gnn_cudagraph_probe.cpp).
#
# Objectif : mesurer si le forward TorchScript de la borne GNN est capturable
# par at::cuda::CUDAGraph et, si oui, quel gain ça donne par rapport au chemin
# eager actuel — sur la même forme (n_g=20, N_z=50, m=5) que le benchmark
# sdedit_vs_homogene (job 15987340) qui a établi que le forward (~0.84ms/appel)
# est dominé par l'overhead fixe de dispatch, pas le volume de message passing
# (cf. mémoire "sdedit-vs-homogene-forward-overhead-confirmed").
#
# Soumettre :
#   sbatch bench_gnn_cudagraph_probe.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=bench_cudagraph_probe
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

module load cuda/12.6
VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export LD_LIBRARY_PATH="$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

MODULE_HOMOGENE="gnn/runs/train_scaled/ckpt_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_10k_ep0197.pt.scripted.pt"
N_G=20
N_Z=50
M=5
N_ITERS_EAGER=2000
N_WARMUP=50
N_ITERS_GRAPH=2000

echo "======================================================================"
echo "  bench_gnn_cudagraph_probe — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "=== (Re)compilation gnn_cudagraph_probe ==="
cmake -S solver_bab -B solver_bab/build -DCMAKE_BUILD_TYPE=Release -DGNN_DEVICE=cuda >/dev/null
cmake --build solver_bab/build --target gnn_cudagraph_probe -j "${SLURM_CPUS_PER_TASK:-4}"
PROBE="$REPO/solver_bab/build/gnn_cudagraph_probe"

echo ""
echo "======================================================================"
echo "  variant=homogene n_g=$N_G N_z=$N_Z m=$M"
echo "======================================================================"
"$PROBE" "$MODULE_HOMOGENE" cuda "$N_G" "$N_Z" "$M" "$N_ITERS_EAGER" "$N_WARMUP" "$N_ITERS_GRAPH" homogene

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
