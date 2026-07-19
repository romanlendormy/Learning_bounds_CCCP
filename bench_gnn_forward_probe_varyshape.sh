#!/bin/bash
# bench_gnn_forward_probe_varyshape.sh — Teste si le changement de forme du
# graphe d'entrée (n_g = nombre d'items encore libres, qui diminue avec la
# profondeur dans un vrai B&B) redéclenche la compilation TorchScript à
# chaque nouvelle forme rencontrée. Suite à bench_gnn_forward_probe.sh, qui a
# montré un pattern reproductible : exactement les 2 premiers appels sont
# très lents (jusqu'à ~2.5s), puis régime stable — sur CPU ET GPU, donc pas
# un simple warm-up CUDA mais probablement l'executor JIT de TorchScript
# (mode "profiling" : 1-2 exécutions non optimisées avant fusion/spécialisation).
#
# Ici : alterne le masque entre n_g=n (tout libre) et n_g=n/2 toutes les
# vary_ng itérations, pour voir si CHAQUE bascule de forme coûte à nouveau
# ~2s (recompilation par forme, pas de cache) ou si seule la première
# rencontre de chaque forme est coûteuse (cache par forme, bon signe).
#
# Soumettre :
#   sbatch bench_gnn_forward_probe_varyshape.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=bench_gnn_probe_vary
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:15:00
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
echo "  bench_gnn_forward_probe_varyshape — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "=== n=20/N=50, device=cuda, bascule n_g toutes les 10 itérations (200 iters) ==="
"$PROBE" "$DATASET" "$CKPT" 0.10 cuda 0 200 0 10

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
