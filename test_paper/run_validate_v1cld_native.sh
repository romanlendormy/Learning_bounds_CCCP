#!/bin/bash
# run_validate_v1cld_native.sh — Compare test_gnn_v1cld_native_batched (C++)
# vs validate_v1cld_native_reference.py (référence Python) sur le nœud
# racine d'une même instance — le test de correction le plus important
# avant tout branchement dans le vrai B&B.

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=validate_v1cld_native
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )}"
cd "$REPO"

source solver_bab/setup_gpu_env.sh
VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n10-m5-a50-we-r42-S20.txt"
CKPT="gnn_sd_edit/runs_v1cld/ckpt_v1cld_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_v1cld_n10_20_500ep_ep0500.pt"

echo "=== C++ (test_gnn_v1cld_native_batched) ==="
./solver_bab/build/test_gnn_v1cld_native_batched "$DATASET" "$CKPT" cpu

echo ""
echo "=== Python (référence) ==="
python3 test_paper/validate_v1cld_native_reference.py "$DATASET" "$CKPT"
