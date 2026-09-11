#!/bin/bash
#SBATCH --account=def-qcappart_cpu
#SBATCH --partition=cpubase_bycore_b4
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=dump_ckpt_args
#SBATCH --output=logs/dump_ckpt_args_%j.out
#SBATCH --error=logs/dump_ckpt_args_%j.err
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
module load StdEnv/2023
module load python/3.11
source "$HOME/envs/learning_bounds/bin/activate"
export OMP_NUM_THREADS=2

python3 "$REPO/gnn_ssp/dump_ckpt_args.py" \
  "$REPO/gnn_ssp/runs/train_scaled/ckpt_scaled_dld_E3_T12_N20_E3_T12_N50_E4_T12_N20_E4_T12_N50_E4_T8_N20_E4_T8_N50_lr0.0003_bs32_h64_seed42_v6_periodpool_1k_ep0150.pt"
