#!/bin/bash
# run_compare_zaware_n20_N500.sh — Compare gnn_batched d'ORIGINE (ignore les z)
# vs Z-AWARE (les exclut) sur seed=43 (n=20, N=500) — même instance/checkpoint/
# réglages (K=64, device=cuda) que le run V1LD déjà collecté dans results.csv
# (objectif=4637.3978, solve_time_s=53.802, nodes=12975).

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=compare_zaware_n20_N500
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )}"
cd "$REPO"

source solver_bab/setup_gpu_env.sh

BUILD="$REPO/solver_bab/build"
DATASET="$REPO/test_paper/instance_n20_m5_N500_seed43.txt"
CKPT="gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"

echo "======================================================================"
echo "  [ORIGINE] bab_mkp_gecode mode=gnn_batched (K=64, cuda)"
echo "======================================================================"
"$BUILD/bab_mkp_gecode" "$DATASET" 0.10 50 0 0 200 gnn_batched \
  non 0.1 "$CKPT" cuda python3 ipc sd_edit 0.0 0.0 1.0 64 1.0 - 1 native

echo ""
echo "======================================================================"
echo "  [Z-AWARE] bab_mkp_gecode_gnn_batched_zaware (K=64, cuda)"
echo "======================================================================"
"$BUILD/bab_mkp_gecode_gnn_batched_zaware" "$DATASET" "$CKPT" 0.10 -1 0 cuda 64
