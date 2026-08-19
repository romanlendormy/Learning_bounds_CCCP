#!/bin/bash
# run_zstats_v1cld_n20_N500.sh — Mesure la fréquence de résolution précoce des
# z (pendant le x-branching) sur l'instance seed=43 (n=20, N=500), mêmes
# réglages que le run de comparaison V1LD/V1CLD déjà fait (K=64, surrogate,
# cuda) — via le binaire autonome bab_mkp_gecode_v1cld_gnn_batched (rebuilt
# avec les compteurs de diagnostic dbg_zstats_*, gecode_bab_v1cld_gnn_batched.cpp).

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=zstats_v1cld_n20_N500
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
CKPT="gnn_sd_edit/runs_v1cld/ckpt_v1cld_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_v1cld_n10_20_500ep_ep0500.pt"

"$BUILD/bab_mkp_gecode_v1cld_gnn_batched" "$DATASET" "$CKPT" 0.10 -1 0 cuda 64
