#!/bin/bash
# run_sanity_v1cld_gnn_batched_mode.sh — Vérifie que le mode CLI intégré
# "gnn_batched_v1cld" de bab_mkp_gecode (main_gecode.cpp) donne le MÊME
# résultat que le binaire autonome bab_mkp_gecode_v1cld_gnn_batched déjà
# validé (test_paper/run_sanity_v1cld_gnn_batched.sh : optimum=2366.3560
# identique au mode "v1cld" exact) — et reproduit la structure exacte des
# appels de run_gnn_array.sh (V1LD) pour V1CLD.

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=sanity_v1cld_gnn_batched_mode
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )}"
cd "$REPO"

source solver_bab/setup_gpu_env.sh

BUILD="$REPO/solver_bab/build"
DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n10-m5-a50-we-r42-S20.txt"
CKPT="gnn_sd_edit/runs_v1cld/ckpt_v1cld_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_v1cld_n10_20_500ep_ep0500.pt"
EPS=0.10
T_ROOT=200
T_NODE=50
TIME_LIM_S=0
K=8

echo "======================================================================"
echo "  Sanity check — mode gnn_batched_v1cld (bab_mkp_gecode) vs v1cld exact"
echo "  Attendu : Valeur optimale = 2366.3560 (identique au sanity check"
echo "  précédent sur le binaire autonome)"
echo "======================================================================"

"$BUILD/bab_mkp_gecode" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" gnn_batched_v1cld \
  non 0.1 "$CKPT" cpu python3 ipc sd_edit 0.0 0.0 1.0 "$K" 1.0 - 1 native
