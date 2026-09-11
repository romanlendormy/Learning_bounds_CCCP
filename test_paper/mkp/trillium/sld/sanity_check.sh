#!/bin/bash
# sanity_check.sh — Valide le pipeline complet (ref + gnn_batched + parsing)
# sur UNE seule petite instance (n15/N100/seed43) avant de soumettre l'array
# complet sur les 120 instances. GPU requis (gnn_batched).
#
# Soumettre :
#   sbatch test_paper/mkp/trillium/sld/sanity_check.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=test_paper_sanity
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../../.." && pwd )}"
cd "$REPO"

source solver_bab/setup_gpu_env.sh
VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

TESTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
mkdir -p "$TESTDIR/logs"
BINARY="$REPO/solver_bab/build/bab_mkp_gecode"
DATASET="$TESTDIR/instance_n15_m5_N100_seed43.txt"
CKPT="gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
EPS=0.10
T_ROOT=200
T_NODE=50
TIME_LIM_S=0
K=64

echo "======================================================================"
echo "  sanity_check — n=15 N=100 m=5 seed=43 — $(date)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "=== MODE=ref ==="
REF_LOG="$TESTDIR/logs/sanity_ref.log"
"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" ref 2>&1 | tee "$REF_LOG"

echo ""
echo "=== MODE=gnn_batched ==="
GNN_LOG="$TESTDIR/logs/sanity_gnn.log"
"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" gnn_batched \
  non 0.1 "$CKPT" cuda python3 ipc sd_edit 0.0 0.0 1.0 "$K" 1.0 surrogate 1 native \
  2>&1 | tee "$GNN_LOG"

echo ""
echo "=== Parsing ==="
python3 "$TESTDIR/parse_result.py" --header
python3 "$TESTDIR/parse_result.py" ref 15 100 5 43 "$REF_LOG"
python3 "$TESTDIR/parse_result.py" gnn_batched 15 100 5 43 "$GNN_LOG"

echo ""
echo "Terminé — $(date)"
