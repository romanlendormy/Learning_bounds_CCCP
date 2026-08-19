#!/bin/bash
# run_gnn_v1cld_single_n20_N500.sh — Test V1CLD gnn_batched sur UNE instance
# (n=20, N=500, seed=43) avec les MÊMES réglages que le run V1LD de production
# (K=64, zeta_method=surrogate, device=cuda, backend=native) — pour comparer
# directement à la ligne V1LD déjà collectée dans results.csv (même seed) :
#   20,500,5,43,gnn_batched,OPTIMAL,4637.3978,...,solve_time_s=53.802,
#   t_bab_pure_s=21.009, t_bound_total_s=32.793, t_gnn_graph_build_s=12.079,
#   t_gnn_forward_s=13.449, t_gnn_bound_compute_s=7.265
#
# Objectif : répartition du temps hors B&B pur (graphe / forward GNN / ζ).

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=test_gnn_v1cld_n20_N500
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
VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

TESTDIR="$REPO/test_paper"
DATASET="$TESTDIR/instance_n20_m5_N500_seed43.txt"
BINARY="$REPO/solver_bab/build/bab_mkp_gecode"
CKPT="gnn_sd_edit/runs_v1cld/ckpt_v1cld_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_v1cld_n10_20_500ep_ep0500.pt"
EPS=0.10
T_ROOT=200
T_NODE=50
TIME_LIM_S=0
K=64

N_ITEMS=20; N_SCEN=500; M=5; SEED=43

LOGFILE="$TESTDIR/logs/gnn_v1cld_n${N_ITEMS}_N${N_SCEN}_seed${SEED}.log"
RESULT_CSV="$TESTDIR/results/gnn_v1cld_n${N_ITEMS}_N${N_SCEN}_seed${SEED}.csv"

{
echo "======================================================================"
echo "  test_paper gnn_batched_v1cld — n=$N_ITEMS N=$N_SCEN m=$M seed=$SEED — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  K=$K  zeta_method=surrogate  gnn_node_stride=1  backend=native"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
} | tee "$LOGFILE"

"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" gnn_batched_v1cld \
  non 0.1 "$CKPT" cuda python3 ipc sd_edit 0.0 0.0 1.0 "$K" 1.0 surrogate 1 native \
  2>&1 | tee -a "$LOGFILE"

python3 "$TESTDIR/parse_result.py" gnn_batched "$N_ITEMS" "$N_SCEN" "$M" "$SEED" "$LOGFILE" > "$RESULT_CSV"

echo "" | tee -a "$LOGFILE"
echo "Terminé — $(date)" | tee -a "$LOGFILE"
