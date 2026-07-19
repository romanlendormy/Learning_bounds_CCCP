#!/bin/bash
# compare_gnn_v1ld_n20.sh — Comparaison borne GNN one-shot (backend natif) vs
# V1LD (T_node=50 itérations de sous-gradient/nœud) sur l'instance
# n=20/m=5/N=50 — suite de compare_gnn_v1ld sur n=10 (cf. session
# "load train checkpoint into GECODE BAB").
#
# Pas de time_limit contraignant (mode "sans timeout") : chaque run a jusqu'à
# TIME_LIM_S pour prouver l'optimalité par lui-même.
#
# Soumettre :
#   sbatch compare_gnn_v1ld_n20.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=cmp_gnn_v1ld_n20
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n20-m5-a50-we-r42-S50.txt"
CKPT="gnn/runs/train_scaled/ckpt_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_10k_ep0197.pt"
EPS=0.10
T_ROOT=200
T_NODE=50
TIME_LIM_S=3300   # 55 min par run x2 (gnn + v1ld) = 110 min, sous les 2h SLURM (marge démarrage/venv)

VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export LD_LIBRARY_PATH="$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

BINARY="$REPO/solver_bab/build/bab_mkp_gecode"

echo "======================================================================"
echo "  compare_gnn_v1ld_n20 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  Dataset=$DATASET"
echo "  Checkpoint=$CKPT"
echo "  eps=$EPS  T_root=$T_ROOT  T_node=$T_NODE  time_limit=${TIME_LIM_S}s"
echo "======================================================================"

echo ""
echo "=== MODE=gnn (backend=native) ==="
"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" gnn non 0.1 "$CKPT" cpu python3 native

echo ""
echo "=== MODE=v1ld (T_node=$T_NODE, T_root=$T_ROOT) ==="
"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" v1ld

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
