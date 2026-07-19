#!/bin/bash
# compare_ref_gnn_song.sh — REF (sans borne) vs GNN natif (device=cuda) sur une
# instance du jeu "song" (Song et al.), pas "pisinger" comme les tests
# précédents — n=20, m=5, N=100 (ccmk-song-mknap1_4_m5-S100.txt), compatible
# avec le checkpoint (m=5, n_list=[10,20]). Cf. session "load train checkpoint
# into GECODE BAB".
#
# Soumettre :
#   sbatch compare_ref_gnn_song.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=cmp_ref_gnn_song
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

DATASET="data/knapsack/datasets-song/ccmk-song-mknap1_4_m5-S100.txt"
CKPT="gnn/runs/train_scaled/ckpt_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_10k_ep0197.pt"
EPS=0.10
T_ROOT=200
T_NODE=50
TIME_LIM_S=3300

VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export LD_LIBRARY_PATH="$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

BINARY="$REPO/solver_bab/build/bab_mkp_gecode"

echo "======================================================================"
echo "  compare_ref_gnn_song — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  Dataset=$DATASET"
echo "  eps=$EPS  T_root=$T_ROOT  T_node=$T_NODE  time_limit=${TIME_LIM_S}s"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "=== MODE=ref (squelette nu, sans borne) ==="
"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" ref

echo ""
echo "=== MODE=gnn (backend=native, device=cuda) ==="
"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" gnn non 0.1 "$CKPT" cuda python3 native

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
