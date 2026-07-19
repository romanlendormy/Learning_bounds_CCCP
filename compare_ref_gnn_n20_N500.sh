#!/bin/bash
# compare_ref_gnn_n20_N500.sh — Gecode BAB : REF (squelette nu, sans borne)
# vs GNN natif (checkpoint homogène, backend native, device=cpu) sur
# l'instance n=20/m=5/N=500 (ccmk-song-pisinger-n20-m5-a50-we-r42-S500.txt,
# générée le 2026-07-08 — seul S=50 existait pour n=20 jusqu'ici). n=20 est
# dans la distribution d'entraînement (n∈{10,20}), mais N=500 est hors
# distribution (le dataset train_scaled ne couvrait que N∈{20,50,100}, cf.
# [[project_train_scaled_data_selection]]) — donc généralisation OOD en N,
# pas en n. Calqué sur compare_ref_gnn_n25.sh.
#
# Checkpoint : ckpt_scaled_..._v1_10k_ep0200.pt (idem n25).
#
# Soumettre :
#   sbatch compare_ref_gnn_n20_N500.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=cmp_ref_gnn_n20_N500
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n20-m5-a50-we-r42-S500.txt"
CKPT="gnn/runs/train_scaled/ckpt_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_10k_ep0200.pt"
EPS=0.10
T_ROOT=200
T_NODE=50
TIME_LIM_S=3300   # par méthode ; 2 méthodes x 55 min < 2h SLURM

VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export LD_LIBRARY_PATH="$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

BINARY="$REPO/solver_bab/build/bab_mkp_gecode"

echo "======================================================================"
echo "  compare_ref_gnn_n20_N500 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  Dataset=$DATASET"
echo "  Checkpoint=$CKPT"
echo "  eps=$EPS  T_root=$T_ROOT  T_node=$T_NODE  time_limit=${TIME_LIM_S}s"
echo "======================================================================"

echo ""
echo "=== MODE=ref (squelette nu, sans borne) ==="
"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" ref

echo ""
echo "=== MODE=gnn (backend=native, device=cpu, checkpoint homogène ep0200) ==="
"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" gnn non 0.1 "$CKPT" cpu python3 native

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
