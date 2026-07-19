#!/bin/bash
# compare_ref_v1ld_v1cld_gnn_n10.sh — Comparaison à 4 méthodes de borne du B&B
# Gecode CCMKP sur n=10, N∈{20,50,100} :
#   1. ref   : squelette B&B nu, sans borne (référence pour isoler l'effet du
#              branchement/incumbent/moteur DFS de l'effet de la borne).
#   2. v1ld  : borne V1LD par descente de sous-gradient (T_root/T_node itérations).
#   3. gnn   : borne V1LD one-shot par GNN (checkpoint sd_edit, 500 epochs,
#              n=10/20, N=20/50/100 — gnn_sd_edit/runs/train_scaled/
#              ckpt_sdedit_scaled_n10_20_..._v1_sdedit_500ep_ep0500.pt),
#              backend IPC (variant=sd_edit — le backend natif ne supporte que
#              la topologie homogène, cf. session de mise en place), device=cuda.
#   4. v1cld : borne V1CLD par descente de sous-gradient (T_root/T_node itérations).
#
# Chaque run imprime : nodes, depth max, fails (dont % dû à la borne), t_solve
# total, t_bab_pure, et depuis le patch root_bound_value (cette session) la
# valeur de la borne à la racine + son gap vs l'optimal. v1ld/v1cld font
# T_root=200 itérations de sous-gradient à la racine puis T_node=50 à chaque
# nœud suivant (warm-start) ; gnn ne fait aucune itération (borne one-shot).
#
# Soumettre :
#   sbatch compare_ref_v1ld_v1cld_gnn_n10.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=cmp_4methods_n10
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

CKPT="gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
EPS=0.10
T_ROOT=200
T_NODE=50
TIME_LIM_S=600   # par run ; n=10 est petit, marge large pour l'inférence GNN

VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export LD_LIBRARY_PATH="$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

BINARY="$REPO/solver_bab/build/bab_mkp_gecode"

echo "======================================================================"
echo "  compare_ref_v1ld_v1cld_gnn_n10 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  Checkpoint GNN : $CKPT"
echo "  eps=$EPS  T_root=$T_ROOT  T_node=$T_NODE  time_limit=${TIME_LIM_S}s/run"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for N in 20 50 100; do
    DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n10-m5-a50-we-r42-S${N}.txt"

    echo ""
    echo "######################################################################"
    echo "  n=10  N=${N}  —  $(date)"
    echo "######################################################################"

    echo ""
    echo "=== [n=10 N=${N}] MODE=ref (squelette nu, sans borne) ==="
    "$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" ref

    echo ""
    echo "=== [n=10 N=${N}] MODE=v1ld (sous-gradient, T_root=$T_ROOT, T_node=$T_NODE) ==="
    "$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" v1ld

    echo ""
    echo "=== [n=10 N=${N}] MODE=gnn (backend=ipc, variant=sd_edit, device=cuda) ==="
    "$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" gnn non 0.1 "$CKPT" cuda python3 ipc sd_edit

    echo ""
    echo "=== [n=10 N=${N}] MODE=v1cld (sous-gradient, T_root=$T_ROOT, T_node=$T_NODE) ==="
    "$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" v1cld
done

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
