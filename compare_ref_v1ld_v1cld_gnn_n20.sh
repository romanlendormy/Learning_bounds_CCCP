#!/bin/bash
# compare_ref_v1ld_v1cld_gnn_n20.sh — Comparaison à 4 méthodes de borne du B&B
# Gecode CCMKP sur n=20, N∈{20,50,100} — cf. compare_ref_v1ld_v1cld_gnn_n10.sh
# (même structure). Checkpoint sd_edit couvre n=10 ET n=20 (pas hors distribution
# ici, contrairement au sweep n=25).
#
# Calibrage préalable (calibrate_n20.sh, cap 120s) a montré qu'à N=100 ni v1ld
# (104 ms/nœud, 1151 nœuds en 120s) ni gnn (7 ms/nœud mais 17008 nœuds en 120s)
# n'atteignent l'optimalité dans une petite fenêtre — arbre B&B bien plus gros
# qu'à n=10. time_limit relevé à 1800s/run (certains runs à N=100 resteront
# probablement FEASIBLE/TIMEOUT plutôt que OPTIMAL — donnée honnête en soi).
#
# Soumettre :
#   sbatch compare_ref_v1ld_v1cld_gnn_n20.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=cmp_4methods_n20
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

CKPT="gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
EPS=0.10
T_ROOT=200
T_NODE=50
TIME_LIM_S=1800   # 30 min/run — cf. calibrage, N=100 peut ne pas atteindre OPTIMAL

VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export LD_LIBRARY_PATH="$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

BINARY="$REPO/solver_bab/build/bab_mkp_gecode"

echo "======================================================================"
echo "  compare_ref_v1ld_v1cld_gnn_n20 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  Checkpoint GNN : $CKPT"
echo "  eps=$EPS  T_root=$T_ROOT  T_node=$T_NODE  time_limit=${TIME_LIM_S}s/run"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for N in 20 50 100; do
    DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n20-m5-a50-we-r42-S${N}.txt"

    echo ""
    echo "######################################################################"
    echo "  n=20  N=${N}  —  $(date)"
    echo "######################################################################"

    echo ""
    echo "=== [n=20 N=${N}] MODE=ref (squelette nu, sans borne) ==="
    "$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" ref

    echo ""
    echo "=== [n=20 N=${N}] MODE=v1ld (sous-gradient, T_root=$T_ROOT, T_node=$T_NODE) ==="
    "$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" v1ld

    echo ""
    echo "=== [n=20 N=${N}] MODE=gnn (backend=ipc, variant=sd_edit, device=cuda) ==="
    "$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" gnn non 0.1 "$CKPT" cuda python3 ipc sd_edit

    echo ""
    echo "=== [n=20 N=${N}] MODE=v1cld (sous-gradient, T_root=$T_ROOT, T_node=$T_NODE) ==="
    "$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" v1cld
done

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
