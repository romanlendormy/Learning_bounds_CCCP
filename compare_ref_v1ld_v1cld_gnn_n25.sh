#!/bin/bash
# compare_ref_v1ld_v1cld_gnn_n25.sh — Comparaison à 4 méthodes de borne du B&B
# Gecode CCMKP sur n=25, N∈{20,50,100} — cf. compare_ref_v1ld_v1cld_gnn_n20.sh
# (même structure). ATTENTION : n=25 est hors distribution pour le checkpoint
# sd_edit (entraîné uniquement sur n=10/20).
#
# Calibrage préalable (calibrate_n25.sh, cap 120s) : à N=100, v1ld ne termine
# même pas le calcul de la borne racine en moins de 120s (165-337s pour les
# 200 itérations de sous-gradient T_root, un seul nœud traité) — bien plus
# dur qu'à n=20. gnn reste rapide par nœud (~7.6ms) mais n'atteint pas non
# plus l'optimalité en 120s (15782 nœuds explorés, TIMEOUT). Décision (cf.
# session) : garder N=100 dans le sweep avec time_limit=1800s/run, en
# acceptant que v1ld/v1cld y restent probablement FEASIBLE/TIMEOUT plutôt que
# OPTIMAL — donnée honnête ("meilleure borne atteinte en 30 min"), pas un bug.
# N=20/N=50 devraient converger normalement.
#
# Soumettre :
#   sbatch compare_ref_v1ld_v1cld_gnn_n25.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=cmp_4methods_n25
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
TIME_LIM_S=1800   # 30 min/run — N=100 restera probablement FEASIBLE/TIMEOUT pour v1ld/v1cld (cf. calibrage)

VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export LD_LIBRARY_PATH="$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

BINARY="$REPO/solver_bab/build/bab_mkp_gecode"

echo "======================================================================"
echo "  compare_ref_v1ld_v1cld_gnn_n25 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  Checkpoint GNN : $CKPT  (HORS DISTRIBUTION : entraîné sur n=10/20, pas n=25)"
echo "  eps=$EPS  T_root=$T_ROOT  T_node=$T_NODE  time_limit=${TIME_LIM_S}s/run"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for N in 20 50 100; do
    DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n25-m5-a50-we-r42-S${N}.txt"

    echo ""
    echo "######################################################################"
    echo "  n=25  N=${N}  —  $(date)"
    echo "######################################################################"

    echo ""
    echo "=== [n=25 N=${N}] MODE=ref (squelette nu, sans borne) ==="
    "$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" ref

    echo ""
    echo "=== [n=25 N=${N}] MODE=v1ld (sous-gradient, T_root=$T_ROOT, T_node=$T_NODE) ==="
    "$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" v1ld

    echo ""
    echo "=== [n=25 N=${N}] MODE=gnn (backend=ipc, variant=sd_edit, device=cuda) ==="
    "$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" gnn non 0.1 "$CKPT" cuda python3 ipc sd_edit

    echo ""
    echo "=== [n=25 N=${N}] MODE=v1cld (sous-gradient, T_root=$T_ROOT, T_node=$T_NODE) ==="
    "$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" v1cld
done

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
