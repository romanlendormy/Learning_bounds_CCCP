#!/bin/bash
# validate_sdedit_ipc.sh — Job de validation courte : backend IPC (variant
# sd_edit, nouvellement branché — cf. session "comparaison n=10/N={20,50,100}
# ref/v1ld/v1cld/gnn") sur CPU puis GPU, avant de lancer le job de comparaison
# complet. Vérifie : (1) pas de hang/deadlock, (2) temps d'inférence par nœud
# raisonnable sur un vrai nœud de calcul (le login node a montré ~38s/appel,
# probablement de la contention CPU partagée), (3) device=cuda fonctionne bien
# via IPC (jamais testé jusqu'ici — tous les runs précédents utilisaient le
# backend natif).
#
# Soumettre :
#   sbatch validate_sdedit_ipc.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=validate_sdedit_ipc
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n10-m5-a50-we-r42-S20.txt"
CKPT="gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
EPS=0.10
T_ROOT=200
T_NODE=50
TIME_LIM_S=300

VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export LD_LIBRARY_PATH="$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

BINARY="$REPO/solver_bab/build/bab_mkp_gecode"

echo "======================================================================"
echo "  validate_sdedit_ipc — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  Dataset=$DATASET"
echo "  Checkpoint=$CKPT"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "=== MODE=gnn backend=ipc variant=sd_edit device=cpu ==="
time "$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" gnn non 0.1 "$CKPT" cpu python3 ipc sd_edit

echo ""
echo "=== MODE=gnn backend=ipc variant=sd_edit device=cuda ==="
time "$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" gnn non 0.1 "$CKPT" cuda python3 ipc sd_edit

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
