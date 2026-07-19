#!/bin/bash
# check_freeze_and_K_sweep_n20_N100.sh — Valide torch::jit::freeze +
# optimize_for_inference (K=64, comparable au dernier run stride=1 connu :
# job 16675158, 11494 nœuds, obj=5221.9914, 12.671s, borne 9.185s dont
# fwd 5.334s), PUIS balaye K (32/64/128/256) avec freeze+optimize actifs
# pour voir si des batches plus gros amortissent mieux le coût fixe du
# forward (hypothèse : le goulot est le dispatch/lancement de kernels, pas
# le calcul — cf. gnn_light : ÷4.86 paramètres ne donne que ×1.33).
#
# gnn_node_stride=1 partout (remis par défaut, cf. échange).
#
# Soumettre :
#   sbatch check_freeze_and_K_sweep_n20_N100.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=check_freeze_K_sweep_n20_N100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -uo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

source solver_bab/setup_gpu_env.sh
VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

BINARY="$REPO/solver_bab/build/bab_mkp_gecode"
CKPT="gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n20-m5-a50-we-r42-S100.txt"
EPS=0.10
T_ROOT=200
T_NODE=50
TIME_LIMIT=300

echo "======================================================================"
echo "  check_freeze_and_K_sweep_n20_N100 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Noeud : $(hostname)"
echo "  Réf. AVANT freeze (job 16675158, K=64, stride=1) : 11494 nœuds,"
echo "  obj=5221.9914, 12.671s (borne 9.185s dont fwd 5.334s)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "--- REF (squelette nu, sans borne) ---"
"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIMIT" 0 "$T_ROOT" ref \
  || echo "[ERREUR] REF a échoué (code $?)"

for K in 32 64 128 256; do
  echo ""
  echo "--- gnn_batched K=$K (zeta=surrogate, stride=1, freeze+optimize_for_inference actifs) ---"
  "$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIMIT" 0 "$T_ROOT" gnn_batched \
    non 0.1 "$CKPT" cuda python3 ipc sd_edit 0.0 0.0 1.0 "$K" 1.0 surrogate 1 native \
    || echo "[ERREUR] K=$K a échoué (code $?)"
done

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
