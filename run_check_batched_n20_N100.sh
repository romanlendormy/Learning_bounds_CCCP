#!/bin/bash
# run_check_batched_n20_N100.sh — revalidation rapide (une seule instance,
# n=20/N=100) du gain de batching GNN en B&B après le fix SCALE=1000->10000
# (bug faux-optimal, cf. project_gnn_batched_false_optimal_bug_investigation).
# Résultat de référence AVANT ce fix (job 16046384) : REF=8.4s/99302 nœuds,
# prod mono-nœud=70.0s/25890 nœuds, batché K=128=26.0s/25399 nœuds, objectif
# optimal identique (5221.992) partout.
#
# N'attend pas le sweep complet des 8 combos (job 16406490, en attente
# ~2 jours) — juste cette instance, pour trancher rapidement.
#
# Soumettre :
#   sbatch run_check_batched_n20_N100.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=check_batched_n20_N100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:20:00
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

CKPT="gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
PROD="$REPO/solver_bab/build/bab_mkp_gecode"
BATCHED="$REPO/solver_bab/build/bab_mkp_gecode_gnn_batched"
DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n20-m5-a50-we-r42-S100.txt"
EPS=0.10
T_ROOT=200
T_NODE=50
TIME_LIMIT=300

echo "======================================================================"
echo "  check_batched_n20_N100 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Noeud : $(hostname)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "--- REF (squelette nu) ---"
"$PROD" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIMIT" 0 "$T_ROOT" ref \
  || echo "[ERREUR] REF a échoué (code $?)"

echo ""
echo "--- PROD mono-nœud (gnn native sd_edit, K=1 implicite) ---"
"$PROD" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIMIT" 0 "$T_ROOT" gnn non 0.1 "$CKPT" cuda python3 native sd_edit 0.0 0.0 \
  || echo "[ERREUR] PROD a échoué (code $?)"

for K in 1 32 128; do
  echo ""
  echo "--- BATCHÉ K=$K ---"
  "$BATCHED" "$DATASET" "$CKPT" "$EPS" "$TIME_LIMIT" 0 cuda "$K" \
    || echo "[ERREUR] BATCHÉ K=$K a échoué (code $?)"
done

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
