#!/bin/bash
# bench_gnn_batched_bab_vs_production.sh — Étape 2e du plan de batching :
# compare le B&B GNN batché (bab_mkp_gecode_gnn_batched) au B&B GNN mono-nœud
# de PRODUCTION (bab_mkp_gecode ... gnn native sd_edit), directement — pas
# une comparaison interne K=1 (cf. décision utilisateur 2026-07-14).
#
# 1. Petite instance (n=10/S20) : non-régression (nodes/objectif doivent
#    correspondre, à la tolérance numérique déjà validée à l'Étape 2d ~1e-3 —
#    PAS bit-exact, contrairement à l'Étape 1, puisque le batché passe par un
#    chemin de calcul différent même à K=1).
# 2. Instance connue lente pour le GNN mono-nœud (n=20/N=100 — cf. mémoire
#    "compare-ref-gnn-n20-N100-indist" : GNN ÷3.8 nœuds mais ×12.7 temps total
#    vs REF) : LE test qui compte — le batching corrige-t-il ce problème ?
#
# Soumettre :
#   sbatch bench_gnn_batched_bab_vs_production.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=gnn_batched_bab_vs_prod
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

source "$REPO/solver_bab/setup_gpu_env.sh"
VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

CKPT="gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
PROD="$REPO/solver_bab/build/bab_mkp_gecode"
BATCHED="$REPO/solver_bab/build/bab_mkp_gecode_gnn_batched"

echo "======================================================================"
echo "  bench_gnn_batched_bab_vs_production — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

DATASET1="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n10-m5-a50-we-r42-S20.txt"
echo ""
echo "======================================================================"
echo "  1. Non-régression — n=10/N=20 (petite instance, time_limit=-1)"
echo "======================================================================"
echo "--- PRODUCTION (gnn native sd_edit, mono-nœud) ---"
"$PROD" "$DATASET1" 0.1 30 -1 0 200 gnn 0 0.1 "$CKPT" cuda python3 native sd_edit

for K in 1 4 16; do
  echo ""
  echo "--- BATCHÉ K=$K ---"
  "$BATCHED" "$DATASET1" "$CKPT" 0.1 -1 0 cuda "$K"
done

DATASET2="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n20-m5-a50-we-r42-S100.txt"
echo ""
echo "======================================================================"
echo "  2. Le test qui compte — n=20/N=100 (GNN mono-nœud connu lent vs REF)"
echo "======================================================================"
echo "--- PRODUCTION (gnn native sd_edit, mono-nœud) ---"
"$PROD" "$DATASET2" 0.1 30 300 0 200 gnn 0 0.1 "$CKPT" cuda python3 native sd_edit

for K in 1 8 32 128; do
  echo ""
  echo "--- BATCHÉ K=$K ---"
  "$BATCHED" "$DATASET2" "$CKPT" 0.1 300 0 cuda "$K"
done

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
