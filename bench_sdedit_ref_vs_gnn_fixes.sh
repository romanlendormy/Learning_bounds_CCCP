#!/bin/bash
# bench_sdedit_ref_vs_gnn_fixes.sh — REF vs GNN v1LD sd_edit avec les fixs
# actuellement en place :
#   - batching GPU du forward (Étape 2e, bab_mkp_gecode_gnn_batched,
#     GNNSdEditNativeBatched) — K variable, cf.
#     [[project_gnn_batched_bab_integration_result]]
#   - surrogate + parallélisation OpenMP du calcul de ζ (bnb_solver.cpp,
#     -fopenmp lié sur bab_mkp_gecode ET bab_mkp_gecode_gnn_batched) — cf.
#     [[project_gnn_bab_openmp_zeta_fix_validated]], confirmé topologie-
#     agnostique donc actif pour sd_edit sans changement, cf.
#     [[project_sdedit_native_backend_port]]
#
# PAS inclus (décision explicite, 2026-07-14) : vectorisation torch de la
# construction des arêtes — jamais portée à sd_edit (gnn_sd_edit_native*.cpp
# encore en boucles scalaires push_back), cf.
# [[project_gnn_edges_vectorized_deployed]] ("how to apply" : scopé homogène
# uniquement).
#
# Deux instances :
#   1. n=10/N=20 : contrôle rapide (non-régression, arbre trop petit pour
#      voir un effet du batching).
#   2. n=20/N=100 : LE test qui compte (in-distribution, GNN mono-nœud connu
#      12.7x plus lent que REF en wall-clock malgré ÷3.8 nœuds — cf.
#      [[project_compare_ref_gnn_n20_N100_indist_result]]). Le batching
#      corrige-t-il ce problème pour de vrai ?
#
# Pour chaque instance : REF (squelette nu) ; GNN natif mono-nœud (backend
# native, variant sd_edit — donne la décomposition graphe/forward/ζ la plus
# fine disponible pour sd_edit) ; GNN batché à K=1/32/128 (décomposition
# identique + effet du batching).
#
# Soumettre :
#   sbatch bench_sdedit_ref_vs_gnn_fixes.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=bench_sdedit_ref_vs_gnn_fixes
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:25:00
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

EPS=0.10
T_ROOT=200
T_NODE=50

echo "======================================================================"
echo "  bench_sdedit_ref_vs_gnn_fixes — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)   OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  Checkpoint sd_edit : $CKPT"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ── 1. n=10/N=20 : contrôle rapide (non-régression) ─────────────────────────
DATASET1="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n10-m5-a50-we-r42-S20.txt"
TIME_LIM1=60

echo ""
echo "======================================================================"
echo "  1. n=10/N=20 — contrôle rapide"
echo "======================================================================"

echo ""
echo "--- [1a] REF ---"
"$PROD" "$DATASET1" "$EPS" "$T_NODE" "$TIME_LIM1" 0 "$T_ROOT" ref

echo ""
echo "--- [1b] GNN natif mono-nœud (native, sd_edit) ---"
"$PROD" "$DATASET1" "$EPS" "$T_NODE" "$TIME_LIM1" 0 "$T_ROOT" gnn non 0.1 "$CKPT" cuda python3 native sd_edit

echo ""
echo "--- [1c] GNN batché K=1 ---"
"$BATCHED" "$DATASET1" "$CKPT" "$EPS" "$TIME_LIM1" 0 cuda 1

# ── 2. n=20/N=100 : le test qui compte ──────────────────────────────────────
DATASET2="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n20-m5-a50-we-r42-S100.txt"
TIME_LIM2=300

echo ""
echo "======================================================================"
echo "  2. n=20/N=100 — le test qui compte"
echo "======================================================================"

echo ""
echo "--- [2a] REF ---"
"$PROD" "$DATASET2" "$EPS" "$T_NODE" "$TIME_LIM2" 0 "$T_ROOT" ref

echo ""
echo "--- [2b] GNN natif mono-nœud (native, sd_edit) ---"
"$PROD" "$DATASET2" "$EPS" "$T_NODE" "$TIME_LIM2" 0 "$T_ROOT" gnn non 0.1 "$CKPT" cuda python3 native sd_edit

for K in 1 32 128; do
  echo ""
  echo "--- [2c] GNN batché K=$K ---"
  "$BATCHED" "$DATASET2" "$CKPT" "$EPS" "$TIME_LIM2" 0 cuda "$K"
done

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
