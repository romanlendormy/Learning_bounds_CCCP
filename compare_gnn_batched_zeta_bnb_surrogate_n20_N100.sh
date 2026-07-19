#!/bin/bash
# compare_gnn_batched_zeta_bnb_surrogate_n20_N100.sh — A/B test demandé :
# méthode ζ_i APRÈS la prédiction λ du GNN (calcul de L1(λ), mode=gnn_batched)
# en "bnb" (solve_zeta_batch, historique/prod) vs "surrogate"
# (solve_zeta_batch_surrogate, gnn/bnb_solver.cpp) — cf. doc
# GNNSdEditNativeBatched::use_surrogate_zeta (gnn_sd_edit_native_batched.hpp).
#
# Référence connue (run_check_batched_n20_N100.sh, avant ce changement) :
# REF=8.4s/99302 nœuds, batché K=128=26.0s/25399 nœuds, optimal 5221.992.
# ζ_i retourne EXACTEMENT la même valeur dans les deux méthodes (validé dans
# le B&B V1LD séquentiel équivalent : mêmes nœuds/objectif, ×2.25 plus rapide,
# cf. mémoire project_compare_v1ld_zeta_bnb_surrogate) — donc nœuds/objectif
# doivent être IDENTIQUES ici aussi entre bnb et surrogate ; seul le temps
# doit différer. Sert de sanity check : un écart de nœuds/objectif signalerait
# un bug.
#
# device=cuda requis (contrairement au test V1LD séquentiel) : le forward GNN
# tourne réellement sur GPU ici.
#
# Soumettre :
#   sbatch compare_gnn_batched_zeta_bnb_surrogate_n20_N100.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=cmp_gnnbatch_zeta_bnb_surr_n20_N100
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
K=64

echo "======================================================================"
echo "  compare_gnn_batched_zeta_bnb_surrogate_n20_N100 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Noeud : $(hostname)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "--- REF (squelette nu, sans borne) ---"
"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIMIT" 0 "$T_ROOT" ref \
  || echo "[ERREUR] REF a échoué (code $?)"

echo ""
echo "--- gnn_batched zeta_method=bnb (historique/prod, K=$K) ---"
"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIMIT" 0 "$T_ROOT" gnn_batched \
  non 0.1 "$CKPT" cuda python3 ipc sd_edit 0.0 0.0 1.0 "$K" 1.0 bnb \
  || echo "[ERREUR] gnn_batched bnb a échoué (code $?)"

echo ""
echo "--- gnn_batched zeta_method=surrogate (K=$K) ---"
"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIMIT" 0 "$T_ROOT" gnn_batched \
  non 0.1 "$CKPT" cuda python3 ipc sd_edit 0.0 0.0 1.0 "$K" 1.0 surrogate \
  || echo "[ERREUR] gnn_batched surrogate a échoué (code $?)"

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
