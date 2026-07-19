#!/bin/bash
# compare_ref_gnn_n20_N1000_gnn.sh — Gecode BAB : GNN batché (mode=gnn_batched,
# checkpoint sd_edit ep0500, backend native, K=64, device=cuda) SEUL sur
# l'instance n=20/m=5/N=1000 (générée le 2026-07-17, seed=42 —
# ccmk-song-pisinger-n20-m5-a50-we-r42-S1000.txt), sans AUCUN timeout
# algorithmique interne (time_limit=0 → within_time_budget toujours vrai,
# cf. gecode_bab_gnn_batched.cpp:122 : `if (time_lim <= 0.0) return true;`).
#
# Job séparé de compare_ref_gnn_n20_N1000_ref.sh — cf. en-tête de ce dernier
# pour la raison (éviter qu'une méthode sans timeout n'affame l'autre du
# temps SLURM restant dans un job séquentiel unique).
#
# Choix gnn_batched (pas le mode "gnn" mono-nœud) : le mode gnn mono-nœud
# avait OOM-killed à N=500 en device=cpu/4G (cf.
# [[project_compare_ref_gnn_n20_N500]]). gnn_batched est le chemin de
# production recommandé depuis sa fusion dans bab_mkp_gecode (cf.
# [[project_gnn_batched_merged_into_prod]]), variante sd_edit (arêtes
# scénario = étoile, O(N)) plutôt qu'homogène (clique, O(N²)) — mieux
# dimensionné pour N=1000.
#
# ATTENTION : N=1000 jamais testé (2x le N=500 déjà OOD sévère, cf.
# [[project_gnn_batched_robustness_sweep_result]]). Sans timeout interne, si
# le problème est intraitable à cette échelle le job consommera les 24h
# pleines sans converger — c'est le plafond de sécurité demandé par
# l'utilisateur, pas une garantie de convergence.
#
# Soumettre :
#   sbatch compare_ref_gnn_n20_N1000_gnn.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=cmp_gnn_n20_N1000
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n20-m5-a50-we-r42-S1000.txt"
CKPT="gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
EPS=0.10
T_ROOT=200
T_NODE=50
TIME_LIM_S=0   # 0 = pas de timeout algorithmique interne (cf. en-tête)
K=64           # défaut prod (argv 18) — cf. [[project_gnn_batched_merged_into_prod]]

source solver_bab/setup_gpu_env.sh
VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

BINARY="$REPO/solver_bab/build/bab_mkp_gecode"

echo "======================================================================"
echo "  compare_ref_gnn_n20_N1000_gnn — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  Dataset=$DATASET"
echo "  Checkpoint=$CKPT"
echo "  eps=$EPS  T_root=$T_ROOT  T_node=$T_NODE  time_limit=illimité  K=$K  mur SLURM=24h"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "=== MODE=gnn_batched (checkpoint sd_edit, backend native, device=cuda, K=$K) ==="
"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" gnn_batched non 0.1 \
  "$CKPT" cuda python3 native sd_edit 0.0 0.0 1.0 "$K"

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
