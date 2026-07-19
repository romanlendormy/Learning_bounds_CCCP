#!/bin/bash
# compare_ref_gnn_n20_N1000_ref.sh — Gecode BAB : REF (squelette nu, sans
# borne) SEUL sur l'instance n=20/m=5/N=1000 (générée le 2026-07-17, seed=42
# — ccmk-song-pisinger-n20-m5-a50-we-r42-S1000.txt), sans AUCUN timeout
# algorithmique interne (time_limit=0 → within_time_budget toujours vrai,
# cf. gecode_bab.cpp:226 : `if (time_lim <= 0.0) return true;`).
#
# Job séparé de compare_ref_gnn_n20_N1000_gnn.sh (plutôt qu'un seul job
# séquentiel REF puis gnn_batched) : sans timeout interne, si REF prenait
# beaucoup de temps il pourrait affamer gnn_batched du temps SLURM restant
# dans un job unique. Deux jobs indépendants, chacun avec son propre plafond
# SLURM (24h, seul filet de sécurité restant, demandé par l'utilisateur).
#
# REF n'utilise aucune borne GNN/V1LD → pas besoin de GPU, seulement CPU.
# Attendu rapide (n=20/N=500 : 62s, 70183 nœuds — N=1000 ne devrait pas
# dépasser quelques minutes, le coût par nœud croît avec N mais le nombre de
# nœuds dépend surtout de n, fixe ici).
#
# Soumettre :
#   sbatch compare_ref_gnn_n20_N1000_ref.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=cmp_ref_n20_N1000
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n20-m5-a50-we-r42-S1000.txt"
EPS=0.10
T_ROOT=200
T_NODE=50
TIME_LIM_S=0   # 0 = pas de timeout algorithmique interne (cf. en-tête)

VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export LD_LIBRARY_PATH="$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

BINARY="$REPO/solver_bab/build/bab_mkp_gecode"

echo "======================================================================"
echo "  compare_ref_gnn_n20_N1000_ref — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  Dataset=$DATASET"
echo "  eps=$EPS  T_root=$T_ROOT  T_node=$T_NODE  time_limit=illimité  mur SLURM=24h"
echo "======================================================================"

echo ""
echo "=== MODE=ref (squelette nu, sans borne) ==="
"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" ref

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
