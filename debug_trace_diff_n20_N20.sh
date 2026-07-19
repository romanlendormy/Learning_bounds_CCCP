#!/bin/bash
# debug_trace_diff_n20_N20.sh — instrumentation nœud par nœud pour localiser
# précisément la divergence entre PRODUCTION (bab_mkp_gecode, gnn native
# sd_edit, mono-nœud) et BATCHÉ (bab_mkp_gecode_gnn_batched, K=1) sur
# l'instance n=20/N=20 qui produit un faux "OPTIMAL" (5130.397 au lieu de
# 5166.381, cf. mémoire "gnn_batched_robustness_sweep_result").
#
# Chaque appel de borne GNN, dans les deux binaires, émet maintenant sur
# stderr une ligne "TRACE,<x_mask>,<depth>,<fp>,<ub_relative>,<ub>" quand
# GNN_TRACE_DEBUG est défini (instrumentation ajoutée dans gecode_bab.hpp
# et gecode_bab_gnn_batched.cpp, opt-in, aucun effet sinon).
#
# x_mask encode l'état complet des n items (0/1/2=libre) — clé stable pour
# recouper la MÊME position dans l'arbre entre les deux exécutions, même si
# l'ordre d'exploration diffère (DFS immédiat en prod vs bucket différé en
# batché).
#
# Soumettre :
#   sbatch debug_trace_diff_n20_N20.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=debug_trace_diff_n20_N20
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
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

DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n20-m5-a50-we-r42-S20.txt"
CKPT="gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
PROD="$REPO/solver_bab/build/bab_mkp_gecode"
BATCHED="$REPO/solver_bab/build/bab_mkp_gecode_gnn_batched"

TRACE_PROD="$REPO/logs/trace_prod_n20_N20_${SLURM_JOB_ID:-local}.csv"
TRACE_BATCHED="$REPO/logs/trace_batched_n20_N20_${SLURM_JOB_ID:-local}.csv"

echo "======================================================================"
echo "  debug_trace_diff_n20_N20 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Noeud : $(hostname)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "=== PRODUCTION (gnn native sd_edit, mono-noeud), GNN_TRACE_DEBUG=1 ==="
GNN_TRACE_DEBUG=1 "$PROD" "$DATASET" 0.1 50 300 0 200 gnn non 0.1 "$CKPT" cuda python3 native sd_edit \
  2> >(grep '^TRACE,' > "$TRACE_PROD")
echo "Lignes tracées (prod) : $(wc -l < "$TRACE_PROD")"

echo ""
echo "=== BATCHÉ K=1, GNN_TRACE_DEBUG=1 ==="
GNN_TRACE_DEBUG=1 "$BATCHED" "$DATASET" "$CKPT" 0.1 300 0 cuda 1 \
  2> >(grep '^TRACE,' > "$TRACE_BATCHED")
echo "Lignes tracées (batché) : $(wc -l < "$TRACE_BATCHED")"

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "  Traces : $TRACE_PROD"
echo "           $TRACE_BATCHED"
echo "======================================================================"
