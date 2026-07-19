#!/bin/bash
# bench_zeta_openmp_bab_n20_N50.sh — Teste si le fix OpenMP de
# solve_zeta_batch (gnn/bnb_solver.cpp, #pragma omp parallel for
# schedule(dynamic), jamais compilé avec -fopenmp pour les vrais exécutables
# B&B jusqu'ici — seule la sonde isolée gnn_native_probe l'avait, cf. session
# 2026-07-13 et solver_bab/CMakeLists.txt) réduit le temps réel de calcul de
# borne DANS le B&B complet (gecode BAB + GNN natif), pas seulement pour un
# appel isolé à la racine (déjà mesuré : ~41.6ms -> ~10.75ms à n_g=20,
# OMP_NUM_THREADS=4, cf. jobs 15745430/15752392 vs 15922519).
#
# Compare 3 exécutions sur LA MÊME instance (n=20/m=5/N=50, fichier existant
# ccmk-song-pisinger-n20-m5-a50-we-r42-S50.txt — même fichier que la sonde
# isolée, pour rester comparable) :
#   1. REF   (squelette nu, sans borne)                      — contexte
#   2. GNN   binaire AVANT fix (bab_mkp_gecode_noomp_baseline, ζ séquentiel)
#   3. GNN   binaire APRÈS fix (bab_mkp_gecode, ζ parallèle OpenMP)
#
# Le binaire imprime déjà "Calcul borne L1(λ) (ζ)" (temps cumulé) et "Temps
# moyen par nœud" — comparer directement ces lignes entre (2) et (3).
#
# Soumettre :
#   sbatch bench_zeta_openmp_bab_n20_N50.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=bench_omp_bab_n20N50
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n20-m5-a50-we-r42-S50.txt"
CKPT="gnn/runs/train_scaled/ckpt_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_10k_ep0197.pt"
EPS=0.10
T_ROOT=200
T_NODE=50
TIME_LIM_S=900   # par exécution ; 3 exécutions x 15 min < 1h SLURM

VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export LD_LIBRARY_PATH="$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

BIN_FIXED="$REPO/solver_bab/build/bab_mkp_gecode"
BIN_NOOMP="$REPO/solver_bab/build/bab_mkp_gecode_noomp_baseline"

echo "======================================================================"
echo "  bench_zeta_openmp_bab_n20_N50 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)   OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  Dataset=$DATASET"
echo "  eps=$EPS  T_root=$T_ROOT  T_node=$T_NODE  time_limit=${TIME_LIM_S}s (par exécution)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "=== [1/3] MODE=ref (squelette nu, sans borne) ==="
"$BIN_FIXED" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" ref

echo ""
echo "=== [2/3] MODE=gnn — binaire AVANT fix (ζ séquentiel, no -fopenmp) ==="
"$BIN_NOOMP" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" gnn non 0.1 "$CKPT" cuda python3 native

echo ""
echo "=== [3/3] MODE=gnn — binaire APRÈS fix (ζ parallèle, OMP_NUM_THREADS=$OMP_NUM_THREADS) ==="
"$BIN_FIXED" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" gnn non 0.1 "$CKPT" cuda python3 native

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
