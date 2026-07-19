#!/bin/bash
# bench_gnn_batched_robustness_sweep.sh — robustesse du B&B GNN batché
# (bab_mkp_gecode_gnn_batched) au-delà des 2 instances déjà validées
# (n=10/N=20 non-régression, n=20/N=100 gain x2.7, cf.
# project_gnn_batched_bab_integration_result). Objectif : voir si le gain
# généralise sur d'autres n/N in-distribution (n in {10,20}, N in
# {20,50,100}) ET sur le cas OOD connu cassé (n=20 et n=10, N=500 — cf.
# project_compare_ref_gnn_n20_N500, GNN mono-noeud jamais résolu).
#
# Pour chaque dataset : REF (squelette nu) + PROD (gnn native sd_edit,
# mono-noeud) + BATCHED à K in {1, 32, 128}.
#
# Pas de `set -e` : une instance qui échoue (timeout, OOM, crash Gecode)
# ne doit pas faire perdre les résultats des 7 autres.
#
# Soumettre :
#   sbatch bench_gnn_batched_robustness_sweep.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=gnn_batched_robustness
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:30:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -uo pipefail   # PAS de -e : un run qui échoue ne doit pas tuer le sweep

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
DATADIR="data/knapsack/datasets-pisinger"
EPS=0.10
T_ROOT=200
T_NODE=50

echo "======================================================================"
echo "  bench_gnn_batched_robustness_sweep — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Noeud : $(hostname)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

run_one() {
  local n="$1" N="$2" time_limit="$3"
  local dataset="$DATADIR/ccmk-song-pisinger-n${n}-m5-a50-we-r42-S${N}.txt"

  echo ""
  echo "######################################################################"
  echo "  n=$n  N=$N  time_limit=${time_limit}s"
  echo "######################################################################"

  echo "--- REF (squelette nu) ---"
  "$PROD" "$dataset" "$EPS" "$T_NODE" "$time_limit" 0 "$T_ROOT" ref \
    || echo "[ERREUR] REF n=$n N=$N a échoué (code $?)"

  echo ""
  echo "--- PRODUCTION (gnn native sd_edit, mono-noeud) ---"
  "$PROD" "$dataset" "$EPS" "$T_NODE" "$time_limit" 0 "$T_ROOT" gnn non 0.1 "$CKPT" cuda python3 native sd_edit \
    || echo "[ERREUR] PROD n=$n N=$N a échoué (code $?)"

  for K in 1 32 128; do
    echo ""
    echo "--- BATCHÉ K=$K ---"
    "$BATCHED" "$dataset" "$CKPT" "$EPS" "$time_limit" 0 cuda "$K" \
      || echo "[ERREUR] BATCHED K=$K n=$n N=$N a échoué (code $?)"
  done
}

# --- in-distribution : n in {10,20} x N in {20,50,100} ---
run_one 10 20  60
run_one 10 50  120
run_one 10 100 300
run_one 20 20  60
run_one 20 50  120
run_one 20 100 300

# --- OOD connu (N=500, hors distribution d'entraînement N in {20,50,100}) ---
run_one 10 500 180
run_one 20 500 180

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
