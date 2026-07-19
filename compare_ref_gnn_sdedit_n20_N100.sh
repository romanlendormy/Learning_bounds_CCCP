#!/bin/bash
# compare_ref_gnn_sdedit_n20_N100.sh — Gecode BAB : REF vs GNN natif variant
# sd_edit (étoile scénario-scénario ancrée au scénario 0, O(n_g·N) au lieu de
# la clique O(n_g·N²) homogène) sur la MÊME instance n=20/m=5/N=100 que
# compare_ref_gnn_n20_N100.sh — pour mesurer proprement l'impact de la
# vectorisation torch des arêtes (portée depuis gnn_v1ld_native.cpp), cf.
# [[project_gnn_edges_vectorized_deployed]].
#
# Lancé UNE FOIS AVANT le portage (baseline scalaire) puis UNE FOIS APRÈS
# (avec le même binaire/LibTorch, pour éviter le confondant rencontré côté
# homogène où la migration LibTorch 2.13.0 concurrente avait pollué la
# comparaison forward).
#
# Soumettre :
#   sbatch compare_ref_gnn_sdedit_n20_N100.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=cmp_ref_gnn_sdedit_n20_N100
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n20-m5-a50-we-r42-S100.txt"
CKPT="gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
EPS=0.10
T_ROOT=200
T_NODE=50
TIME_LIM_S=900   # par méthode ; suffisant pour cette instance (cf. n20/N100 homogène ~8-110s)

source solver_bab/setup_gpu_env.sh
VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

BINARY="$REPO/solver_bab/build/bab_mkp_gecode"

echo "======================================================================"
echo "  compare_ref_gnn_sdedit_n20_N100 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)   OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  Dataset=$DATASET"
echo "  Checkpoint=$CKPT"
echo "  eps=$EPS  T_root=$T_ROOT  T_node=$T_NODE  time_limit=${TIME_LIM_S}s"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "=== [1/2] MODE=ref (squelette nu, sans borne) ==="
"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" ref

echo ""
echo "=== [2/2] MODE=gnn (backend=native, device=cuda, variant=sd_edit) ==="
"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" gnn non 0.1 "$CKPT" cuda python3 native sd_edit

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
