#!/bin/bash
# compare_ref_gnn_n20_N100.sh — Gecode BAB : REF (squelette nu, sans borne)
# vs GNN natif (checkpoint homogène ep0200, backend native, device=cuda) sur
# l'instance n=20/m=5/N=100 (ccmk-song-pisinger-n20-m5-a50-we-r42-S100.txt).
# n=20 ET N=100 sont tous les deux DANS la distribution d'entraînement
# (n∈{10,20}, N∈{20,50,100}, cf. [[project_train_scaled_data_selection]]) —
# comparaison in-distribution propre, pas de facteur OOD à isoler.
#
# Reflète les derniers changements (non commités au moment du lancement,
# solver_bab/CMakeLists.txt + mkp/gecode_bab.{cpp,hpp} + mkp/main_gecode.cpp) :
#   - fix OpenMP sur solve_zeta_batch (gnn/bnb_solver.cpp) réellement linké
#     sur bab_mkp_gecode (pas seulement la sonde isolée) — parallélise le
#     calcul de ζ (tri surrogate + B&B par dimension) sur les N scénarios,
#     cf. [[project_gnn_bab_openmp_zeta_fix_validated]] (×2.94 sur ζ mesuré
#     à n=20/N=50 ; N=100 ici pour amplifier l'effet — 2x plus de scénarios
#     indépendants à paralléliser par nœud).
#   - backend GNN natif (gnn_v1ld_native.cpp + bnb_solver.cpp) désormais
#     linké DIRECTEMENT dans bab_mkp_gecode (plus besoin d'un exécutable à
#     part pour ce chemin).
#
# Binaire recompilé via compile_cc_job.sh (WITH_CUDA=1) avant ce job — cf.
# dépendance --dependency=afterok à la soumission.
#
# Soumettre (après recompilation) :
#   sbatch compare_ref_gnn_n20_N100.sh
# Ou en chaîne directe avec la recompilation :
#   sbatch --dependency=afterok:<job_compile> compare_ref_gnn_n20_N100.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=cmp_ref_gnn_n20_N100
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n20-m5-a50-we-r42-S100.txt"
CKPT="gnn/runs/train_scaled/ckpt_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_10k_ep0200.pt"
EPS=0.10
T_ROOT=200
T_NODE=50
TIME_LIM_S=3300   # par méthode ; 2 méthodes x 55 min < 2h SLURM

source solver_bab/setup_gpu_env.sh
VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

BINARY="$REPO/solver_bab/build/bab_mkp_gecode"

echo "======================================================================"
echo "  compare_ref_gnn_n20_N100 — $(date)"
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
echo "=== [2/2] MODE=gnn (backend=native, device=cuda, checkpoint homogène ep0200) ==="
"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" gnn non 0.1 "$CKPT" cuda python3 native

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
