#!/bin/bash
# check_active_x_cache_surrogate_n20_N100.sh — Confirme l'addition des deux
# optimisations validées séparément sur n=20/N=100, K=64, stride=1 :
#   - cache active_x-invariant (gnn_sd_edit_native_batched.cpp, toujours actif,
#     pas de flag) : validé bit-identique, construction graphe 5.4-5.5s→2.3s
#     (cf. job 16640950)
#   - zeta_method=surrogate (solve_zeta_batch_surrogate, gnn/bnb_solver.cpp) :
#     validé bit-identique, ζ 4.42s→1.66s (cf. job 16633631)
# Les deux touchent des parties disjointes du pipeline (construction de
# graphe vs résolution ζ) — ce run vérifie qu'elles s'additionnent bien
# plutôt que de se marcher dessus.
#
# Référence AVANT les deux (job 16633631, zeta=bnb, SANS cache) :
#   11494 nœuds, obj=5221.9914, temps borne 14.50s (graphe 5.43s / forward
#   4.66s / ζ 4.42s), temps total 17.49s.
# Avec cache SEUL (job 16640950, zeta=bnb) :
#   11494 nœuds (identique), temps borne 12.49s (graphe 2.30s / forward
#   5.82s / ζ 4.38s), temps total 15.65s.
#
# Soumettre :
#   sbatch check_active_x_cache_surrogate_n20_N100.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=check_cache_surr_n20_N100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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

BINARY="$REPO/solver_bab/build/bab_mkp_gecode"
CKPT="gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n20-m5-a50-we-r42-S100.txt"
EPS=0.10
T_ROOT=200
T_NODE=50
TIME_LIMIT=300
K=64

echo "======================================================================"
echo "  check_active_x_cache_surrogate_n20_N100 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Noeud : $(hostname)"
echo "  Réf. AVANT tout (16633631) : 11494 nœuds, borne 14.50s (graphe 5.43/fwd 4.66/zeta 4.42)"
echo "  Cache seul (16640950)      : 11494 nœuds, borne 12.49s (graphe 2.30/fwd 5.82/zeta 4.38)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "--- gnn_batched (K=$K, zeta_method=surrogate, stride=1) cache actif + surrogate ---"
"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIMIT" 0 "$T_ROOT" gnn_batched \
  non 0.1 "$CKPT" cuda python3 ipc sd_edit 0.0 0.0 1.0 "$K" 1.0 surrogate 1 \
  || echo "[ERREUR] gnn_batched a échoué (code $?)"

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
