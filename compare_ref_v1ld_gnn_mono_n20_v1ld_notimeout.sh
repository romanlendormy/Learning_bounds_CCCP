#!/bin/bash
# compare_ref_v1ld_gnn_mono_n20_v1ld_notimeout.sh — comme
# compare_ref_v1ld_gnn_mono_n20_fastcheck.sh, mais V1LD tourne SANS timeout
# interne (time_lim <= 0 => aucune limite de temps dans solve_gecode_bab_mkp,
# cf. gecode_bab.cpp:222-223) pour obtenir un vrai OPTIMAL et donc un
# "temps total"/"temps moyen par nœud" honnêtes, pas tronqués par un budget
# arbitraire. REF et GNN gardent leur budget existant (60/120/300s) car ils
# atteignent déjà OPTIMAL largement dans cette fenêtre (cf. run précédent,
# job 16427041).
#
# ATTENTION : durée de V1LD à N=100 incertaine — au run précédent (300s cap),
# il n'avait fait que 4348/? nœuds à 69ms/nœud, gap 22.5% encore ouvert.
# Walltime SLURM généreux (6h) pour ne pas tuer le job avant convergence,
# mais N=100 pourrait quand même ne pas finir dans cette fenêtre.
#
# Soumettre :
#   sbatch compare_ref_v1ld_gnn_mono_n20_v1ld_notimeout.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=cmp_ref_v1ld_gnn_n20_notimeout
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=06:00:00
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

CKPT="gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
BINARY="$REPO/solver_bab/build/bab_mkp_gecode"
DATADIR="data/knapsack/datasets-pisinger"
EPS=0.10
T_ROOT=200
T_NODE=50

echo "======================================================================"
echo "  compare_ref_v1ld_gnn_mono_n20_v1ld_notimeout — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Noeud : $(hostname)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

declare -A TIME_LIM=( [20]=60 [50]=120 [100]=300 )

for N in 20 50 100; do
  DATASET="$DATADIR/ccmk-song-pisinger-n20-m5-a50-we-r42-S${N}.txt"
  TL="${TIME_LIM[$N]}"

  echo ""
  echo "######################################################################"
  echo "  n=20  N=$N"
  echo "######################################################################"

  echo ""
  echo "--- [n=20 N=$N] REF (squelette nu, sans borne, time_limit=${TL}s) ---"
  "$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TL" 0 "$T_ROOT" ref \
    || echo "[ERREUR] REF n=20 N=$N a échoué (code $?)"

  echo ""
  echo "--- [n=20 N=$N] V1LD (sous-gradient, SANS timeout) ---"
  "$BINARY" "$DATASET" "$EPS" "$T_NODE" -1 0 "$T_ROOT" v1ld \
    || echo "[ERREUR] V1LD n=20 N=$N a échoué (code $?)"

  echo ""
  echo "--- [n=20 N=$N] GNN mono-nœud (backend=native, variant=sd_edit, device=cuda, time_limit=${TL}s) ---"
  "$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TL" 0 "$T_ROOT" gnn non 0.1 "$CKPT" cuda python3 native sd_edit \
    || echo "[ERREUR] GNN n=20 N=$N a échoué (code $?)"
done

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
