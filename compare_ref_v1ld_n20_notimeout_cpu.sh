#!/bin/bash
# compare_ref_v1ld_n20_notimeout_cpu.sh — comme
# compare_ref_v1ld_gnn_mono_n20_v1ld_notimeout.sh, mais SANS --gres=gpu et
# SANS le mode gnn (retiré de la boucle) : REF et V1LD n'appellent jamais de
# fonction CUDA (V1LD résout ζ via OR-Tools CP-SAT, pur CPU — cf. mémoire
# "mkp_v1ld_subproblems_bench"). Le binaire bab_mkp_gecode reste lié à
# libtorch-cuda (external/libtorch-cuda), donc il faut quand même charger les
# modules CUDA/cuDNN/cusparselt pour résoudre les dépendances partagées
# transitives de libtorch_cuda.so au chargement — mais aucun GPU physique
# n'est requis (vérifié : `source setup_gpu_env.sh` + run ref/v1ld en local
# sur un nœud sans GPU fonctionne, résultats identiques à ceux obtenus sous
# --gres=gpu:h100:1). Objectif : éviter la queue GPU (~1.5j de priorité,
# job 16450552 annulé) pour un calcul qui ne l'utilise pas.
#
# V1LD tourne SANS timeout interne (time_lim <= 0 => aucune limite de temps
# dans solve_gecode_bab_mkp, cf. gecode_bab.cpp:222-223) pour obtenir un vrai
# OPTIMAL. REF garde son budget existant (60/120/300s) car il atteint déjà
# OPTIMAL largement dans cette fenêtre (cf. job 16427041).
#
# ATTENTION : durée de V1LD à N=100 incertaine — au run précédent (300s cap),
# il n'avait fait que 4348/? nœuds à 69ms/nœud, gap 22.5% encore ouvert.
# Walltime généreux (6h) pour ne pas tuer le job avant convergence, mais
# N=100 pourrait quand même ne pas finir dans cette fenêtre.
#
# Soumettre :
#   sbatch compare_ref_v1ld_n20_notimeout_cpu.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=cmp_ref_v1ld_n20_notimeout_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
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

BINARY="$REPO/solver_bab/build/bab_mkp_gecode"
DATADIR="data/knapsack/datasets-pisinger"
EPS=0.10
T_ROOT=200
T_NODE=50

echo "======================================================================"
echo "  compare_ref_v1ld_n20_notimeout_cpu — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Noeud : $(hostname)"
echo "======================================================================"

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
done

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
