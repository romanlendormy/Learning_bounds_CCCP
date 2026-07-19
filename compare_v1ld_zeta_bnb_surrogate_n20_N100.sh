#!/bin/bash
# compare_v1ld_zeta_bnb_surrogate_n20_N100.sh — A/B test demandé : méthode ζ_i
# du VRAI B&B V1LD (mode=v1ld) en "bnb" (MKPProblemBnB, historique/prod) vs
# "surrogate" (MKPProblemBnBSurrogate, borne de branchement surrogate — cf.
# mkp_surrogate_bound.hpp), comparés à REF (squelette nu, sans borne) sur
# n=20/N=100. cf. doc MKPSharedData::use_surrogate_zeta (gecode_bab.hpp).
#
# ζ_i retourne EXACTEMENT la même valeur optimale dans les deux méthodes
# (seule la preuve d'optimalité de la borne de branchement interne diffère)
# => même trajectoire λ, même arbre Gecode (mêmes nœuds/objectif) dans les
# deux cas v1ld — seul le TEMPS doit différer. Sert de sanity check : si les
# nœuds/objectif diffèrent entre bnb et surrogate, il y a un bug.
#
# time_lim=-1 (comme compare_ref_v1ld_n20_notimeout_cpu.sh, job 16472864) :
# aucune limite interne pour obtenir un vrai OPTIMAL et un temps comparable
# honnête. Référence connue (même dataset, mode bnb) : 18398 nœuds, 767.995s,
# optimal 5221.9914 (cf. mémoire project_compare_ref_v1ld_n20_N20_50_100_result).
#
# CPU-only (pas de --gres=gpu) : REF/V1LD n'appellent jamais de fonction CUDA
# (cf. mémoire project_mkp_v1ld_subproblems_bench) ; le binaire bab_mkp_gecode
# reste lié à libtorch-cuda, donc setup_gpu_env.sh reste nécessaire pour
# résoudre les dépendances partagées transitives au chargement (pas de GPU
# physique requis).
#
# Soumettre :
#   sbatch compare_v1ld_zeta_bnb_surrogate_n20_N100.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=cmp_zeta_bnb_surr_n20_N100
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
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
DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n20-m5-a50-we-r42-S100.txt"
EPS=0.10
T_ROOT=200
T_NODE=50
REF_TIME_LIM=300

echo "======================================================================"
echo "  compare_v1ld_zeta_bnb_surrogate_n20_N100 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Noeud : $(hostname)"
echo "======================================================================"

echo ""
echo "--- REF (squelette nu, sans borne, time_limit=${REF_TIME_LIM}s) ---"
"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$REF_TIME_LIM" 0 "$T_ROOT" ref \
  || echo "[ERREUR] REF a échoué (code $?)"

echo ""
echo "--- V1LD zeta_method=bnb (historique/prod, SANS timeout) ---"
"$BINARY" "$DATASET" "$EPS" "$T_NODE" -1 0 "$T_ROOT" v1ld \
  non 0.1 "" cpu python3 ipc homogene 0.0 0.0 1.0 64 1.0 bnb \
  || echo "[ERREUR] V1LD bnb a échoué (code $?)"

echo ""
echo "--- V1LD zeta_method=surrogate (SANS timeout) ---"
"$BINARY" "$DATASET" "$EPS" "$T_NODE" -1 0 "$T_ROOT" v1ld \
  non 0.1 "" cpu python3 ipc homogene 0.0 0.0 1.0 64 1.0 surrogate \
  || echo "[ERREUR] V1LD surrogate a échoué (code $?)"

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
