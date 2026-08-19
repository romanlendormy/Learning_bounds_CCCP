#!/bin/bash
# run_sanity_gnn_batched_zaware.sh — Vérification de bout-en-bout du nouveau
# driver bab_mkp_gecode_gnn_batched_zaware (V1LD, exploite les z résolus) :
# doit converger vers le MÊME optimum que le mode "v1ld" exact ET que le
# gnn_batched d'ORIGINE (ignore les z) — la borne lagrangienne reste valide
# dans les deux cas (exclusion exacte, pas approximation, cf. discussion de
# session), donc les trois doivent donner un objectif identique.

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=sanity_gnn_batched_zaware
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )}"
cd "$REPO"

source solver_bab/setup_gpu_env.sh
VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"

BUILD="$REPO/solver_bab/build"
DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n10-m5-a50-we-r42-S20.txt"
CKPT="gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"

echo "======================================================================"
echo "  Sanity check — bab_mkp_gecode_gnn_batched_zaware (V1LD) vs v1ld exact"
echo "  vs gnn_batched d'origine"
echo "  Instance : $DATASET"
echo "======================================================================"

echo ""
echo "=== [1] bab_mkp_gecode <dataset> ... v1ld (EXACT, référence) ==="
"$BUILD/bab_mkp_gecode" "$DATASET" 0.1 30 -1 0 200 v1ld

echo ""
echo "=== [2] bab_mkp_gecode mode=gnn_batched (ORIGINE, ignore les z, K=8) ==="
# bab_mkp_gecode_gnn_batched (binaire autonome) a un bug CMake préexistant
# (sources AOTI manquantes) sans rapport avec ce travail — on utilise le
# binaire intégré à la place, mode="gnn_batched", déjà validé cette session.
"$BUILD/bab_mkp_gecode" "$DATASET" 0.1 30 -1 0 200 gnn_batched \
  non 0.1 "$CKPT" cpu python3 ipc sd_edit 0.0 0.0 1.0 8 1.0 - 1 native

echo ""
echo "=== [3] bab_mkp_gecode_gnn_batched_zaware (K=1, device=cpu) ==="
"$BUILD/bab_mkp_gecode_gnn_batched_zaware" "$DATASET" "$CKPT" 0.1 -1 0 cpu 1

echo ""
echo "=== [4] bab_mkp_gecode_gnn_batched_zaware (K=8, device=cpu) ==="
"$BUILD/bab_mkp_gecode_gnn_batched_zaware" "$DATASET" "$CKPT" 0.1 -1 0 cpu 8

echo ""
echo "======================================================================"
echo "  Attendu : même 'Valeur optimale' dans les 4 blocs."
echo "======================================================================"
