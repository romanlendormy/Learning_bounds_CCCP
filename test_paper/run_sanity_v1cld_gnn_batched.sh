#!/bin/bash
# run_sanity_v1cld_gnn_batched.sh — Vérification de bout-en-bout du nouveau
# driver bab_mkp_gecode_v1cld_gnn_batched : sur une petite instance, la borne
# lagrangienne reste valide pour TOUT lambda (dualité faible), donc peu
# importe la qualité du GNN, le B&B GNN-batché doit converger vers le MÊME
# optimum que le mode "v1cld" exact (ζ DP 1D exact, déjà validé en
# production) de bab_mkp_gecode — pas seulement "feasible", mais bien
# l'objectif optimal identique (à 1e-4 près, SCALE=10000).

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=sanity_v1cld_gnn_batched
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )}"
cd "$REPO"

source solver_bab/setup_gpu_env.sh

BUILD="$REPO/solver_bab/build"
DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n10-m5-a50-we-r42-S20.txt"
CKPT="gnn_sd_edit/runs_v1cld/ckpt_v1cld_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_v1cld_n10_20_500ep_ep0500.pt"

echo "======================================================================"
echo "  Sanity check — bab_mkp_gecode_v1cld_gnn_batched vs v1cld exact"
echo "  Instance : $DATASET"
echo "======================================================================"

echo ""
echo "=== [1] bab_mkp_gecode <dataset> ... v1cld (EXACT, référence) ==="
"$BUILD/bab_mkp_gecode" "$DATASET" 0.1 30 -1 0 200 v1cld

echo ""
echo "=== [2] bab_mkp_gecode_v1cld_gnn_batched (GNN, K=1, device=cpu) ==="
"$BUILD/bab_mkp_gecode_v1cld_gnn_batched" "$DATASET" "$CKPT" 0.1 -1 0 cpu 1

echo ""
echo "=== [3] bab_mkp_gecode_v1cld_gnn_batched (GNN, K=8, device=cpu) ==="
"$BUILD/bab_mkp_gecode_v1cld_gnn_batched" "$DATASET" "$CKPT" 0.1 -1 0 cpu 8

echo ""
echo "======================================================================"
echo "  Attendu : même 'Valeur optimale' dans les 3 blocs, statut OPTIMAL"
echo "  pour [1] et statut OPTIMAL/FEASIBLE avec objectif IDENTIQUE pour"
echo "  [2]/[3] (la borne lagrangienne reste valide pour tout lambda)."
echo "======================================================================"
