#!/bin/bash
# build_bab_mkp_gecode_v1cld_mode.sh — Rebuild de bab_mkp_gecode après ajout
# du mode CLI "gnn_batched_v1cld" (main_gecode.cpp) + des sources
# gecode_bab_v1cld_gnn_batched.cpp / gnn_v1cld_sd_edit_native_batched.cpp à la
# cible existante (pas de nouvelle cible ici, donc pas de piège gmake, mais
# reconfigure explicite quand même par prudence — cf. build_test_gnn_v1cld.sh).

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=build_bab_mkp_gecode_v1cld_mode
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )}"
cd "$REPO"

module load StdEnv/2023
module load gcc/12.3
module load cmake/3.27.7

BAB_BUILD="$REPO/solver_bab/build"

echo "======================================================================"
echo "  build_bab_mkp_gecode_v1cld_mode — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "======================================================================"

cmake -S "$REPO/solver_bab" -B "$BAB_BUILD" 2>&1 | tail -10
cmake --build "$BAB_BUILD" --target bab_mkp_gecode -j4 2>&1

[[ -x "$BAB_BUILD/bab_mkp_gecode" ]] || { echo "ERREUR : binaire non généré"; exit 1; }

echo ""
echo "[OK] $BAB_BUILD/bab_mkp_gecode"
echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
