#!/bin/bash
# scratch/check_cmake_configure.sh — Verifie juste que cmake --build (etape
# configure/generate) reussit sur solver_bab/build apres suppression des
# cibles de test B1/B2/etape2 SSP -- CPU seul, pas de build reel necessaire
# pour valider que plus rien ne reference un fichier manquant.
#
# Usage : sbatch scratch/check_cmake_configure.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=check_cmake
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0:10:00
#SBATCH --output=scratch/logs/check_cmake_%j.out
#SBATCH --error=scratch/logs/check_cmake_%j.err

set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )}"
cd "$REPO"
mkdir -p scratch/logs
module load StdEnv/2023 gcc/12.3 cmake/3.27.7

echo "--- cmake --build (juste le check_build_system / configure) ---"
touch solver_bab/CMakeLists.txt
cmake --build solver_bab/build --target bab_mkp_gecode -j1 2>&1 | tail -30
echo
echo "exit code: $?"
