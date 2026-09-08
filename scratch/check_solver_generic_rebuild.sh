#!/bin/bash
# scratch/check_solver_generic_rebuild.sh — Verifie que solver_generic/CMakeLists.txt
# configure ET compile reellement apres le nettoyage (retrait de covset/, kp/,
# mkp_problem_bnb_lp, ssp_cld2_problem, tests/, bindings.cpp/pybind11).
#
# check_cmake_configure.sh ne verifie QUE solver_bab, qui linke contre
# libv1ld_generic_core.a en IMPORTED STATIC -- un .a perime construit AVANT
# la suppression passerait ce check sans jamais recompiler solver_generic
# lui-meme (piege deja identifie : project_solver_generic_stale_build_trap).
# Ce script rebuild reellement v1ld_generic_core + les 2 tests unitaires
# generiques gardes (test_ssp_shift, test_ssp_bnb) et les EXECUTE.
#
# Usage : sbatch scratch/check_solver_generic_rebuild.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=check_solver_generic
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0:15:00
#SBATCH --output=scratch/logs/check_solver_generic_%j.out
#SBATCH --error=scratch/logs/check_solver_generic_%j.err

set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )}"
cd "$REPO"
mkdir -p scratch/logs
module load StdEnv/2023 gcc/12.3 cmake/3.27.7

BUILD="solver_generic/build_cc"

echo "--- reconfigure (touch CMakeLists.txt + cmake --build force le regen) ---"
touch solver_generic/CMakeLists.txt

echo
echo "--- build v1ld_generic_core (bibliotheque statique nettoyee) ---"
cmake --build "$BUILD" --target v1ld_generic_core -j2 2>&1 | tail -60
rc1=${PIPESTATUS[0]}
echo "exit code v1ld_generic_core: $rc1"

echo
echo "--- build test_ssp_shift ---"
cmake --build "$BUILD" --target test_ssp_shift -j2 2>&1 | tail -40
rc2=${PIPESTATUS[0]}
echo "exit code test_ssp_shift: $rc2"

echo
echo "--- build test_ssp_bnb ---"
cmake --build "$BUILD" --target test_ssp_bnb -j2 2>&1 | tail -40
rc3=${PIPESTATUS[0]}
echo "exit code test_ssp_bnb: $rc3"

echo
echo "======================================================================"
echo "--- run test_ssp_shift ---"
"$BUILD/test_ssp_shift"
rc4=$?
echo "exit code (run) test_ssp_shift: $rc4"

echo
echo "--- run test_ssp_bnb ---"
"$BUILD/test_ssp_bnb"
rc5=$?
echo "exit code (run) test_ssp_bnb: $rc5"

echo
echo "======================================================================"
if [[ $rc1 -eq 0 && $rc2 -eq 0 && $rc3 -eq 0 && $rc4 -eq 0 && $rc5 -eq 0 ]]; then
    echo "TOUT PASS : solver_generic reconfigure/recompile/s'execute correctement apres nettoyage."
else
    echo "ECHEC : rc1=$rc1 rc2=$rc2 rc3=$rc3 rc4=$rc4 rc5=$rc5"
fi
