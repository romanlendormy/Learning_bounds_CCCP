#!/bin/bash
# run_test_ssp.sh — Compile et exécute run_ssp_tests() sur un nœud CPU.
#
# Soumettre :
#   sbatch run_test_ssp.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=test_ssp
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:10:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

PROJECT=/project/6062881/romanlen/Learning_bounds_CCCP
GECODE=${PROJECT}/external/gecode

echo "=============================="
echo "  test_ssp — $(date)"
echo "  Job : ${SLURM_JOB_ID}   Nœud : $(hostname)"
echo "=============================="

# ── Compilation ───────────────────────────────────────────────────────────────
BUILD=${SLURM_TMPDIR}/test_ssp_bin
echo "Compilation dans ${SLURM_TMPDIR} ..."

g++ -std=c++17 -O2 \
    -I "${PROJECT}/gecode_base" \
    -I "${PROJECT}/solver_generic" \
    -I "${GECODE}/include" \
    -o "${BUILD}" \
    "${PROJECT}/gecode_base/gecode_solver.cpp" \
    "${PROJECT}/gecode_base/test_ssp.cpp" \
    -L "${GECODE}/lib" \
    -Wl,-rpath,"${GECODE}/lib" \
    -lgecodeint -lgecodekernel -lgecodesupport -lgecodesearch -lgecodeminimodel

echo "Compilation OK — $(date)"

# ── Exécution ─────────────────────────────────────────────────────────────────
echo ""
"${BUILD}"
STATUS=$?

echo ""
echo "=============================="
if [ ${STATUS} -eq 0 ]; then
    echo "  run_ssp_tests : TOUS OK"
else
    echo "  run_ssp_tests : ECHEC (exit ${STATUS})"
fi
echo "  Fin : $(date)"
echo "=============================="

exit ${STATUS}
