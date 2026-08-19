#!/bin/bash
# run_bench_v1cld.sh — Lance bench_v1cld_subproblems sur le manifest existant
# (50 instances Pisinger, n∈{10,20,25}). CPU seul. Mémoire large par
# précaution : b_max~3220 sur la plus petite instance (n=10/N=20) donne déjà
# une table DP projetée à ~2.8 Go (SCALE=10000) — les instances n=20/n=25 du
# manifest ont probablement des capacités du même ordre de grandeur, donc pas
# d'explosion supplémentaire attendue liée à n, mais à surveiller quand même.
#
# Soumettre :
#   sbatch test_paper/run_bench_v1cld.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=bench_v1cld
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )}"
cd "$REPO"

echo "======================================================================"
echo "  run_bench_v1cld — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "======================================================================"

./solver_bab/build/bench_v1cld_subproblems test_paper/manifest_v1cld_n20_n25.txt 0.10 --skip-dp

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
