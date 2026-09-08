#!/bin/bash
# scratch/check_ssp_stochastic_tests.sh — Lance pytest sur ssp_stochastic/tests/
# apres retrait de generator.py, calibrate.py, groundtruth.py, loader.py,
# config_default.yaml, sweep_bound_efficacy.py, sweep_difficulty.py,
# test_loader.py, test_groundtruth.py (modules pre-adaptatifs jamais
# reconstruits, zero consommateur dans la chaine prod/generation GNN).
#
# Usage : sbatch scratch/check_ssp_stochastic_tests.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=check_ssp_stochastic
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0:10:00
#SBATCH --output=scratch/logs/check_ssp_stochastic_%j.out
#SBATCH --error=scratch/logs/check_ssp_stochastic_%j.err

set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )}"
cd "$REPO"
mkdir -p scratch/logs
module load StdEnv/2023 python/3.11 scipy-stack
source "$HOME/envs/learning_bounds/bin/activate"

echo "--- pytest ssp_stochastic/tests/ -v ---"
python -m pytest ssp_stochastic/tests/ -v 2>&1 | tail -80
echo
echo "exit code: ${PIPESTATUS[0]}"
