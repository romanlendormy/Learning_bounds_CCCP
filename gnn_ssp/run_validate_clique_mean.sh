#!/bin/bash
# gnn_ssp/run_validate_clique_mean.sh — Lance validate_clique_mean_equivalence.py
# (petit test CPU, mais libgomp refuse de créer des threads sur le login node
# -- même un test minuscule doit passer par sbatch sur ce cluster).
#
# Soumission :
#   sbatch gnn_ssp/run_validate_clique_mean.sh

#SBATCH --account=def-qcappart_cpu
#SBATCH --partition=cpubase_bycore_b4
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=validate_clique_mean
#SBATCH --output=logs/validate_clique_mean_%j.out
#SBATCH --error=logs/validate_clique_mean_%j.err
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
module load StdEnv/2023
module load python/3.11
source "$HOME/envs/learning_bounds/bin/activate"
export OMP_NUM_THREADS=2

python3 "$REPO/gnn_ssp/validate_clique_mean_equivalence.py"
