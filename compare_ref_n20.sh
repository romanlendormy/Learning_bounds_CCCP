#!/bin/bash
# compare_ref_n20.sh — Gecode BAB en mode REF (squelette nu, sans borne
# Lagrangienne — seule la propagation CP + l'élagage par incumbent agissent)
# sur l'instance n=20/m=5/N=50, pour mesurer la valeur ajoutée de la borne
# (GNN/V1LD) par rapport à un B&B non guidé. Cf. session "load train
# checkpoint into GECODE BAB".
#
# Soumettre :
#   sbatch compare_ref_n20.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=cmp_ref_n20
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n20-m5-a50-we-r42-S50.txt"
EPS=0.10
TIME_LIM_S=3300   # même budget que gnn/v1ld pour comparaison directe

export LD_LIBRARY_PATH="$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"
BINARY="$REPO/solver_bab/build/bab_mkp_gecode"

echo "======================================================================"
echo "  compare_ref_n20 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  Dataset=$DATASET  eps=$EPS  time_limit=${TIME_LIM_S}s  mode=ref"
echo "======================================================================"

echo ""
echo "=== MODE=ref (squelette nu, sans borne) ==="
"$BINARY" "$DATASET" "$EPS" 50 "$TIME_LIM_S" 0 200 ref

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
