#!/bin/bash
# scratch/sanity_check_ref_mode.sh — Complète sanity_check_cleanup.sh : vérifie
# que le mode "ref" (squelette B&B nu, sans borne GNN — la référence utilisée
# dans test_paper/results.csv et test_paper_trillium/results_ssp.csv) tourne
# toujours après le nettoyage. Binaires déjà validés par sanity_check_cleanup.sh
# (pas de rebuild ici).
#
# Usage :
#   sbatch scratch/sanity_check_ref_mode.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=sanity_ref
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=0:20:00
#SBATCH --output=scratch/logs/sanity_ref_%j.out
#SBATCH --error=scratch/logs/sanity_ref_%j.err

set -uo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )}"
cd "$REPO"
mkdir -p scratch/logs

module load StdEnv/2023 gcc/12.3 cmake/3.27.7
source solver_bab/setup_gpu_env.sh

declare -A RESULT
section() { echo; echo "======================================================================"; echo "  $1"; echo "======================================================================"; }

# ── 1. B&B ref MKP minimal ───────────────────────────────────────────────────
section "1/2 B&B ref MKP minimal (n=15,N=100,seed=43, T_root=20,T_node=10, 60s)"
LOG1="scratch/logs/check_ref_mkp.log"
timeout 90 solver_bab/build/bab_mkp_gecode \
    test_paper/instance_n15_m5_N100_seed43.txt 0.10 10 60 0 20 ref \
    > "$LOG1" 2>&1
cat "$LOG1"
if grep -qiE "OPTIMAL|TIMEOUT|INFEASIBLE" "$LOG1" && ! grep -qiE "segmentation fault|terminate called|traceback|core dumped|std::runtime_error|bad_alloc" "$LOG1"; then
    RESULT[1_ref_mkp]="PASS"
else
    RESULT[1_ref_mkp]="FAIL"
fi

# ── 2. B&B ref SSP minimal ───────────────────────────────────────────────────
section "2/2 B&B ref SSP minimal (E4_T8_N50/rep0, 60s)"
LOG2="scratch/logs/check_ref_ssp.log"
timeout 90 solver_bab/build_gpu_ssp/test_gnn_v1cld_bab \
    test_paper_trillium/ssp/E4_T8_N50/rep0 ignored_ckpt 0.0 60 ref \
    > "$LOG2" 2>&1
cat "$LOG2"
if grep -qiE "status=|OPTIMAL|TIMEOUT|INFEASIBLE" "$LOG2" && ! grep -qiE "segmentation fault|terminate called|traceback|core dumped|std::runtime_error|bad_alloc" "$LOG2"; then
    RESULT[2_ref_ssp]="PASS"
else
    RESULT[2_ref_ssp]="FAIL"
fi

# ── Résumé ────────────────────────────────────────────────────────────────────
section "RÉSUMÉ"
for k in 1_ref_mkp 2_ref_ssp; do
    printf "  %-12s %s\n" "$k" "${RESULT[$k]:-?}"
done
