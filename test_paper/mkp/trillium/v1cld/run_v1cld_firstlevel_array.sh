#!/bin/bash
# run_v1cld_firstlevel_array.sh — mesure UNIQUEMENT le "gap à la racine" V1CLD
# (stop_after_first_level=1, cf. gecode_bab_v1cld_gnn_batched.cpp/2026-08-24) :
# arrête le B&B dès que la borne au 1er niveau (enfants directs de la racine)
# est capturée, sans attendre la résolution complète — temps ~constant,
# indépendant de n/N. NE PAS confondre avec run_v1cld_array.sh (résolution
# complète, résultats dans test_paper/mkp/trillium/v1cld/results/) : celui-ci
# écrit dans un dossier scratch séparé, jamais dans results/.
#
# "Valeur optimale" de CE run est le meilleur incumbent trouvé avant l'arrêt
# (PAS l'optimum) — comparer "Borne GNN 1er niveau" à l'optimum déjà connu
# via ref (test_paper/mkp/trillium/v1ld/results/ref_*.csv), pas à ce champ.

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/def-qcappart/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=test_paper_trillium_v1cld_firstlevel
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/romanlen/test_paper_trillium_v1cld_firstlevel/slurm_logs/slurm_%x_%A_%a.out
#SBATCH --error=/scratch/romanlen/test_paper_trillium_v1cld_firstlevel/slurm_logs/slurm_%x_%A_%a.err

set -euo pipefail

REPO="/project/def-qcappart/romanlen/Learning_bounds_CCCP"
cd "$REPO"

module load StdEnv/2023 gcc/12.3
source solver_bab/setup_gpu_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

TESTDIR="$REPO/test_paper/mkp/trillium/v1cld"
SCRATCH_OUT="/scratch/romanlen/test_paper_trillium_v1cld_firstlevel"
mkdir -p "$SCRATCH_OUT/logs" "$SCRATCH_OUT/results" "$SCRATCH_OUT/slurm_logs"

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TESTDIR/instances_list.txt")
read -r N_ITEMS N_SCEN M SEED RELPATH <<< "$LINE"
DATASET="$TESTDIR/$RELPATH"

if [[ ! -f "$DATASET" ]]; then
    echo "[ERREUR] instance introuvable : $DATASET" >&2
    exit 1
fi

BINARY="$REPO/solver_bab/build/bab_mkp_gecode_v1cld_gnn_batched"
CKPT="gnn_sd_edit/runs_v1cld/ckpt_v1cld_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_v1cld_n10_20_500ep_ep0500.pt"
EPS=0.10
REP=0
TIME_LIM_S=0
K=64

LOGFILE="$SCRATCH_OUT/logs/v1cld_n${N_ITEMS}_N${N_SCEN}_seed${SEED}_firstlevel.log"
RESULT_CSV="$SCRATCH_OUT/results/v1cld_n${N_ITEMS}_N${N_SCEN}_seed${SEED}_firstlevel.csv"

{
echo "======================================================================"
echo "  test_paper_trillium_v1cld FIRST-LEVEL ONLY — n=$N_ITEMS N=$N_SCEN m=$M seed=$SEED — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}   Nœud : $(hostname)"
echo "  K=$K  stop_after_first_level=1"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
} | tee "$LOGFILE"

"$BINARY" "$DATASET" "$CKPT" "$EPS" "$TIME_LIM_S" "$REP" cuda "$K" 1 \
  2>&1 | tee -a "$LOGFILE"

python3 "$TESTDIR/parse_result.py" v1cld_gnn_batched "$N_ITEMS" "$N_SCEN" "$M" "$SEED" "$LOGFILE" > "$RESULT_CSV"

echo "" | tee -a "$LOGFILE"
echo "Terminé — $(date)" | tee -a "$LOGFILE"
