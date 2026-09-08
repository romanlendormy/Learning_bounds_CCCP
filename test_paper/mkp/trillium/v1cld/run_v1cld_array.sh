#!/bin/bash
# run_v1cld_array.sh (Trillium) — mode v1cld_gnn_batched (borne V1CLD batchée,
# K=64, backend native, checkpoint dédié runs_v1cld) sur les 120 instances de
# test_paper/mkp/trillium/v1cld/, une tâche d'array par instance
# (instances_list.txt, format : n N m seed path).
#
# Binaire DIFFÉRENT de bab_mkp_gecode : bab_mkp_gecode_v1cld_gnn_batched, une
# cible séparée (cf. solver_bab/mkp/main_bab_v1cld_gnn_batched.cpp), avec sa
# propre signature CLI :
#   bab_mkp_gecode_v1cld_gnn_batched <dataset> <ckpt> [eps=0.1] [time_limit=-1]
#     [rep=0] [device=cuda] [K=1]
# (pas de T_root/T_node — non utilisés par ce driver ; time_limit n'est de
# toute façon PAS branché dans gecode_bab_v1cld_gnn_batched.cpp — cf. sanity
# check du 2026-08-19 — seul le mur SLURM protège contre un run trop long).
#
# Sanity-check (n15/N100/seed43, K=64) déjà validé le 2026-08-19 : OPTIMAL,
# objectif 3436.0230 — identique à ref et à gnn_batched (v1LD).
#
# Mêmes adaptations Trillium que run_gnn_array.sh (V1LD) :
#   - --chdir /project/def-qcappart, --gpus-per-node=1 (pas --gres, obligatoire
#     même si ça n'était pas nécessaire ici), pas de --mem.
#   - logs/résultats vers $SCRATCH (/project en lecture seule sur les nœuds de
#     calcul) — à rapatrier après coup depuis le login node.
#   - module load StdEnv/2023 gcc/12.3 avant setup_gpu_env.sh (dépendance Lmod).
#   - Cache TorchScript (.v1cld_scripted_batched.pt) déjà PRÉ-GÉNÉRÉ pour ce
#     checkpoint (venv_export_only, torch==2.13.0) — sinon échec
#     ModuleNotFoundError: torch au premier appel.
#   - parse_result.py étendu pour accepter mode="v1cld_gnn_batched" (même
#     format de sortie que gnn_batched, sans les lignes "Borne GNN 1er niveau" —
#     first_level_bound_value/gap_root_pct resteront vides, normal).
#
# Soumettre :
#   sbatch --array=1-120 test_paper/mkp/trillium/v1cld/run_v1cld_array.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/def-qcappart/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=test_paper_trillium_v1cld
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/romanlen/test_paper_trillium_v1cld/slurm_logs/slurm_%x_%A_%a.out
#SBATCH --error=/scratch/romanlen/test_paper_trillium_v1cld/slurm_logs/slurm_%x_%A_%a.err

set -euo pipefail

REPO="/project/def-qcappart/romanlen/Learning_bounds_CCCP"
cd "$REPO"

module load StdEnv/2023 gcc/12.3
source solver_bab/setup_gpu_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

TESTDIR="$REPO/test_paper/mkp/trillium/v1cld"
SCRATCH_OUT="/scratch/romanlen/test_paper_trillium_v1cld"
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
TIME_LIM_S=0   # non branché côté solveur — mur SLURM (24h) seul filet de sécurité
K=64

LOGFILE="$SCRATCH_OUT/logs/v1cld_n${N_ITEMS}_N${N_SCEN}_seed${SEED}.log"
RESULT_CSV="$SCRATCH_OUT/results/v1cld_n${N_ITEMS}_N${N_SCEN}_seed${SEED}.csv"

{
echo "======================================================================"
echo "  test_paper_trillium_v1cld — n=$N_ITEMS N=$N_SCEN m=$M seed=$SEED — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}   Nœud : $(hostname)"
echo "  K=$K  backend=native (v1cld)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
} | tee "$LOGFILE"

"$BINARY" "$DATASET" "$CKPT" "$EPS" "$TIME_LIM_S" "$REP" cuda "$K" \
  2>&1 | tee -a "$LOGFILE"

python3 "$TESTDIR/parse_result.py" v1cld_gnn_batched "$N_ITEMS" "$N_SCEN" "$M" "$SEED" "$LOGFILE" > "$RESULT_CSV"

echo "" | tee -a "$LOGFILE"
echo "Terminé — $(date)" | tee -a "$LOGFILE"
