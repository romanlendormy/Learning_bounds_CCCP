#!/bin/bash
# test_paper/mkp/sensitivity/run_sensitivity_array.sh — Sensibilité au nombre de cœurs CPU :
# mêmes binaires et réglages que sld/run_gnn_array.sh (SLD) et dld/run_dld_array.sh (DLD),
# OMP_NUM_THREADS = --cpus-per-task. 24 tâches (tasks.txt : méthode n N m graine) :
# SLD (n, N) ∈ {(20,500), (20,1000), (25,100), (25,500), (30,100)} et
# DLD (n, N) ∈ {(20,500), (20,1000), (25,100)}, graines 43..45.
#
# Soumettre depuis la racine du dépôt, une fois par nombre de cœurs :
#   sbatch --array=1-24 --cpus-per-task=4  --export=ALL,TAG=c4  test_paper/mkp/sensitivity/run_sensitivity_array.sh
#   sbatch --array=1-24 --cpus-per-task=24 --export=ALL,TAG=c24 test_paper/mkp/sensitivity/run_sensitivity_array.sh

#SBATCH --account=def-qcappart
#SBATCH --job-name=mkp_sensitivity
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_%x_%A_%a.out
#SBATCH --error=logs/slurm_%x_%A_%a.err

set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO"   # les binaires B&B doivent tourner depuis la racine du dépôt
source env/modules_trillium.sh
source env/setup_gpu_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

TD="$REPO/test_paper/mkp"
INSTANCES_DIR="${INSTANCES_DIR:-$TD/instances}"
TAG=${TAG:-c${SLURM_CPUS_PER_TASK:-4}}
OUT="${OUT_DIR:-${SCRATCH:-$REPO/runs}/test_paper_mkp/sensitivity}/$TAG"
mkdir -p "$OUT/logs" "$OUT/results"

read -r METHOD N_ITEMS N_SCEN M SEED <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TD/sensitivity/tasks.txt")"
DATASET="$INSTANCES_DIR/instance_n${N_ITEMS}_m${M}_N${N_SCEN}_seed${SEED}.txt"
[[ -f "$DATASET" ]] || { echo "[ERREUR] instance introuvable : $DATASET" >&2; exit 1; }

NAME="${METHOD}_n${N_ITEMS}_N${N_SCEN}_seed${SEED}"
LOG="$OUT/logs/$NAME.log"
EPS=0.10; K=64
{
echo "# sensibilité cœurs — $METHOD n=$N_ITEMS N=$N_SCEN m=$M seed=$SEED — TAG=$TAG — nœud $(hostname) — job ${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
echo "# affinité CPU : $(taskset -cp $$ | sed 's/.*: //') ; OMP_NUM_THREADS=$OMP_NUM_THREADS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
} > "$LOG" 2>&1

T0=$(date +%s)
if [[ "$METHOD" == "sld" ]]; then
  CKPT=models/mkp/sld/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt
  "$REPO/build/solver_bab/bab_mkp_gecode" "$DATASET" "$EPS" 50 0 0 200 gnn_batched \
    non 0.1 "$CKPT" cuda python3 ipc sd_edit 0.0 0.0 1.0 "$K" 1.0 surrogate 1 native >> "$LOG" 2>&1
  MODE=gnn_batched
else
  CKPT=models/mkp/dld/ckpt_dld_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_dld_n10_20_500ep_ep0500.pt
  "$REPO/build/solver_bab/bab_mkp_gecode_dld_gnn_batched" "$DATASET" "$CKPT" "$EPS" 0 0 cuda "$K" >> "$LOG" 2>&1
  MODE=dld_gnn_batched
fi
RC=$?
echo "# fin rc=$RC temps_mur=$(( $(date +%s) - T0 ))s" >> "$LOG"
python3 "$TD/parse_result.py" $MODE "$N_ITEMS" "$N_SCEN" "$M" "$SEED" "$LOG" > "$OUT/results/$NAME.csv"
