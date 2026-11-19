#!/bin/bash
# test_paper/mkp/dld/run_dld_array.sh — LD avec la borne DLD (λ prédit par le GNN
# DLD) sur les 120 instances de test : moteur batché (K = 64), backend LibTorch
# natif, une tâche d'array par ligne de test_paper/mkp/instances_list.txt.
#
# CLI de bab_mkp_gecode_dld_gnn_batched (cf. solver_bab/mkp/main_bab_dld_gnn_batched.cpp) :
#   <dataset> <ckpt> [eps] [time_limit] [rep] [device] [K] [stop_after_first_level]
# time_limit n'est pas utilisé par ce moteur : le mur SLURM (24 h) est la seule limite.
# Mémoire GPU : au-delà de n=20/N=500, 80 Go (H100) ne suffisent plus ; dans l'article,
# ces instances ont tourné sur H200 (colonne gpu_model des résultats).
#
# FIRST_LEVEL_ONLY=1 : arrête le B&B dès que la borne du premier niveau (enfants de la
# racine) est connue — mesure du gap à la racine, résultats dans dld/results_firstlevel/.
#
# Prérequis : export TorchScript du modèle déjà généré (cf. test_paper/mkp/README.md).
# Soumettre depuis la racine du dépôt :
#   sbatch --array=1-120 test_paper/mkp/dld/run_dld_array.sh
#   sbatch --array=1-120 --time=00:30:00 --export=ALL,FIRST_LEVEL_ONLY=1 test_paper/mkp/dld/run_dld_array.sh

#SBATCH --account=def-qcappart
#SBATCH --job-name=mkp_dld_gnn
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%x_%A_%a.out
#SBATCH --error=logs/slurm_%x_%A_%a.err

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO"   # les binaires B&B doivent tourner depuis la racine du dépôt
source env/modules_trillium.sh
source env/setup_gpu_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

TD="$REPO/test_paper/mkp"
INSTANCES_DIR="${INSTANCES_DIR:-$TD/instances}"
FIRST_LEVEL_ONLY="${FIRST_LEVEL_ONLY:-0}"
SUFFIX=""; [[ "$FIRST_LEVEL_ONLY" == 1 ]] && SUFFIX="_firstlevel"
OUT="${OUT_DIR:-${SCRATCH:-$REPO/runs}/test_paper_mkp/dld$SUFFIX}"
mkdir -p "$OUT/logs" "$OUT/results"

read -r N_ITEMS N_SCEN M SEED FILE <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TD/instances_list.txt")"
DATASET="$INSTANCES_DIR/$FILE"
[[ -f "$DATASET" ]] || { echo "[ERREUR] instance introuvable : $DATASET" >&2; exit 1; }

CKPT="models/mkp/dld/ckpt_dld_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_dld_n10_20_500ep_ep0500.pt"
EPS=0.10; K=64

TAG="n${N_ITEMS}_N${N_SCEN}_seed${SEED}$SUFFIX"
LOG="$OUT/logs/dld_$TAG.log"
{
echo "# dld gnn_batched — n=$N_ITEMS N=$N_SCEN m=$M seed=$SEED — K=$K first_level_only=$FIRST_LEVEL_ONLY — $(date)"
echo "# job ${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0} — nœud $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
} > "$LOG"

"$REPO/build/solver_bab/bab_mkp_gecode_dld_gnn_batched" "$DATASET" "$CKPT" "$EPS" 0 0 cuda "$K" "$FIRST_LEVEL_ONLY" \
  >> "$LOG" 2>&1

python3 "$TD/parse_result.py" dld_gnn_batched "$N_ITEMS" "$N_SCEN" "$M" "$SEED" "$LOG" > "$OUT/results/dld_$TAG.csv"
