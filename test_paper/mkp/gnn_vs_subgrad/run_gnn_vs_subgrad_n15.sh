#!/bin/bash
# test_paper/mkp/gnn_vs_subgrad/run_gnn_vs_subgrad_n15.sh — GNN vs sous-gradient à la racine, n = 15.
# Pour chaque instance (N ∈ {100, 500, 1000}, graines 43..52, SLD et DLD) :
#   (a) borne GNN en un seul forward à la racine (mode gnn_root / gnn_root_dld) ;
#   (b) sous-gradient à froid (mode sld / dld) avec ROOT_SUBGRAD_TRACE=1 et T_root = 3000,
#       jusqu'à convergence : trace best_upper(t) sur stderr.
# t* = min{t : best_upper(t) ≤ borne GNN} est calculé ensuite par parse_gnn_vs_subgrad.py.
#
# 60 tâches : 1-30 = SLD, 31-60 = DLD ; dans chaque bloc, idx = tâche-1 (mod 30),
# N = [100, 500, 1000][idx / 10], graine = 43 + idx % 10.
# Soumettre depuis la racine du dépôt :
#   sbatch --array=1-60 test_paper/mkp/gnn_vs_subgrad/run_gnn_vs_subgrad_n15.sh

#SBATCH --account=def-qcappart
#SBATCH --job-name=gnn_vs_subgrad_n15
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm_%x_%A_%a.out
#SBATCH --error=logs/slurm_%x_%A_%a.err

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO"   # les binaires B&B doivent tourner depuis la racine du dépôt
source env/modules_trillium.sh
source env/setup_gpu_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

INSTANCES_DIR="${INSTANCES_DIR:-$REPO/test_paper/mkp/instances}"
OUT="${OUT_DIR:-${SCRATCH:-$REPO/runs}/test_paper_mkp/gnn_vs_subgrad_n15}"
mkdir -p "$OUT/logs"

BINARY="$REPO/build/solver_bab/bab_mkp_gecode"
N_VALUES=(100 500 1000)
T_ROOT=3000
TIME_LIMIT_S=60   # limite interne du solveur ; la convergence à n = 15 prend < 1700 itérations

TASK=$((SLURM_ARRAY_TASK_ID - 1))
if [[ $TASK -lt 30 ]]; then FORMULATION="sld"; IDX=$TASK; else FORMULATION="dld"; IDX=$((TASK - 30)); fi
N_SCEN=${N_VALUES[$((IDX / 10))]}
SEED=$((43 + IDX % 10))

if [[ "$FORMULATION" == "sld" ]]; then
    GNN_MODE="gnn_root"
    CKPT="models/mkp/sld/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
    GNN_EXTRA_ARGS=(cuda python3 native sd_edit)
else
    GNN_MODE="gnn_root_dld"
    CKPT="models/mkp/dld/ckpt_dld_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_dld_n10_20_500ep_ep0500.pt"
    GNN_EXTRA_ARGS=(cuda python3 native)
fi

DATASET="$INSTANCES_DIR/instance_n15_m5_N${N_SCEN}_seed${SEED}.txt"
[[ -f "$DATASET" ]] || { echo "[ERREUR] instance introuvable : $DATASET" >&2; exit 1; }

TAG="n15_N${N_SCEN}_seed${SEED}_${FORMULATION}"
echo "# gnn_vs_subgrad n15 — $TAG — job ${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0} — nœud $(hostname) — $(date)"

# (a) borne GNN à la racine
"$BINARY" "$DATASET" 0.10 30 30 0 200 "$GNN_MODE" non 0.1 "$CKPT" "${GNN_EXTRA_ARGS[@]}" \
    > "$OUT/logs/gnn_root_${TAG}.log" 2>&1
grep "racine" "$OUT/logs/gnn_root_${TAG}.log" || echo "[ATTENTION] pas de borne racine (GNN)" >&2

# (b) trace du sous-gradient à froid
ROOT_SUBGRAD_TRACE=1 "$BINARY" "$DATASET" 0.10 30 "$TIME_LIMIT_S" 0 "$T_ROOT" "$FORMULATION" \
    > "$OUT/logs/subgrad_${TAG}.stdout.log" 2> "$OUT/logs/subgrad_${TAG}.stderr.log"
grep "racine" "$OUT/logs/subgrad_${TAG}.stdout.log" || echo "[ATTENTION] pas de borne racine (sous-gradient)" >&2
echo "  lignes SUBGRAD_TRACE : $(grep -c '^SUBGRAD_TRACE' "$OUT/logs/subgrad_${TAG}.stderr.log" || true)"
