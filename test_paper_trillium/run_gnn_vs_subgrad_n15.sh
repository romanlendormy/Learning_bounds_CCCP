#!/bin/bash
# run_gnn_vs_subgrad_n15.sh (Trillium) — pour chaque instance de test
# (n=15, N∈{100,500,1000}, seed∈43..52, V1LD et V1CLD), lance :
#   (a) le forward GNN root-only ("gnn_root" pour V1LD, "gnn_root_v1cld"
#       pour V1CLD) → root_bound_value = borne obtenue en un seul forward
#       GNN à la racine.
#   (b) le sous-gradient déjà en production ("v1ld"/"v1cld"), avec
#       ROOT_SUBGRAD_TRACE=1 et T_root=3000 (>> nb d'itérations nécessaires
#       à la convergence par rel_tol, cf. sanity check job 821840 :
#       convergence observée entre ~780 et ~1700 itérations sur n15/N100)
#       → trace best_upper(t) à la racine, dumpée sur stderr.
#
# Objectif : comparer la qualité de la borne obtenue par UN SEUL forward GNN
# à la racine à celle obtenue par un nombre t d'itérations de sous-gradient —
# déduire t* = min{t : sous-gradient(t) <= GNN_root} en post-traitement
# (parse_gnn_vs_subgrad.py), pour évaluer l'efficacité du GNN sur cette tâche.
#
# 60 tâches d'array : task 1-30 = V1LD (10 seeds x 3 N), task 31-60 = V1CLD.
# Grille par task (1-30) : idx = task-1 ; N = [100,500,1000][idx/10] ;
# seed = 43 + idx%10.
#
# Soumettre :
#   sbatch --array=1-60 test_paper_trillium/run_gnn_vs_subgrad_n15.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/def-qcappart/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=gnn_vs_subgrad_n15
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/romanlen/gnn_vs_subgrad_n15/slurm_logs/slurm_%x_%A_%a.out
#SBATCH --error=/scratch/romanlen/gnn_vs_subgrad_n15/slurm_logs/slurm_%x_%A_%a.err

set -euo pipefail

REPO="/project/def-qcappart/romanlen/Learning_bounds_CCCP"
cd "$REPO"

module load StdEnv/2023 gcc/12.3
source solver_bab/setup_gpu_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

SCRATCH_OUT="/scratch/romanlen/gnn_vs_subgrad_n15"
mkdir -p "$SCRATCH_OUT/logs" "$SCRATCH_OUT/slurm_logs"

BINARY="$REPO/solver_bab/build/bab_mkp_gecode"
N_VALUES=(100 500 1000)
T_ROOT=3000
TIME_LIMIT_S=60   # borne interne du solveur — largement suffisant pour que
                   # les deux cold-starts (T_root=3000) convergent par rel_tol
                   # (cf. sanity check : convergence observée sous 1700 iters)

TASK=$((SLURM_ARRAY_TASK_ID - 1))
if [[ $TASK -lt 30 ]]; then
    FORMULATION="v1ld"
    IDX=$TASK
else
    FORMULATION="v1cld"
    IDX=$((TASK - 30))
fi
N_SCEN=${N_VALUES[$((IDX / 10))]}
SEED=$((43 + IDX % 10))

if [[ "$FORMULATION" == "v1ld" ]]; then
    TESTDIR="$REPO/test_paper_trillium"
    GNN_MODE="gnn_root"
    SUBGRAD_MODE="v1ld"
    CKPT="$REPO/gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
    GNN_EXTRA_ARGS=(cuda python3 native sd_edit)
else
    TESTDIR="$REPO/test_paper_trillium_v1cld"
    GNN_MODE="gnn_root_v1cld"
    SUBGRAD_MODE="v1cld"
    CKPT="$REPO/gnn_sd_edit/runs_v1cld/ckpt_v1cld_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_v1cld_n10_20_500ep_ep0500.pt"
    GNN_EXTRA_ARGS=(cuda python3 native)
fi

DATASET="$TESTDIR/instance_n15_m5_N${N_SCEN}_seed${SEED}.txt"
if [[ ! -f "$DATASET" ]]; then
    echo "[ERREUR] instance introuvable : $DATASET" >&2
    exit 1
fi

TAG="n15_N${N_SCEN}_seed${SEED}_${FORMULATION}"
GNN_LOG="$SCRATCH_OUT/logs/gnn_root_${TAG}.log"
SUBGRAD_STDOUT="$SCRATCH_OUT/logs/subgrad_${TAG}.stdout.log"
SUBGRAD_STDERR="$SCRATCH_OUT/logs/subgrad_${TAG}.stderr.log"

echo "======================================================================"
echo "  gnn_vs_subgrad_n15 — $TAG — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}   Nœud : $(hostname)"
echo "======================================================================"

echo "--- (a) forward GNN root-only (mode=$GNN_MODE) ---"
"$BINARY" "$DATASET" 0.10 30 30 0 200 "$GNN_MODE" non 0.1 "$CKPT" "${GNN_EXTRA_ARGS[@]}" \
    > "$GNN_LOG" 2>&1
grep "racine" "$GNN_LOG" || echo "[ATTENTION] pas de ligne 'racine' dans $GNN_LOG" >&2

echo "--- (b) trace sous-gradient (mode=$SUBGRAD_MODE, T_root=$T_ROOT) ---"
ROOT_SUBGRAD_TRACE=1 "$BINARY" "$DATASET" 0.10 30 "$TIME_LIMIT_S" 0 "$T_ROOT" "$SUBGRAD_MODE" \
    > "$SUBGRAD_STDOUT" 2> "$SUBGRAD_STDERR"
grep "racine" "$SUBGRAD_STDOUT" || echo "[ATTENTION] pas de ligne 'racine' dans $SUBGRAD_STDOUT" >&2
N_TRACE=$(grep -c "^SUBGRAD_TRACE" "$SUBGRAD_STDERR" || true)
echo "  nb lignes SUBGRAD_TRACE : $N_TRACE"

echo ""
echo "Terminé — $TAG — $(date)"
