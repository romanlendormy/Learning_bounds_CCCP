#!/bin/bash
# test_paper/ccpo/run_ld_array.sh — campagne CCPO de l'article, un bras par soumission :
#   MODE=gnn       LD  : λ prédit par le GNN (modèle ep57), forward CUDA
#   MODE=gnn_zero  LD0 : λ = 0 (même moteur, sans réseau)
# Moteur batché (buckets de K = 32 nœuds), ζ OpenMP sur les scénarios + hybride SCIP, presolve exact,
# θ résiduel ; réglages dans campaign_config.env. Les deux bras tournent sur le même matériel :
# quart de nœud GPU Trillium (1 H100, 24 cœurs). Une tâche d'array par ligne de instances_list.txt.
# Soumettre depuis la racine du dépôt :
#   sbatch --array=1-120 --export=ALL,MODE=gnn      test_paper/ccpo/run_ld_array.sh
#   sbatch --array=1-120 --export=ALL,MODE=gnn_zero test_paper/ccpo/run_ld_array.sh
# Reprise : une tâche dont le CSV existe déjà s'arrête sans rien écraser.

#SBATCH --account=def-qcappart
#SBATCH --job-name=ccpo_ld
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%x_%A_%a.out
#SBATCH --error=logs/slurm_%x_%A_%a.err

set -uo pipefail
MODE=${MODE:?MODE=gnn|gnn_zero}
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"
source env/modules_trillium.sh
source env/setup_gpu_env.sh
TD="$REPO/test_paper/ccpo"
source "$TD/campaign_config.env"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
BIN="$REPO/build/solver_bab/bab_ccpo"

INSTANCES_DIR="${INSTANCES_DIR:-$TD/instances}"
OUT="${OUT_DIR:-${SCRATCH:-$REPO/runs}/test_paper_ccpo}"
NAME=$([[ $MODE == gnn ]] && echo ld || echo ld0)
mkdir -p "$OUT/logs" "$OUT/results/$NAME"

read -r n N idx seed cert <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TD/instances_list.txt")"
EP="$INSTANCES_DIR/n${n}_N${N}/i${idx}/ep_000000"
[[ -d "$EP" ]] || { echo "[ERREUR] instance introuvable : $EP" >&2; exit 1; }
TAG="n${n}_N${N}_i${idx}"
RES="$OUT/results/$NAME/${NAME}_$TAG.csv"
[[ -s "$RES" ]] && { echo "[SKIP] $RES existe"; exit 0; }
LOG="$OUT/logs/${NAME}_$TAG.log"
{
echo "# ccpo $NAME (mode $MODE) — n=$n N=$N idx=$idx seed=$seed certificat=$cert — $(date)"
echo "# job ${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0} — nœud $(hostname) — OMP_NUM_THREADS=$OMP_NUM_THREADS"
nvidia-smi -L 2>/dev/null || true
} > "$LOG"

# Arguments positionnels de bab_ccpo (cf. solver_bab/ccpo/main_gecode_ccpo.cpp) :
#   épisode time_lim mode T_root T_node K omp_threads [modèle torch_threads device]
ARGS=("$EP" "$TL" "$MODE" "$T_ROOT" "$T_NODE" "$K" "$OMP_NUM_THREADS")
[[ $MODE == gnn ]] && ARGS+=("$MODEL" "$TORCH_THREADS" "$DEVICE")
T0=$(date +%s)
timeout -k 60 "$KILL_ABS" "$BIN" "${ARGS[@]}" >> "$LOG" 2>&1
RC=$?
echo "# fin rc=$RC temps_mur=$(( $(date +%s) - T0 ))s" >> "$LOG"
RUN_RC=$RC RUN_ELAPSED=$(( $(date +%s) - T0 )) python3 "$TD/parse_result.py" "$MODE" "$n" "$N" "$idx" "$seed" "$LOG" > "$RES.tmp" \
  && mv -f "$RES.tmp" "$RES"
