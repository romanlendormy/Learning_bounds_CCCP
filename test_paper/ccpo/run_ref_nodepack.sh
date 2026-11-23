#!/bin/bash
# test_paper/ccpo/run_ref_nodepack.sh — campagne CCPO de l'article, bras REF : B&B Gecode sans borne lagrangienne, avec la
# détection exacte de faisabilité résiduelle à chaque nœud (mode `feas` du solveur ; aucun λ, aucun réseau).
# REF n'utilise pas de GPU : il tourne sur des nœuds CPU entiers de Trillium, PER_NODE instances à la fois, chacune sur
# CORES cœurs physiques distincts (taskset). Une tâche d'array par paquet de PER_NODE lignes de instances_list.txt.
# Même binaire, mêmes instances et même arrêt à 23 h 50 que LD/LD0 ; ζ n'y sert qu'au test de faisabilité (coût nul),
# donc sans l'hybride SCIP.
# Soumettre depuis la racine du dépôt (nœud de connexion CPU), 120 instances / 8 par nœud :
#   sbatch --array=1-15 test_paper/ccpo/run_ref_nodepack.sh
# Reprise : une instance dont le CSV existe déjà est sautée sans rien écraser.

#SBATCH --account=def-qcappart
#SBATCH --job-name=ccpo_ref
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%x_%A_%a.out

set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"
source env/modules_trillium.sh
# Build CUDA (repéré au stub NVSHMEM) : bibliothèques CUDA nécessaires même sans GPU
[[ -e "$REPO/build/solver_bab/libnvshmem_host.so.3" ]] && source env/setup_gpu_env.sh
TD="$REPO/test_paper/ccpo"
source "$TD/campaign_config.env"
export CCPO_ZETA_SCIP=0
BIN="$REPO/build/solver_bab/bab_ccpo"
PER_NODE=${PER_NODE:-8}; CORES=${CORES:-24}

INSTANCES_DIR="${INSTANCES_DIR:-$TD/instances}"
OUT="${OUT_DIR:-${SCRATCH:-$REPO/runs}/test_paper_ccpo}"
mkdir -p "$OUT/logs" "$OUT/results/ref"
NL=$(grep -c . "$TD/instances_list.txt")
first=$(( (SLURM_ARRAY_TASK_ID - 1) * PER_NODE + 1 )); last=$(( SLURM_ARRAY_TASK_ID * PER_NODE )); (( last > NL )) && last=$NL
mapfile -t PHYS < <(lscpu -p=CPU,CORE,SOCKET | grep -v '^#' | awk -F, '!seen[$3","$2]++ {print $1}')
echo "# nœud $(hostname) — tâche ${SLURM_ARRAY_TASK_ID} : lignes $first-$last — ${#PHYS[@]} cœurs physiques"

run_one() {  # ligne de instances_list.txt, cœurs alloués
  local line=$1 cpus=$2 n N idx seed cert
  read -r n N idx seed cert <<< "$(sed -n "${line}p" "$TD/instances_list.txt")"
  local EP="$INSTANCES_DIR/n${n}_N${N}/i${idx}/ep_000000" TAG="n${n}_N${N}_i${idx}"
  [[ -d "$EP" ]] || { echo "[ERREUR] instance introuvable : $EP" >&2; return; }
  local RES="$OUT/results/ref/ref_$TAG.csv" LOG="$OUT/logs/ref_$TAG.log"
  [[ -s "$RES" ]] && { echo "[SKIP] $RES existe"; return; }
  echo "# ccpo ref (mode feas) — n=$n N=$N idx=$idx seed=$seed certificat=$cert — $(date)" > "$LOG"
  echo "# job ${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0} — nœud $(hostname) — cœurs $cpus" >> "$LOG"
  local T0=$(date +%s)
  OMP_NUM_THREADS=$CORES timeout -k 60 "$KILL_ABS" taskset -c "$cpus" "$BIN" "$EP" "$TL" feas >> "$LOG" 2>&1
  local RC=$?
  echo "# fin rc=$RC temps_mur=$(( $(date +%s) - T0 ))s" >> "$LOG"
  RUN_RC=$RC RUN_ELAPSED=$(( $(date +%s) - T0 )) python3 "$TD/parse_result.py" feas "$n" "$N" "$idx" "$seed" "$LOG" > "$RES.tmp" \
    && mv -f "$RES.tmp" "$RES"
}
slot=0
for line in $(seq "$first" "$last"); do
  run_one "$line" "$(IFS=,; echo "${PHYS[*]:$((slot * CORES)):$CORES}")" &
  slot=$((slot + 1))
done
wait
