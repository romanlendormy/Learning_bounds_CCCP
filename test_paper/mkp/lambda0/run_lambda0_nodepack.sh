#!/bin/bash
# test_paper/mkp/lambda0/run_lambda0_nodepack.sh — LD0 : même B&B que sld/run_gnn_array.sh
# (gnn_batched, K = 64, ζ surrogate, T_root = 200, T_node = 50, ε = 0,1, 4 threads), mais
# la borne SLD est évaluée en λ = 0 à chaque nœud (GNN_ZERO_LAMBDA=1 : forward GNN sauté,
# même évaluation de L(λ)). CPU seul.
#
# Nœuds CPU entiers : chaque tâche d'array lance PER_NODE instances en parallèle, chacune
# épinglée sur CORES cœurs physiques (48 × 4 = 192 cœurs sur Trillium).
# Limite solveur 23 h 50 (le solveur imprime alors ses résultats, statut TIMEOUT/FEASIBLE).
#
# Soumettre depuis la racine du dépôt (nœud de connexion CPU de Trillium) :
#   sbatch --array=1-3 test_paper/mkp/lambda0/run_lambda0_nodepack.sh

#SBATCH --account=def-qcappart
#SBATCH --job-name=mkp_lambda0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%x_%A_%a.out

set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO"   # les binaires B&B doivent tourner depuis la racine du dépôt
source env/modules_trillium.sh
source env/setup_gpu_env.sh   # Gecode / OR-Tools / LibTorch dans LD_LIBRARY_PATH

TD="$REPO/test_paper/mkp"
LIST="$TD/instances_list.txt"
INSTANCES_DIR="${INSTANCES_DIR:-$TD/instances}"
OUT="${OUT_DIR:-${SCRATCH:-$REPO/runs}/test_paper_mkp/lambda0}"
BIN="$REPO/build/solver_bab/bab_mkp_gecode"
CKPT="models/mkp/sld/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"  # chargé, inutilisé (λ = 0)
EPS=0.10; T_ROOT=200; T_NODE=50; K=64
TIME_LIM_S=${TIME_LIM_S:-85800}   # 23 h 50
PER_NODE=${PER_NODE:-48}; CORES=${CORES:-4}
mkdir -p "$OUT/logs" "$OUT/results"

NL=$(grep -c . "$LIST")
first=$(( (SLURM_ARRAY_TASK_ID - 1) * PER_NODE + 1 )); last=$(( SLURM_ARRAY_TASK_ID * PER_NODE )); (( last > NL )) && last=$NL
# Un CPU logique par cœur physique (pas d'hyperthread partagé entre deux instances)
mapfile -t PHYS < <(lscpu -p=CPU,CORE,SOCKET | grep -v '^#' | awk -F, '!seen[$3","$2]++ {print $1}')
echo "# nœud $(hostname) | tâche ${SLURM_ARRAY_TASK_ID} : lignes $first-$last | ${#PHYS[@]} cœurs physiques"

run_one() {
  local line=$1 cpus=$2
  read -r n N m seed file <<< "$(sed -n "${line}p" "$LIST")"
  local TAG="n${n}_N${N}_seed${seed}" LOG
  LOG="$OUT/logs/lambda0_$TAG.log"
  echo "# lambda0 — n=$n N=$N m=$m seed=$seed — K=$K zeta=surrogate time_lim=$TIME_LIM_S — $(date) — nœud $(hostname) cœurs $cpus" > "$LOG"
  GNN_ZERO_LAMBDA=1 OMP_NUM_THREADS=$CORES MKL_NUM_THREADS=$CORES timeout -k 60 $((TIME_LIM_S + 1800)) taskset -c "$cpus" \
    "$BIN" "$INSTANCES_DIR/$file" $EPS $T_NODE $TIME_LIM_S 0 $T_ROOT gnn_batched \
    non 0.1 "$CKPT" cpu python3 ipc sd_edit 0.0 0.0 1.0 $K 1.0 surrogate 1 native >> "$LOG" 2>&1
  echo "# fin rc=$?" >> "$LOG"
  python3 "$TD/parse_result.py" lambda0 "$n" "$N" "$m" "$seed" "$LOG" > "$OUT/results/lambda0_$TAG.csv"
}

slot=0
for line in $(seq "$first" "$last"); do
  cpus=$(IFS=,; echo "${PHYS[*]:$((slot * CORES)):$CORES}")
  run_one "$line" "$cpus" &
  slot=$((slot + 1))
done
wait
echo "# fin tâche ${SLURM_ARRAY_TASK_ID} : $(date)"
