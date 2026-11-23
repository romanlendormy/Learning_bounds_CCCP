#!/bin/bash
# test_paper/ccpo/gnn_vs_subgrad/run_subgrad_nodepack.sh — phase 2 de t* (CPU seul), après run_gnn.sh : pour chaque instance,
# relit λ_GNN, ré-évalue L(λ_GNN) sur ce nœud (coût d'évaluation d'un appel GNN mesuré sur le même CPU que le sous-gradient),
# puis sous-gradient à froid (λ = 0, pas 1/√(t+1), meilleur itéré), arrêté dès que sa borne atteint L(λ_GNN) : t* et son
# temps sont mesurés ; censure à T_MAX itérations ou TIME_LIM s. Nœuds CPU entiers : PER_NODE instances à la fois, chacune
# sur CORES cœurs physiques distincts (taskset). Instances déjà faites sautées.
# Soumettre depuis la racine du dépôt, sur un nœud de connexion CPU, 120 instances / 8 par nœud :
#   sbatch --array=1-15 -o "$SCRATCH/slurm_tstar_sub_%A_%a.out" test_paper/ccpo/gnn_vs_subgrad/run_subgrad_nodepack.sh
#SBATCH --account=def-qcappart
#SBATCH --job-name=ccpo_tstar_sub
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=24
#SBATCH --time=01:30:00
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO"
source env/modules_trillium.sh
# Build CUDA (repéré au stub NVSHMEM) : bibliothèques CUDA nécessaires même sans GPU
[[ -e build/solver_bab/libnvshmem_host.so.3 ]] && source env/setup_gpu_env.sh
source test_paper/ccpo/gnn_vs_subgrad/config.env
PER_NODE=${PER_NODE:-8}; CORES=${CORES:-24}
LISTFILE=${LISTFILE:-test_paper/ccpo/instances_list.txt}
mkdir -p "$OUT/logs_subgrad" "$OUT/results_subgrad"
NL=$(grep -c . "$LISTFILE")
first=$(( (SLURM_ARRAY_TASK_ID - 1) * PER_NODE + 1 )); last=$(( SLURM_ARRAY_TASK_ID * PER_NODE )); (( last > NL )) && last=$NL
mapfile -t PHYS < <(lscpu -p=CPU,CORE,SOCKET | grep -v '^#' | awk -F, '!seen[$3","$2]++ {print $1}')
echo "# phase subgrad — nœud $(hostname) — tâche ${SLURM_ARRAY_TASK_ID} : lignes $first-$last — binaire $(md5sum $BIN | cut -d' ' -f1)"

run_one() {  # ligne de la liste, cœurs alloués
  local line=$1 cpus=$2 n N idx seed cert
  read -r n N idx seed cert <<< "$(sed -n "${line}p" "$LISTFILE")"
  local NAME=n${n}_N${N}_i${idx}
  local LAM="$OUT/lambda/lambda_$NAME.txt" CSV="$OUT/results_subgrad/subgrad_$NAME.csv" LOG="$OUT/logs_subgrad/subgrad_$NAME.log"
  [[ -s $CSV ]] && return
  [[ -s $LAM ]] || { echo "[ERREUR] $LAM absent : lancer run_gnn.sh d'abord"; return; }
  echo "# n=$n N=$N idx=$idx — nœud $(hostname) cœurs $cpus — T_max=$T_MAX time_lim=$TIME_LIM reps=$REPS — $(env | grep -E '^(CCPO_ZETA|OMP_WAIT)' | sort | tr '\n' ' ')" > "$LOG"
  local LINE RC
  LINE=$(OMP_NUM_THREADS=$CORES timeout -k 60 $(( TIME_LIM + 1800 )) taskset -c "$cpus" \
         $BIN subgrad "$INSTANCES_DIR/n${n}_N${N}/i${idx}/ep_000000" "$LAM" $CORES $T_MAX $TIME_LIM $REPS 2>> "$LOG")
  RC=$?
  echo "# fin rc=$RC" >> "$LOG"
  [[ $RC -eq 0 && -n "$LINE" ]] || { echo "[ÉCHEC] $NAME rc=$RC"; return; }
  printf "idx,seed,n,N,n_free,gnn_bound,gnn_eval_cpu_s,subgrad_first,subgrad_best,t_star,t_star_time_s,iters_run,subgrad_time_s,stop_reason,conv_iter,last_impr_iter,last_impr_time_s,omp,host\n%s,%s,%s,%s,%s\n" \
    $idx $seed "$LINE" $CORES $(hostname) > "$CSV"
  echo "[OK] $NAME : $LINE"
}
slot=0
for line in $(seq "$first" "$last"); do
  run_one "$line" "$(IFS=,; echo "${PHYS[*]:$((slot * CORES)):$CORES}")" &
  slot=$((slot + 1))
done
wait
