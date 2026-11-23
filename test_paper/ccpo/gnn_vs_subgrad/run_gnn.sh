#!/bin/bash
# test_paper/ccpo/gnn_vs_subgrad/run_gnn.sh — phase 1 de t* (GPU, courte) : à la racine de chacune des 120 instances,
# λ prédit par le GNN (modèle ep57, forward CUDA, graphe et forward du B&B batché, batch de taille 1), puis L(λ) ;
# pré-chauffage exclu, médiane de REPS appels. λ est écrit dans $OUT/lambda/ pour la phase 2. Un seul job pour les 120
# instances ; celles déjà faites sont sautées (relancer le job suffit).
# Soumettre depuis la racine du dépôt, sur un nœud de connexion GPU (Trillium : trig-login01) :
#   sbatch -o "$SCRATCH/slurm_tstar_gnn_%j.out" test_paper/ccpo/gnn_vs_subgrad/run_gnn.sh
#SBATCH --account=def-qcappart
#SBATCH --job-name=ccpo_tstar_gnn
#SBATCH --partition=debug
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=00:59:00
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO"
source env/modules_trillium.sh
source env/setup_gpu_env.sh
source test_paper/ccpo/gnn_vs_subgrad/config.env
OMP=${SLURM_CPUS_PER_TASK:-24}; export OMP_NUM_THREADS=$OMP
mkdir -p "$OUT/lambda" "$OUT/logs_gnn" "$OUT/results_gnn"
echo "# phase gnn — nœud $(hostname) — job ${SLURM_JOB_ID:-local} — omp=$OMP reps=$REPS — binaire $(md5sum $BIN | cut -d' ' -f1) — modèle $(md5sum $MODEL | cut -d' ' -f1)"
nvidia-smi -L 2>/dev/null || true
n_ok=0; n_skip=0; n_err=0
while read -r n N idx seed cert; do
  NAME=n${n}_N${N}_i${idx}
  CSV="$OUT/results_gnn/gnn_$NAME.csv"; LAM="$OUT/lambda/lambda_$NAME.txt"
  [[ -s $CSV && -s $LAM ]] && { n_skip=$((n_skip + 1)); continue; }
  LINE=$(timeout -k 30 1800 $BIN gnn "$INSTANCES_DIR/n${n}_N${N}/i${idx}/ep_000000" $MODEL $DEVICE $OMP $TORCH_THREADS "$LAM" $REPS \
         2> "$OUT/logs_gnn/gnn_$NAME.log")
  RC=$?
  if [[ $RC -ne 0 || -z "$LINE" ]]; then echo "[ÉCHEC] $NAME rc=$RC"; n_err=$((n_err + 1)); continue; fi
  printf "idx,seed,n,N,n_free,gnn_bound,gnn_time_s,gnn_graph_s,gnn_forward_s,gnn_eval_s,device,omp,host\n%s,%s,%s,%s,%s,%s\n" \
    $idx $seed "$LINE" $DEVICE $OMP $(hostname) > "$CSV"
  echo "[OK] $NAME : $LINE"; n_ok=$((n_ok + 1))
done < "${LISTFILE:-test_paper/ccpo/instances_list.txt}"
echo "# fin phase gnn : $n_ok faites, $n_skip déjà présentes, $n_err échecs"
