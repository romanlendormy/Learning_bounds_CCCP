#!/bin/bash
# bench_gnn_batched_depth_gated_sweep_n20_N100.sh — REF vs GNN sd_edit BATCHÉ
# (mode=gnn_batched, K nœuds par forward) à n=20/N=100, même sweep de
# gnn_max_depth_frac que bench_gnn_depth_gated_sweep_n20_N100.sh (version
# séquentielle, job 16492704 — aucun seuil n'y battait REF).
#
# Motivation : le batching amortit le coût du forward GNN sur K nœuds (gain
# mesuré ×2.7 wall-clock, 70s→26s, n=20/N=100 — cf. mémoire
# "gnn_batched_bab_integration_result"). Le coût par nœud y est donc bien
# plus faible qu'en séquentiel (~2ms/nœud) — la question : est-ce que couper
# la borne à certaines profondeurs change la conclusion (aucun seuil ne bat
# REF) une fois ce coût par nœud déjà réduit par le batching ?
#
# Note sur la sémantique de gnn_max_depth_frac dans ce backend : le bucketing
# par pending[ng_depth] EST un bucketing par profondeur exacte
# (depth = n - ng_depth), donc le seuil s'applique par bucket entier, pas nœud
# par nœud — mais le même piège que la version séquentielle s'applique : le
# nœud racine nu n'est jamais mis en pending (extract_pending/bucketing n'a
# lieu que sur les enfants après commit()), donc la plus petite depth qui
# atteint jamais un bucket est 1, pas 0. gnn_max_depth_frac=0.0 devrait donc
# à nouveau reproduire REF exactement (conservé comme sanity-check).
#
# Pas de `set -e` : un seuil qui échoue ne doit pas faire perdre les autres.
#
# Soumettre :
#   sbatch bench_gnn_batched_depth_gated_sweep_n20_N100.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=gnn_batched_depth_sweep_n20N100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -uo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

source solver_bab/setup_gpu_env.sh
VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

CKPT="gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
PROD="$REPO/solver_bab/build/bab_mkp_gecode"
N_ITEMS=20
N_SCEN=100
DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n${N_ITEMS}-m5-a50-we-r42-S${N_SCEN}.txt"
EPS=0.10
T_ROOT=200
T_NODE=50
TIME_LIMIT=300
# K=128 : légèrement meilleur que K=32 à N<=100 d'après le sweep de
# robustesse (job 16406490, cf. mémoire "gnn_batched_robustness_sweep_result").
K=128
THRESHOLDS=(1.0 0.9 0.8 0.7 0.6 0.5 0.0)
# (0.0 conservé comme point de sanity-check : doit reproduire REF exactement)

echo "======================================================================"
echo "  bench_gnn_batched_depth_gated_sweep_n20_N100 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Noeud : $(hostname)"
echo "  n=$N_ITEMS  N=$N_SCEN  time_limit=${TIME_LIMIT}s  K=$K  seuils : ${THRESHOLDS[*]}"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "######################################################################"
echo "  REF (squelette nu, sans borne)"
echo "######################################################################"
"$PROD" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIMIT" 0 "$T_ROOT" ref \
  || echo "[ERREUR] REF a échoué (code $?)"

for THRESHOLD in "${THRESHOLDS[@]}"; do
  echo ""
  echo "######################################################################"
  echo "  GNN BATCHÉ sd_edit — K=$K  gnn_max_depth_frac=$THRESHOLD"
  echo "######################################################################"
  # ordre CLI mode=gnn_batched : ... [ignorés 8-17] [K=argv18] [gnn_max_depth_frac=argv19]
  "$PROD" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIMIT" 0 "$T_ROOT" gnn_batched non 0.1 "$CKPT" cuda python3 native sd_edit 0.0 0.0 0.0 "$K" "$THRESHOLD" \
    || echo "[ERREUR] GNN batché gnn_max_depth_frac=$THRESHOLD a échoué (code $?)"
done

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
