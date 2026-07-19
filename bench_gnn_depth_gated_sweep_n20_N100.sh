#!/bin/bash
# bench_gnn_depth_gated_sweep_n20_N100.sh — REF vs GNN sd_edit (native) à
# n=20/N=100, en balayant plusieurs seuils de profondeur relative
# (gnn_max_depth_frac) AU-DELÀ desquels on arrête de calculer/poster la borne
# — l'inverse de gnn_min_depth_frac (job 16206556, qui gatait près de la
# racine). Ici on garde la borne près de la racine et on la coupe passé une
# certaine profondeur ("le bas" de l'arbre), là où se concentrent le plus de
# nœuds more() ET le plus de fails_from_bound (depths 10-19 sur 20 dans le
# job 16206556) — hypothèse à tester : est-ce que sacrifier le pruning là où
# la borne est la PLUS utile est compensé par le volume de coût économisé
# (le gros de l'arbre), ou est-ce contre-productif ?
#
# Pas de `set -e` : un seuil qui échoue ne doit pas faire perdre les autres.
#
# Soumettre :
#   sbatch bench_gnn_depth_gated_sweep_n20_N100.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=gnn_depth_sweep_n20N100
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
# gnn_max_depth_frac : 1.0 = GNN plein (comportement historique, jamais gaté) ;
# 0.9→0.5 = coupe progressivement plus tôt. NB (découvert au run) : le plus
# petit depth qui atteint réellement more() est 1, PAS 0 — le nœud racine nu
# (avant tout branchement) n'appelle jamais more() dans ce driver (cf.
# solve_gecode_bab_mkp, boucle DFS : root->status() puis push direct, more()
# seulement sur les enfants après commit()). Donc gnn_max_depth_frac=0.0
# (floor(0*n)=0, seuil "depth>0") saute TOUT, dès le premier appel réel —
# équivalent exact à REF, pas "borne à la racine seulement" comme on
# pourrait le lire dans le nom du paramètre.
THRESHOLDS=(1.0 0.9 0.8 0.7 0.6 0.5 0.0)
# (0.0 conservé comme point de sanity-check : doit reproduire REF exactement)

echo "======================================================================"
echo "  bench_gnn_depth_gated_sweep_n20_N100 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Noeud : $(hostname)"
echo "  n=$N_ITEMS  N=$N_SCEN  time_limit=${TIME_LIMIT}s  seuils : ${THRESHOLDS[*]}"
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
  echo "  GNN sd_edit — gnn_max_depth_frac=$THRESHOLD"
  echo "######################################################################"
  # ordre CLI : ... variant gnn_min_depth_frac gnn_min_gap_frac gnn_max_depth_frac
  "$PROD" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIMIT" 0 "$T_ROOT" gnn non 0.1 "$CKPT" cuda python3 native sd_edit 0.0 0.0 "$THRESHOLD" \
    || echo "[ERREUR] GNN gnn_max_depth_frac=$THRESHOLD a échoué (code $?)"
done

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
