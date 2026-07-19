#!/bin/bash
# bench_sdedit_vs_homogene_n20_N50.sh — Compare REF (squelette nu, sans
# borne) vs GNN homogène natif vs GNN sd_edit natif, sur LA MÊME instance
# (n=20/m=5/N=50, ccmk-song-pisinger-n20-m5-a50-we-r42-S50.txt) que le
# benchmark de référence du fix OpenMP (cf. mémoire
# "gnn-bab-openmp-zeta-fix-validated", job 15980254 — mêmes EPS/T_ROOT/
# T_NODE/device pour rester comparable aux chiffres déjà établis : 11.32s
# construction de graphe, 43.74s forward, 3.52s ζ, 59.26s total pour
# l'homogène).
#
# Objectif : voir si l'étoile scénario-scénario de sd_edit (O(n_g·N) au lieu
# de O(n_g·N²) pour la clique homogène) réduit réellement la construction de
# graphe et/ou le forward sur cette instance, et comment le nombre de nœuds
# explorés se compare à REF (élagage utile ou pas).
#
# Checkpoints NON contrôlés à budget d'epochs égal (limitation connue, cf.
# mémoire "portage backend natif sd_edit") : homogène=ep197/10k, sd_edit=
# ep500/10k (même dataset n10+20, 10k graphes, mais epochs différents — le
# run sd_edit 100k comparable est encore PENDING en SLURM au moment du
# lancement de ce script). Le nombre de nœuds explorés reflète donc à la
# fois la topologie ET la qualité du checkpoint — ne pas interpréter un
# écart de nœuds comme purement dû à la topologie sans regarder aussi les
# valeurs de borne à la racine.
#
# Soumettre :
#   sbatch bench_sdedit_vs_homogene_n20_N50.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=bench_sdedit_vs_homogene_n20N50
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n20-m5-a50-we-r42-S50.txt"
CKPT_HOMOGENE="gnn/runs/train_scaled/ckpt_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_10k_ep0197.pt"
CKPT_SDEDIT="gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
EPS=0.10
T_ROOT=200
T_NODE=50
TIME_LIM_S=900   # par exécution ; 3 exécutions x 15 min < 1h SLURM

VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export LD_LIBRARY_PATH="$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

BIN="$REPO/solver_bab/build/bab_mkp_gecode"

echo "======================================================================"
echo "  bench_sdedit_vs_homogene_n20_N50 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)   OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  Dataset=$DATASET"
echo "  eps=$EPS  T_root=$T_ROOT  T_node=$T_NODE  time_limit=${TIME_LIM_S}s (par exécution)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "=== [1/3] MODE=ref (squelette nu, sans borne) ==="
time "$BIN" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" ref

echo ""
echo "=== [2/3] MODE=gnn backend=native variant=homogene (clique O(n_g.N^2)) ==="
time "$BIN" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" gnn non 0.1 "$CKPT_HOMOGENE" cuda python3 native homogene

echo ""
echo "=== [3/3] MODE=gnn backend=native variant=sd_edit (étoile O(n_g.N)) ==="
time "$BIN" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" gnn non 0.1 "$CKPT_SDEDIT" cuda python3 native sd_edit

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
