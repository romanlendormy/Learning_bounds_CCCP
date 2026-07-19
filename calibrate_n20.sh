#!/bin/bash
# calibrate_n20.sh — Calibrage rapide du temps par run avant de lancer le sweep
# complet à n=20 (cf. sweep n=10, session "comparaison n=10/N={20,50,100}") :
# l'arbre B&B à n=20 est potentiellement bien plus gros (cf. précédent test
# natif n=20/N=50 : ~25k nœuds explorés) donc on mesure le taux nœuds/s réel
# sur v1ld et gnn (les deux modes avec un coût par nœud non-négligeable) avant
# de dimensionner time_limit/wall-clock du job de comparaison à 4 méthodes.
#
# Soumettre :
#   sbatch calibrate_n20.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=calibrate_n20
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n20-m5-a50-we-r42-S100.txt"
CKPT="gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
EPS=0.10
T_ROOT=200
T_NODE=50
TIME_LIM_S=120   # cap court : juste pour mesurer le taux nœuds/s, pas pour finir

VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export LD_LIBRARY_PATH="$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

BINARY="$REPO/solver_bab/build/bab_mkp_gecode"

echo "=== calibrate_n20 — $(date) — nœud $(hostname) ==="

echo ""
echo "=== MODE=v1ld (cap ${TIME_LIM_S}s) ==="
time "$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" v1ld

echo ""
echo "=== MODE=gnn backend=ipc variant=sd_edit device=cuda (cap ${TIME_LIM_S}s) ==="
time "$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" gnn non 0.1 "$CKPT" cuda python3 ipc sd_edit

echo ""
echo "=== Terminé — $(date) ==="
