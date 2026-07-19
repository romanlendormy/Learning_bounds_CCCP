#!/bin/bash
# debug_consistency_n20_N20.sh — investigation du bug de correction trouvé
# dans le sweep de robustesse (job 16186510) : sur n=20/N=20, le B&B GNN
# batché retourne un "OPTIMAL" faux (5130.397 au lieu de 5166.381, aux 3 K
# testés y compris K=1). Utilise l'outil gnn_batch_consistency_probe déjà
# existant (Étape 2d) pour comparer directement, masque par masque, le
# backend mono-nœud (GNNSdEditNative, production, INCHANGÉ) au backend
# batché (GNNSdEditNativeBatched) — sur CETTE instance précise, à TOUTES
# les profondeurs (n_g_target de 1 à 19), avec plusieurs seeds.
#
# Objectif : localiser si la divergence vient du modèle/export batché
# (gnn_sd_edit_scriptable_batched.py) lui-même, à quelle profondeur/n_g,
# indépendamment de la logique de bucket/flush du driver B&B.
#
# Soumettre :
#   sbatch debug_consistency_n20_N20.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=debug_consistency_n20_N20
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -uo pipefail   # PAS de -e : un run qui échoue ne doit pas arrêter le balayage

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

source solver_bab/setup_gpu_env.sh
VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n20-m5-a50-we-r42-S20.txt"
CKPT="gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
PROBE="$REPO/solver_bab/build/gnn_batch_consistency_probe"

echo "======================================================================"
echo "  debug_consistency_n20_N20 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Noeud : $(hostname)"
echo "  n=20 N=20 — balayage n_g_target=1..19, seeds=0,1,2, B=16"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

n=20
for ng in $(seq 1 $((n-1))); do
  for seed in 0 1 2; do
    echo ""
    echo "--- n_g_target=$ng  seed=$seed ---"
    "$PROBE" "$DATASET" "$CKPT" 0 0.1 cuda "$ng" 16 "$seed" \
      || echo "[ERREUR ou ÉCHEC TOLÉRANCE] n_g_target=$ng seed=$seed (code $?)"
  done
done

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
