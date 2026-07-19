#!/bin/bash
# _check_kelley_fix_cpu.sh — Vérification rapide (CPU, pas de file d'attente
# GPU) du correctif lam_bound=50 (au lieu de max|c|) sur la méthode C
# (V1LD SONG / Kelley) de compare_v1ld_song_gnn_n10_N50.py. Mêmes --pick_index
# par défaut -> même instance (inst_id=9564) que le run GPU déjà complété, donc
# les colonnes A/B restent comparables si on garde ce run comme référence.
#
# Soumettre :
#   sbatch _check_kelley_fix_cpu.sh

#SBATCH --account=def-qcappart
#SBATCH --job-name=check_kelley_fix
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:25:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

CKPT="gnn/runs/train_scaled/ckpt_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_10k_ep0200.pt"

VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)   $(date)"

python3 -u compare_v1ld_song_gnn_n10_N50.py \
    --ckpt "$CKPT" --device cpu \
    --subgrad_T 500 --kelley_method trustregion --kelley_max_iters 5000 --kelley_time_limit 300

echo "Terminé — $(date)"
