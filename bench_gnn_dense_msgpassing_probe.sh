#!/bin/bash
# bench_gnn_dense_msgpassing_probe.sh — Mesure C++ (pas Python, cf. mémoire
# "cudagraph-python-probe-overestimate") du gain réel de la reformulation
# dense (sans scatter/gather) du message passing sd_edit, validée
# numériquement équivalente à la version sparse actuelle (max|diff|~1e-7 sur
# 6 formes, gnn_sd_edit/prototype_dense_msgpassing.py) — aucun réentraînement,
# mêmes poids.
#
# Soumettre :
#   sbatch bench_gnn_dense_msgpassing_probe.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=bench_dense_msgpassing
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

module load cuda/12.6
VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export LD_LIBRARY_PATH="$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

CKPT_DIR="gnn_sd_edit/runs/train_scaled"
CKPT_BASE="ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
SPARSE_MODULE="$CKPT_DIR/$CKPT_BASE.scripted.pt"
DENSE_MODULE="$CKPT_DIR/$CKPT_BASE.dense_sdedit.scripted.pt"
N_G=20
N_Z=50
M=5
N_ITERS=2000
N_WARMUP=50

echo "======================================================================"
echo "  bench_gnn_dense_msgpassing_probe — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "=== Régénération du module dense (poids à jour, cf. prototype) ==="
python3 gnn_sd_edit/prototype_dense_msgpassing.py --ckpt "$CKPT_DIR/$CKPT_BASE"

echo ""
echo "=== (Re)compilation gnn_dense_msgpassing_probe ==="
cmake -S solver_bab -B solver_bab/build -DCMAKE_BUILD_TYPE=Release -DGNN_DEVICE=cuda >/dev/null
cmake --build solver_bab/build --target gnn_dense_msgpassing_probe -j "${SLURM_CPUS_PER_TASK:-4}"
PROBE="$REPO/solver_bab/build/gnn_dense_msgpassing_probe"

echo ""
echo "======================================================================"
echo "  n_g=$N_G N_z=$N_Z m=$M (racine, forme la plus grande)"
echo "======================================================================"
"$PROBE" "$SPARSE_MODULE" "$DENSE_MODULE" cuda "$N_G" "$N_Z" "$M" "$N_ITERS" "$N_WARMUP"

echo ""
echo "======================================================================"
echo "  n_g=5 N_z=$N_Z m=$M (profond, forme réaliste en profondeur de B&B)"
echo "======================================================================"
"$PROBE" "$SPARSE_MODULE" "$DENSE_MODULE" cuda 5 "$N_Z" "$M" "$N_ITERS" "$N_WARMUP"

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
