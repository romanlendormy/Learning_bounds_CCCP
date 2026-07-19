#!/bin/bash
# bench_gnn_edge_vectorized_probe.sh — Valide l'équivalence numérique ET
# mesure le gain de la construction vectorisée torch des arêtes du graphe GNN
# (templates précalculés + broadcast) vs la construction scalaire actuelle
# (boucles C++ + push_back, build_graph_cpp dans gnn_v1ld_native.cpp) —
# cf. décomposition "arêtes = 59% de la construction de graphe" (job
# 16028283) et discussion "héritage parent vs calcul torch" (l'héritage
# cross-nœud ne réduit pas la complexité ici, cf. gnn_edge_vectorized_probe.cpp).
#
# Sonde autonome : ne dépend d'aucun checkpoint/modèle (topologie pure).
#
# Soumettre :
#   sbatch bench_gnn_edge_vectorized_probe.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=bench_edge_vectorized
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

source solver_bab/setup_gpu_env.sh
VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

N=20
N_Z=100
N_ITERS=500

echo "======================================================================"
echo "  bench_gnn_edge_vectorized_probe — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "=== (Re)compilation gnn_edge_vectorized_probe ==="
cmake -S solver_bab -B solver_bab/build -DCMAKE_BUILD_TYPE=Release -DGNN_DEVICE=cuda >/dev/null
cmake --build solver_bab/build --target gnn_edge_vectorized_probe -j "${SLURM_CPUS_PER_TASK:-4}"
PROBE="$REPO/solver_bab/build/gnn_edge_vectorized_probe"

echo ""
echo "======================================================================"
echo "  device=cpu  (n=$N N=$N_Z n_iters=$N_ITERS)"
echo "======================================================================"
"$PROBE" "$N" "$N_Z" cpu "$N_ITERS"

echo ""
echo "======================================================================"
echo "  device=cuda  (n=$N N=$N_Z n_iters=$N_ITERS)"
echo "======================================================================"
"$PROBE" "$N" "$N_Z" cuda "$N_ITERS"

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
