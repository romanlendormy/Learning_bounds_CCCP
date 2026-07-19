#!/bin/bash
# bench_gnn_forward_probe.sh — Mesure dédiée du coût forward GNN (backend
# natif, GNNV1LDNative::query()) isolée du B&B, avec warm-up explicite.
#
# Objectif : distinguer le coût "à froid" (init CUDA : contexte, premiers
# lancements de kernels, cache cuBLAS) du régime stable, pour vérifier si le
# chiffre 29ms/appel observé sur compare_ref_gnn_gen.sh (n=10/N=20, 188
# appels seulement) est un artefact de dilution sur trop peu d'appels, ou un
# vrai coût — cf. analyse dans la mémoire du projet
# (project_gnn_inference_forward_cost_benchmark).
#
# gnn_native_probe appelle query() à la racine (x_mask tout libre) : n_g = n,
# donc le graphe est à sa taille MAXIMALE — le chiffre mesuré est un majorant
# conservateur du coût par nœud réel (les nœuds profonds ont n_g < n).
#
# Soumettre :
#   sbatch bench_gnn_forward_probe.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=bench_gnn_probe
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export LD_LIBRARY_PATH="$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

CKPT="gnn/runs/train_scaled/ckpt_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_10k_ep0197.pt"
EPSILON=0.10
N_ITERS=2000
N_WARMUP=50

echo "======================================================================"
echo "  bench_gnn_forward_probe — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "=== (Re)compilation gnn_native_probe ==="
cmake --build solver_bab/build --target gnn_native_probe -j "${SLURM_CPUS_PER_TASK:-4}"
PROBE="$REPO/solver_bab/build/gnn_native_probe"

echo ""
echo "=== Génération des instances (n=10/N=20 et n=20/N=100, seed=0) ==="
DATASET_N10_N20=$(python3 "$REPO/collect/gen_bench_instance.py" \
    --n 10 --m 5 --N 20 --seed 0 --alpha 0.5 --R 1000 --correlation weakly_correlated)
DATASET_N20_N100=$(python3 "$REPO/collect/gen_bench_instance.py" \
    --n 20 --m 5 --N 100 --seed 0 --alpha 0.5 --R 1000 --correlation weakly_correlated)
DATASET_N20_N50="$REPO/data/knapsack/datasets-pisinger/ccmk-song-pisinger-n20-m5-a50-we-r42-S50.txt"
trap 'rm -f "$DATASET_N10_N20" "$DATASET_N20_N100"' EXIT

echo "  n=10/N=20  : $DATASET_N10_N20"
echo "  n=20/N=50  : $DATASET_N20_N50 (fichier existant)"
echo "  n=20/N=100 : $DATASET_N20_N100"

echo ""
echo "======================================================================"
echo "  n=10 / N=20 / device=cuda"
echo "======================================================================"
"$PROBE" "$DATASET_N10_N20" "$CKPT" "$EPSILON" cuda 0 "$N_ITERS" "$N_WARMUP"

echo ""
echo "======================================================================"
echo "  n=20 / N=50 / device=cuda"
echo "======================================================================"
"$PROBE" "$DATASET_N20_N50" "$CKPT" "$EPSILON" cuda 0 "$N_ITERS" "$N_WARMUP"

echo ""
echo "======================================================================"
echo "  n=20 / N=50 / device=cpu"
echo "======================================================================"
"$PROBE" "$DATASET_N20_N50" "$CKPT" "$EPSILON" cpu 0 "$N_ITERS" "$N_WARMUP"

echo ""
echo "======================================================================"
echo "  n=20 / N=100 / device=cuda"
echo "======================================================================"
"$PROBE" "$DATASET_N20_N100" "$CKPT" "$EPSILON" cuda 0 "$N_ITERS" "$N_WARMUP"

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
