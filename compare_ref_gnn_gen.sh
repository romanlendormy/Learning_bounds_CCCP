#!/bin/bash
# compare_ref_gnn_gen.sh — REF (squelette nu) vs GNN natif (device=cuda) sur
# une instance générée à la volée (même distribution que la collecte
# collect_s10 : generate_pisinger_instance + generate_scenario, cf.
# collect/collect_large_base_samples.py::generate_song_tempfile), au lieu du
# fichier figé de Song et al. utilisé par compare_ref_gnn_song.sh.
#
# Détaille, comme compare_ref_gnn_song.sh, la décomposition du temps
# d'inférence GNN (construction de graphe / forward / calcul de borne) —
# déjà imprimée par le binaire lui-même en mode gnn.
#
# Soumettre :
#   sbatch compare_ref_gnn_gen.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=cmp_ref_gnn_gen
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

# ── Paramètres de l'instance générée (identiques à compare_ref_gnn_song.sh
#    par défaut : n=20/m=5/N=100, distribution Song via Pisinger) ────────────
N_ITEMS=10
M=5
N_SCEN=20
EPSILON=0.10
ALPHA=0.5
R_PARAM=1000
CORRELATION=weakly_correlated
SEED=0

CKPT="gnn/runs/train_scaled/ckpt_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_10k_ep0197.pt"
T_ROOT=200
T_NODE=50
TIME_LIM_S=3300

VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export LD_LIBRARY_PATH="$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

BINARY="$REPO/solver_bab/build/bab_mkp_gecode"

echo "======================================================================"
echo "  compare_ref_gnn_gen — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  Instance générée : n=$N_ITEMS  m=$M  N=$N_SCEN  eps=$EPSILON"
echo "    alpha=$ALPHA  R=$R_PARAM  correlation=$CORRELATION  seed=$SEED"
echo "  T_root=$T_ROOT  T_node=$T_NODE  time_limit=${TIME_LIM_S}s"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "=== Génération de l'instance (collect_s10-like) ==="
DATASET=$(python3 "$REPO/collect/gen_bench_instance.py" \
    --n "$N_ITEMS" --m "$M" --N "$N_SCEN" --seed "$SEED" \
    --alpha "$ALPHA" --R "$R_PARAM" --correlation "$CORRELATION")
trap 'rm -f "$DATASET"' EXIT
echo "  Fichier : $DATASET"

echo ""
echo "=== MODE=ref (squelette nu, sans borne) ==="
"$BINARY" "$DATASET" "$EPSILON" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" ref

echo ""
echo "=== MODE=gnn (backend=native, device=cuda) ==="
"$BINARY" "$DATASET" "$EPSILON" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" gnn non 0.1 "$CKPT" cuda python3 native

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
