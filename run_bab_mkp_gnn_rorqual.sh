#!/bin/bash
# run_bab_mkp_gnn_rorqual.sh — bab_mkp_gecode en mode borne GNN one-shot
# (bound_mode=GNN, cf. solver_bab/mkp/gecode_bab.hpp), à comparer aux modes
# v1ld/v1cld (descente de sous-gradient) sur la même instance.
#
# Le binaire lance un serveur Python persistant (gnn/gnn_inference_server.py)
# via pipe()+fork()+execvp() — la venv doit donc être active dans l'environnement
# du job (c'est le cas ici, activée avant le lancement du binaire) pour que
# "python3" du PATH résolve vers l'interpréteur qui a torch/torch_geometric.
#
# Soumettre :
#   sbatch run_bab_mkp_gnn_rorqual.sh
#
# Varier l'instance / le mode de comparaison :
#   sbatch --export=ALL,MODE=v1ld                    run_bab_mkp_gnn_rorqual.sh
#   sbatch --export=ALL,DATASET=...,EPS=0.1           run_bab_mkp_gnn_rorqual.sh

# ── Ressources SLURM ──────────────────────────────────────────────────────────
#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=bab_mkp_gnn
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

# ── Paramètres ────────────────────────────────────────────────────────────────
# Dataset au format "song" (colonnes : idx, idx, m, n, rep, A[m*n], b[m], [c[n] sur i==0]).
# Doit correspondre au m entraîné dans le checkpoint (m=5 pour CKPT par défaut) —
# aucun dataset "song" du dépôt n'a m=5/n=10 à ce jour : fournir le sien via
# --export=ALL,DATASET=...
DATASET="${DATASET:-}"
EPS="${EPS:-0.10}"
T_NODE="${T_NODE:-30}"
TIME_LIM="${TIME_LIM:-120}"
REP="${REP:-0}"
T_ROOT="${T_ROOT:-200}"
MODE="${MODE:-gnn}"                 # gnn | v1ld | v1cld — pour comparer sur la même instance
CKPT="${CKPT:-gnn/runs/train_scaled/ckpt_scaled_n10_ep500_lr0.0003_bs32_h128_seed42_v6_n10_500ep_ep0500.pt}"
GNN_DEVICE="${GNN_DEVICE:-cpu}"

# ── Chemins ───────────────────────────────────────────────────────────────────
REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
BINARY="$REPO/solver_bab/build/bab_mkp_gecode"
VENV_PATH="$HOME/envs/learning_bounds"

cd "$REPO"   # gnn_v1ld_client spawn "python3 gnn/gnn_inference_server.py" relatif au cwd

if [ -z "$DATASET" ]; then
    echo "ERREUR : DATASET non défini — passer --export=ALL,DATASET=<chemin> (format song, m/n compatibles avec CKPT)."
    exit 1
fi

# ── Modules & venv (nécessaire même en mode v1ld/v1cld pour rester comparable) ─
module load StdEnv/2023 gcc/12.3.0 python/3.11.5 2>/dev/null || \
    module load python/3.11 2>/dev/null || \
    echo "  (aucun module python chargé)"

if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "ERREUR : venv introuvable à $VENV_PATH"
    exit 1
fi

export LD_LIBRARY_PATH="$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

mkdir -p "$REPO/logs"

echo "======================================================================"
echo "  bab_mkp_gnn — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  Dataset=$DATASET  eps=$EPS  T_node=$T_NODE  time_lim=${TIME_LIM}s  mode=$MODE"
[ "$MODE" = "gnn" ] && echo "  Checkpoint=$CKPT  device=$GNN_DEVICE"
echo "======================================================================"

if [ ! -x "$BINARY" ]; then
    echo "ERREUR : binaire introuvable : $BINARY"
    echo "  → cmake --build solver_bab/build --target bab_mkp_gecode"
    exit 1
fi

echo ""
"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM" "$REP" "$T_ROOT" "$MODE" non 0.1 "$CKPT" "$GNN_DEVICE" python3

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
