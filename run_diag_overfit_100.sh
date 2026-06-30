#!/bin/bash
# run_diag_overfit_100.sh — Diagnostic overfit 100 graphes V1LD.
#
# Job CPU léger (~30-60 min). Le B&B tourne en C++ avec OMP_NUM_THREADS=CPUs.
#
# Soumettre :
#   sbatch run_diag_overfit_100.sh
#
# Options rapides (moins d'epochs, pas de sous-gradient) :
#   sbatch --export=ALL,N_EPOCHS=200,SKIP_SUBGRAD=1 run_diag_overfit_100.sh
#
# Varier N (nombre de scénarios) — liste espace-séparée :
#   sbatch --export=ALL,N_SCENS="50 100 200 500" run_diag_overfit_100.sh
# Retour à la version actuelle (N=500 seul) :
#   sbatch run_diag_overfit_100.sh  (N_SCENS=500 par défaut)

# ── Ressources SLURM ──────────────────────────────────────────────────────────
#SBATCH --account=def-qcappart
#SBATCH --chdir=/home/romanlen/projects/def-qcappart/romanl/Learning_bounds_CCCP
#SBATCH --job-name=diag_overfit100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

# ── Paramètres ────────────────────────────────────────────────────────────────
N_EPOCHS="${N_EPOCHS:-1000}"
EVAL_EVERY="${EVAL_EVERY:-10}"
# Fréquence d'évaluation sur le set d'éval (0 = même que EVAL_EVERY)
EVAL_SET_EVERY="${EVAL_SET_EVERY:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-3e-4}"
N_GRAPHS="${N_GRAPHS:-1000}"
MAX_DEPTH="${MAX_DEPTH:-4}"
SEED="${SEED:-42}"
# N (scénarios) à mélanger — liste espace-séparée. Retour à N=500 seul : laisser vide.
# Chaque N doit avoir data/train_N{N}/ (exception : N=500 → train_base_s30).
# Défaut : mélange uniforme N=100/200/500 (--stratify_N activé).
N_SCENS="${N_SCENS:-100 200 500}"
# Passer SKIP_SUBGRAD=1 pour sauter verify_v1ld (recommandé si temps limité)
SKIP_SUBGRAD="${SKIP_SUBGRAD:-0}"
# Clip par instance (0 = désactivé)
MAX_NORM="${MAX_NORM:-2.0}"
# Warmup supervisé : pré-entraîne le GNN sur λ* sous-gradient (L2) avant la phase Lagrangienne
N_WARMUP="${N_WARMUP:-100}"
WARMUP_LR="${WARMUP_LR:-1e-3}"
WARMUP_T="${WARMUP_T:-50}"
# N max des graphes utilisés dans le warmup (0 = tous). 100 évite la lenteur des N=500 (1.9M edges).
WARMUP_MAX_N="${WARMUP_MAX_N:-100}"
# Max pooling sur les scénarios : 1 = activé (recommandé), 0 = désactivé (v15)
USE_MAX_POOL="${USE_MAX_POOL:-1}"
# Attention θ_i pour le pool global (v20) : 1 = activé, 0 = désactivé
USE_THETA_ATTN="${USE_THETA_ATTN:-0}"
# FiLM conditioning après l'embed (v21) : 1 = activé, 0 = désactivé
USE_FILM="${USE_FILM:-0}"
# Scheduler LR : 0 = désactive CAWR (recommandé avec warmup)
LR_T0="${LR_T0:-0}"
# Facteur multiplicatif CAWR : ignoré si LR_T0=0
LR_T_MULT="${LR_T_MULT:-2}"
# Warmup Lagrangien + cosinus sans restarts : 10% des epochs (100/1000)
LR_WARMUP_EP="${LR_WARMUP_EP:-100}"
# LR constant (désactive tout scheduler) : 1 = activé, 0 = désactivé
NO_SCHEDULER="${NO_SCHEDULER:-0}"
# Dimension cachée du GNN
HIDDEN="${HIDDEN:-64}"
# Nb max de records à charger depuis le fichier mergé (fallback combiné, non-stratifié).
# Avec --stratify_N, les fichiers per-N sont chargés séparément (max 20k/groupe).
MAX_LOAD="${MAX_LOAD:-50000}"

# ── Chemins ───────────────────────────────────────────────────────────────────
REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
VENV_PATH="$HOME/envs/learning_bounds"
# Tag OUT_DIR basé sur les N utilisés (espaces → underscores)
N_SCENS_TAG=$(echo "$N_SCENS" | tr ' ' '_')
# Centrage des λ sur les scénarios (élimine la dérive de jauge) : 1 = activé, 0 = désactivé
CENTER_LAM="${CENTER_LAM:-1}"
OUT_DIR="${OUT_DIR:-$REPO/gnn/runs/diag_overfit_100_v25_evalset_N${N_SCENS_TAG}}"
ALPHA_L2="${ALPHA_L2:-0.0}"
BETA1="${BETA1:-0.9}"
LAM_MAX="${LAM_MAX:-0.0}"
# Poids sur la loss pour les nœuds à depth=0 (1.0 = désactivé)
DEPTH_W0="${DEPTH_W0:-1.0}"
# Convergence complète sous-gradient : 1 = activé (lent), 0 = désactivé
SUBGRAD_FULL="${SUBGRAD_FULL:-0}"
SUBGRAD_FULL_T_MAX="${SUBGRAD_FULL_T_MAX:-2000}"
SUBGRAD_FULL_PATIENCE="${SUBGRAD_FULL_PATIENCE:-30}"
SUBGRAD_FULL_PRECISION="${SUBGRAD_FULL_PRECISION:-0.1}"

echo "======================================================================"
echo "  Diagnostic overfit 100 — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  N_EPOCHS=$N_EPOCHS  EVAL_EVERY=$EVAL_EVERY  BATCH_SIZE=$BATCH_SIZE"
echo "  LR=$LR  N_GRAPHS=$N_GRAPHS  MAX_DEPTH=$MAX_DEPTH  N_SCENS=$N_SCENS  MAX_NORM=$MAX_NORM  STRATIFY_N=${STRATIFY_N:-1}"
echo "  N_WARMUP=$N_WARMUP  WARMUP_LR=$WARMUP_LR  WARMUP_T=$WARMUP_T  USE_MAX_POOL=$USE_MAX_POOL  USE_THETA_ATTN=$USE_THETA_ATTN  USE_FILM=$USE_FILM  LR_T0=$LR_T0  DEPTH_W0=$DEPTH_W0"
echo "  CPUs alloués : ${SLURM_CPUS_PER_TASK:-?}  (OMP_NUM_THREADS idem)"
echo "  OUT_DIR : $OUT_DIR"
echo "======================================================================"

# ── Modules & venv ────────────────────────────────────────────────────────────
module load StdEnv/2023 gcc/12.3.0 python/3.11.5 2>/dev/null || \
    module load python/3.11 2>/dev/null || \
    echo "  (aucun module python chargé)"

if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "ERREUR : venv introuvable à $VENV_PATH"
    exit 1
fi

export LD_LIBRARY_PATH="$REPO/external/gecode/lib:${LD_LIBRARY_PATH:-}"

# Le B&B C++ bénéficie des CPUs alloués sur nœud dédié
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# ── Données → SLURM_TMPDIR (merge multi-N si besoin) ─────────────────────────
echo ""
echo "=== Préparation données (N_SCENS=$N_SCENS) ==="
DATA_LOCAL="$SLURM_TMPDIR/data/merged"
mkdir -p "$DATA_LOCAL"

python "$REPO/collect/merge_datasets.py" \
    --n_scens  $N_SCENS            \
    --n_val    10                  \
    --m_val    5                   \
    --data_root "$REPO/data"       \
    --out_dir  "$DATA_LOCAL"

echo "  OK — $(wc -l < "$DATA_LOCAL/nodes_n10_m5.txt") nœuds fusionnés"

mkdir -p "$OUT_DIR" "$REPO/logs"

# ── Lancement ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Lancement — $(date) ==="

SUBGRAD_FLAG=""
[ "${SKIP_SUBGRAD:-0}" = "1" ] && SUBGRAD_FLAG="--skip_subgrad"

MAX_POOL_FLAG=""
[ "${USE_MAX_POOL:-1}" = "1" ] && MAX_POOL_FLAG="--use_max_pool"

THETA_ATTN_FLAG=""
[ "${USE_THETA_ATTN:-0}" = "1" ] && THETA_ATTN_FLAG="--use_theta_attn"

FILM_FLAG=""
[ "${USE_FILM:-0}" = "1" ] && FILM_FLAG="--use_film"

LR_T0_ARG=""
[ "${LR_T0:-0}" != "0" ] && LR_T0_ARG="--lr_T0 ${LR_T0} --lr_T_mult ${LR_T_MULT}"

LR_WARMUP_ARG=""
[ "${LR_WARMUP_EP:-0}" != "0" ] && LR_WARMUP_ARG="--lr_warmup_ep ${LR_WARMUP_EP}"

NO_SCHEDULER_FLAG=""
[ "${NO_SCHEDULER:-0}" = "1" ] && NO_SCHEDULER_FLAG="--no_scheduler"

SUBGRAD_FULL_FLAG=""
[ "${SUBGRAD_FULL:-0}" = "1" ] && SUBGRAD_FULL_FLAG="--subgrad_full"

CENTER_LAM_FLAG=""
[ "${CENTER_LAM:-1}" = "1" ] && CENTER_LAM_FLAG="--center_lam"

STRATIFY_N_FLAG=""
[ "${STRATIFY_N:-1}" = "1" ] && STRATIFY_N_FLAG="--stratify_N"

python "$REPO/gnn/diag_overfit_100.py" \
    --data_dir   "$DATA_LOCAL"  \
    --out_dir    "$OUT_DIR"     \
    --n_list     10             \
    --n_graphs   "$N_GRAPHS"    \
    --n_epochs   "$N_EPOCHS"    \
    --eval_every "$EVAL_EVERY"  \
    --eval_set_every "$EVAL_SET_EVERY" \
    --batch_size "$BATCH_SIZE"  \
    --lr         "$LR"          \
    --max_depth  "$MAX_DEPTH"   \
    --alpha_l2   "$ALPHA_L2"    \
    --beta1      "$BETA1"       \
    --lam_max    "$LAM_MAX"     \
    --max_norm   "$MAX_NORM"    \
    --n_warmup      "$N_WARMUP"      \
    --warmup_lr     "$WARMUP_LR"     \
    --warmup_max_n  "$WARMUP_MAX_N"  \
    --seed       "$SEED"        \
    --hidden     "$HIDDEN"      \
    --max_load   "$MAX_LOAD"   \
    --device     cpu            \
    --subgrad_full_T_max      "$SUBGRAD_FULL_T_MAX"      \
    --subgrad_full_patience   "$SUBGRAD_FULL_PATIENCE"   \
    --subgrad_full_precision  "$SUBGRAD_FULL_PRECISION"  \
    $SUBGRAD_FLAG               \
    $MAX_POOL_FLAG              \
    $THETA_ATTN_FLAG            \
    $FILM_FLAG                  \
    $LR_T0_ARG                  \
    $LR_WARMUP_ARG              \
    $NO_SCHEDULER_FLAG          \
    $SUBGRAD_FULL_FLAG          \
    $CENTER_LAM_FLAG            \
    $STRATIFY_N_FLAG

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "  Résultats dans : $OUT_DIR"
echo "======================================================================"
