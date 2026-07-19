#!/bin/bash
# run_ablation_N_rorqual.sh — Une config de l'ablation multi-N (voir le protocole
# "Expérience : ablation multi-N sur le GNN V1LD" pour le détail des 7 configs).
#
# Chaque config se soumet via --export=ALL avec CONFIG_NAME + les overrides
# nécessaires. Paramètres communs fixés par défaut (n_list=10, max_depth=4,
# n_epochs=500, seed=42, use_max_pool=1, center_lam=1, max_norm=2.0,
# warmup_max_n=0 — ce dernier est CRITIQUE, voir le piège documenté dans le
# protocole : sans lui, N=200/500 sont exclus de la référence T=50).
#
# Exemples (les 7 configs du protocole) :
#   sbatch --export=ALL,CONFIG_NAME=1a_single_N500,N_SCENS=500,N_GRAPHS=1000 \
#       run_ablation_N_rorqual.sh
#   sbatch --export=ALL,CONFIG_NAME=1b_single_N50,N_SCENS=50,N_GRAPHS=1000 \
#       run_ablation_N_rorqual.sh
#   sbatch --export=ALL,CONFIG_NAME=2_multiN_baseline,N_SCENS="50 200 500",N_GRAPHS=3000 \
#       run_ablation_N_rorqual.sh
#   sbatch --export=ALL,CONFIG_NAME=3_multiN_no_warmup,N_SCENS="50 200 500",N_GRAPHS=3000,N_WARMUP=0 \
#       run_ablation_N_rorqual.sh
#   sbatch --export=ALL,CONFIG_NAME=4_multiN_no_thetaattn,N_SCENS="50 200 500",N_GRAPHS=3000,USE_THETA_ATTN=0 \
#       run_ablation_N_rorqual.sh
#   sbatch --export=ALL,CONFIG_NAME=5_multiN_no_film,N_SCENS="50 200 500",N_GRAPHS=3000,USE_FILM=0 \
#       run_ablation_N_rorqual.sh
#   sbatch --export=ALL,CONFIG_NAME=6_multiN_no_cscale,N_SCENS="50 200 500",N_GRAPHS=3000,NO_CSCALE_NORM=1 \
#       run_ablation_N_rorqual.sh
#   sbatch --export=ALL,CONFIG_NAME=7_multiN_no_attn_no_film,N_SCENS="50 200 500",N_GRAPHS=3000,USE_THETA_ATTN=0,USE_FILM=0 \
#       run_ablation_N_rorqual.sh

#SBATCH --account=def-qcappart_gpu
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=ablation_N
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
# GPU obligatoire : forward+backward mesuré à 6.26s/instance N=500 sur CPU
# vs 0.048s sur H100 (~130x). Sur CPU, 1000 instances N=500 x 500 epochs
# aurait pris des semaines ; sur GPU, quelques heures.

set -euo pipefail

# ── Paramètres — défauts = baseline (config 2) ────────────────────────────────
CONFIG_NAME="${CONFIG_NAME:?CONFIG_NAME est obligatoire (ex: 2_multiN_baseline)}"
N_SCENS="${N_SCENS:-50 200 500}"
N_GRAPHS="${N_GRAPHS:-3000}"
N_EPOCHS="${N_EPOCHS:-500}"
MAX_DEPTH="${MAX_DEPTH:-4}"
SEED="${SEED:-42}"
EVAL_EVERY="${EVAL_EVERY:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-3e-4}"
MAX_NORM="${MAX_NORM:-2.0}"
HIDDEN="${HIDDEN:-64}"
MAX_LOAD="${MAX_LOAD:-50000}"
# Restreint le nombre d'instances de base distinctes (avant split train/eval) par
# groupe N. 0 = pas de restriction. Sert à isoler l'effet de la diversité
# d'instances du nombre de graphes d'entraînement (n_graphs).
MAX_INSTANCES="${MAX_INSTANCES:-0}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-0}"

USE_MAX_POOL="${USE_MAX_POOL:-1}"
USE_THETA_ATTN="${USE_THETA_ATTN:-1}"
USE_FILM="${USE_FILM:-1}"
CENTER_LAM="${CENTER_LAM:-1}"
NO_CSCALE_NORM="${NO_CSCALE_NORM:-0}"

N_WARMUP="${N_WARMUP:-10}"
WARMUP_LR="${WARMUP_LR:-1e-3}"
# CRITIQUE : 0 = pas de filtre. Sans ça, la référence T=50 (et le warmup si
# activé) exclurait silencieusement les N > warmup_max_n (défaut natif = 100).
WARMUP_MAX_N="${WARMUP_MAX_N:-0}"

SKIP_SUBGRAD="${SKIP_SUBGRAD:-0}"

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
VENV_PATH="$HOME/envs/learning_bounds"
OUT_DIR="${OUT_DIR:-$REPO/gnn/runs/ablation_N/$CONFIG_NAME}"

echo "======================================================================"
echo "  Ablation multi-N — config: $CONFIG_NAME — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  N_SCENS=$N_SCENS  N_GRAPHS=$N_GRAPHS  N_EPOCHS=$N_EPOCHS  MAX_DEPTH=$MAX_DEPTH"
echo "  USE_THETA_ATTN=$USE_THETA_ATTN  USE_FILM=$USE_FILM  N_WARMUP=$N_WARMUP  WARMUP_MAX_N=$WARMUP_MAX_N"
echo "  NO_CSCALE_NORM=$NO_CSCALE_NORM"
echo "  OUT_DIR : $OUT_DIR"
echo "======================================================================"

module load StdEnv/2023 gcc/12.3.0 python/3.11.5 2>/dev/null || \
    module load python/3.11 2>/dev/null || \
    echo "  (aucun module python chargé)"

source "$VENV_PATH/bin/activate"

export LD_LIBRARY_PATH="$REPO/external/gecode/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# ── Données → SLURM_TMPDIR (merge multi-N si besoin) ─────────────────────────
DATA_LOCAL="$SLURM_TMPDIR/data/merged"
mkdir -p "$DATA_LOCAL"
python "$REPO/collect/merge_datasets.py" \
    --n_scens  $N_SCENS            \
    --n_val    10                  \
    --m_val    5                   \
    --data_root "$REPO/data"       \
    --out_dir  "$DATA_LOCAL"

mkdir -p "$OUT_DIR" "$REPO/logs"

# ── Flags optionnels ──────────────────────────────────────────────────────────
MAX_POOL_FLAG=""
[ "$USE_MAX_POOL" = "1" ] && MAX_POOL_FLAG="--use_max_pool"
THETA_ATTN_FLAG=""
[ "$USE_THETA_ATTN" = "1" ] && THETA_ATTN_FLAG="--use_theta_attn"
FILM_FLAG=""
[ "$USE_FILM" = "1" ] && FILM_FLAG="--use_film"
CENTER_LAM_FLAG=""
[ "$CENTER_LAM" = "1" ] && CENTER_LAM_FLAG="--center_lam"
NO_CSCALE_FLAG=""
[ "$NO_CSCALE_NORM" = "1" ] && NO_CSCALE_FLAG="--no_c_scale_norm"
SUBGRAD_FLAG=""
[ "$SKIP_SUBGRAD" = "1" ] && SUBGRAD_FLAG="--skip_subgrad"
STRATIFY_N_FLAG="--stratify_N"
MAX_INST_ARG=""
[ "$MAX_INSTANCES" != "0" ] && MAX_INST_ARG="--max_instances $MAX_INSTANCES"
SAVE_CKPT_FLAG=""
[ "$SAVE_CHECKPOINT" = "1" ] && SAVE_CKPT_FLAG="--save_checkpoint"

python "$REPO/gnn/diag_overfit_100.py" \
    --data_dir      "$DATA_LOCAL"   \
    --out_dir       "$OUT_DIR"      \
    --n_list        10              \
    --n_graphs      "$N_GRAPHS"     \
    --n_epochs      "$N_EPOCHS"     \
    --eval_every    "$EVAL_EVERY"   \
    --eval_set_every "$EVAL_EVERY"  \
    --batch_size    "$BATCH_SIZE"   \
    --lr            "$LR"           \
    --max_depth     "$MAX_DEPTH"    \
    --max_norm      "$MAX_NORM"     \
    --n_warmup      "$N_WARMUP"     \
    --warmup_lr     "$WARMUP_LR"    \
    --warmup_max_n  "$WARMUP_MAX_N" \
    --seed          "$SEED"         \
    --hidden        "$HIDDEN"       \
    --max_load      "$MAX_LOAD"     \
    $MAX_INST_ARG                   \
    --device        cuda            \
    $SUBGRAD_FLAG                   \
    $SAVE_CKPT_FLAG                 \
    $MAX_POOL_FLAG                  \
    $THETA_ATTN_FLAG                \
    $FILM_FLAG                      \
    $CENTER_LAM_FLAG                \
    $NO_CSCALE_FLAG                 \
    $STRATIFY_N_FLAG

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "  Résultats dans : $OUT_DIR"
echo "======================================================================"
