#!/bin/bash
# gnn/ccpo/run_train_ccpo.sh — Entraînement du GNN CCPO de l'article (models/ccpo/) :
# features v3 + poids d'arêtes façon CCMKP, hidden 128, lr 3e-4 (montée 2 epochs puis cosinus → 1e-5),
# batch 32, max_norm 0,5 (clip par exemple), perte abs, préchauffage supervisé (300 exemples, T = 30,
# 10 epochs), 200 epochs. Train : ~8 000 nœuds sélectionnés, moins les nœuds au sous-arbre prouvé
# infaisable (--exclude_nodes), plus les nœuds faisables complémentaires (--extra_nodes_dir) ; VAL fixe.
# CPU seul (32 évaluations du Lagrangien en parallèle × 12 threads OpenMP) : ~2,6 min par epoch.
# Reprise automatique depuis le dernier checkpoint : resoumettre la même commande.
#
# Données (cf. gnn/ccpo/README.md) :
#   DATA_DIR  jeu curaté à drapeaux exacts (collect/ccpo/recompute_presolve_flags.py)
#   LABELS    étiquettes de faisabilité des sous-arbres (gnn/ccpo/find_infeasible_nodes.py)
#   EXTRA     nœuds faisables complémentaires (collect/ccpo/build_extra_feasible_nodes.py)
# Soumettre depuis la racine du dépôt (nœud de calcul CPU entier) :
#   sbatch --export=ALL,DATA_DIR=...,LABELS=...,EXTRA=...,OUT_DIR=... gnn/ccpo/run_train_ccpo.sh
#   (SMOKE=1 : run minuscule de contrôle)

#SBATCH --account=def-qcappart
#SBATCH --job-name=train_ccpo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=192
#SBATCH --time=16:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -u
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"
source env/modules_trillium.sh
VENV_PATH="${VENV_PATH:-$HOME/venvs/cccp}"
source "$VENV_PATH/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_WAIT_POLICY=passive

: "${DATA_DIR:?DATA_DIR requis}" "${LABELS:?LABELS requis}" "${EXTRA:?EXTRA requis}"
OUT_DIR="${OUT_DIR:-runs/ccpo}"
SMOKE="${SMOKE:-0}"; N_EP="${N_EP:-200}"
WARM="--n_warmup 10 --warmup_lr 1e-3 --warmup_t 30 --warmup_max_n 20 --warmup_n_examples 300"
if [[ "$SMOKE" == 1 ]]; then
  OUT_DIR="$OUT_DIR/smoke"; WARM="$WARM --warmup_n_examples 8 --n_warmup 1"
  RUN_ARGS="--n_epochs 1 --lr_warmup_ep 0 --train_use_frac 0.02 --metrics_sample 20 --eval_sample 10"
else
  RUN_ARGS="--n_epochs $N_EP --lr_warmup_ep 2 --metrics_sample 300 --eval_sample 60 --keep_ckpts $N_EP"
fi
mkdir -p "$OUT_DIR"
echo "# train_ccpo — job ${SLURM_JOB_ID:-local} sur $(hostname) — $(date) — smoke=$SMOKE"

python3 -u gnn/ccpo/train_ccpo_gnn.py \
    --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" --run_name big6f \
    --target_graphs 10000 --root_frac 0.05 --train_frac 0.8 --seed 42 \
    --lr 3e-4 --eta_min 1e-5 --batch_size 32 --max_norm 0.5 --hidden 128 \
    --clip_mode per_example --loss_mode abs \
    --n_workers 32 --omp_threads 12 --device cpu \
    --eval_every 999 --metrics_every 1 \
    --intra_ckpt_every 5 --keep_every 10 \
    --feat_version 3 --edge_mode mkp --warmup_target_mode E0 $WARM \
    --exclude_nodes "$LABELS" --extra_nodes_dir "$EXTRA" $RUN_ARGS
echo "# fin (code $?) — $(date)"
