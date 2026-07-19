#!/bin/bash
# compare_holdout_multiinstance.sh — Comparaison à 4 méthodes de borne (ref, v1ld,
# gnn, v1cld) sur des instances CCMKP HELD-OUT, n∈{10,20}, N∈{20,50,100}, sur GPU.
#
# Contexte : les jobs compare_ref_v1ld_v1cld_gnn_n{10,20}.sh utilisaient tous
# l'instance seed=42 (défaut du générateur). Or data/train_base_s10/seeds_done_*
# montre que seed=42 fait partie du jeu d'entraînement du GNN (seed_start=0,
# ~10000 instances, pour n=10 ET n=20, toutes tailles N) — donc ce n'est PAS un
# test hors-échantillon fiable pour le gap GNN vs v1ld/v1cld.
#
# Ce job répète le même sweep sur 5 instances neuves, seeds 50001..50005
# (hors de [0,9999], jamais vues à l'entraînement — vérifié sur tous les
# seeds_done_n{10,20}_m5_N{20,50,100}.txt, max=9999). Données déjà générées
# via create_database_knapsack_pisinger.py (--n 10 20 --m 5 --alpha 0.5
# --correlation weakly_correlated --S_sizes 20 50 100 --n_reps 1 --seed <seed>).
#
# Un seed par tâche de array (5 tâches), chaque tâche fait le sweep complet
# n∈{10,20} × N∈{20,50,100} × 4 modes — même structure que
# compare_ref_v1ld_v1cld_gnn_n20.sh (dont le sweep à 1 instance a pris ~57min,
# dominé par v1ld@N=100 ~19min). n=10 est négligeable en comparaison (~20s).
#
# Soumettre :
#   sbatch compare_holdout_multiinstance.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=cmp_holdout
#SBATCH --array=1-5
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_%x_%A_%a.out
#SBATCH --error=logs/slurm_%x_%A_%a.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

SEED=$((50000 + SLURM_ARRAY_TASK_ID))

CKPT="gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
EPS=0.10
T_ROOT=200
T_NODE=50
TIME_LIM_S=1800   # par run (mode, N) — cf. compare_ref_v1ld_v1cld_gnn_n20.sh

VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export LD_LIBRARY_PATH="$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

BINARY="$REPO/solver_bab/build/bab_mkp_gecode"

echo "======================================================================"
echo "  compare_holdout_multiinstance — $(date)"
echo "  Job : ${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}   Nœud : $(hostname)"
echo "  Instance held-out : seed=$SEED  (hors training, cf. en-tête du script)"
echo "  Checkpoint GNN : $CKPT"
echo "  eps=$EPS  T_root=$T_ROOT  T_node=$T_NODE  time_limit=${TIME_LIM_S}s/run"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for N_VAL in 10 20; do
    for N in 20 50 100; do
        DATASET="data/knapsack/datasets-pisinger/ccmk-song-pisinger-n${N_VAL}-m5-a50-we-r${SEED}-S${N}.txt"
        if [[ ! -f "$DATASET" ]]; then
            echo "  [ERREUR] fichier introuvable : $DATASET" >&2
            exit 1
        fi

        echo ""
        echo "######################################################################"
        echo "  seed=$SEED  n=${N_VAL}  N=${N}  —  $(date)"
        echo "######################################################################"

        echo ""
        echo "=== [seed=$SEED n=${N_VAL} N=${N}] MODE=ref (squelette nu, sans borne) ==="
        "$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" ref

        echo ""
        echo "=== [seed=$SEED n=${N_VAL} N=${N}] MODE=v1ld (sous-gradient, T_root=$T_ROOT, T_node=$T_NODE) ==="
        "$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" v1ld

        echo ""
        echo "=== [seed=$SEED n=${N_VAL} N=${N}] MODE=gnn (backend=ipc, variant=sd_edit, device=cuda) ==="
        "$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" gnn non 0.1 "$CKPT" cuda python3 ipc sd_edit

        echo ""
        echo "=== [seed=$SEED n=${N_VAL} N=${N}] MODE=v1cld (sous-gradient, T_root=$T_ROOT, T_node=$T_NODE) ==="
        "$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" v1cld
    done
done

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
