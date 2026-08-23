#!/bin/bash
# run_ref_array.sh (Trillium) — mode ref (squelette nu, sans borne) sur les
# 120 instances de test_paper_trillium/, une tâche d'array par instance
# (test_paper_trillium/instances_list.txt, format : n N m seed path).
#
# PAS ENCORE EXÉCUTÉ (préparé le 2026-08-16, en attendant le go — objectif :
# avoir un ref propre à Trillium, comparable au même matériel que
# gnn_batched, au lieu de comparer contre le ref Rorqual qui a un CPU/scheduler
# différent). Ne pas soumettre sans confirmation explicite.
#
# Différences avec l'original Rorqual (test_paper/run_ref_array.sh) :
#   - --chdir : /project/def-qcappart (pas /project/6062881, Rorqual).
#   - --gpus-per-node=1 AJOUTÉ bien que ref n'utilise jamais le GPU : sur
#     Trillium, --gpus-per-node est OBLIGATOIRE sur CHAQUE job (confirmé
#     empiriquement — sbatch refuse même un job de compilation CPU pur sans
#     cette option). Impact attendu négligeable (ref est CPU-bound ; le GPU
#     alloué reste inutilisé), mais TRES_PER_NODE facturera quand même 1 GPU.
#   - pas de --mem (interdit sur Trillium).
#   - logs/résultats écrits dans $SCRATCH (PAS test_paper_trillium/{logs,results},
#     /project étant en LECTURE SEULE sur les nœuds de calcul Trillium) — à
#     rapatrier après coup depuis le login node.
#   - CORRECTIF 2026-08-17 (après échec des 76 premières tâches du premier essai,
#     job 785349, "libgecodeint.so.49: cannot open shared object file") :
#     solver_bab/setup_gpu_env.sh EST bien nécessaire même en mode ref — pas
#     pour CUDA (ref n'appelle jamais le GNN), mais parce que ce script est
#     aussi ce qui met external/gecode/lib et external/ortools/lib dans
#     LD_LIBRARY_PATH, dont bab_mkp_gecode a besoin pour TOUT mode (un seul
#     binaire, mêmes libs liées). Pas d'activation de venv en revanche :
#     seul parse_result.py (stdlib-only) tourne en Python, python3 système
#     suffit.
#
# Soumettre (PAS FAIT — attend confirmation) :
#   sbatch --array=1-120 test_paper_trillium/run_ref_array.sh
# Rapatrier les résultats depuis $SCRATCH vers le repo après coup (login node) :
#   cp /scratch/romanlen/test_paper_trillium_ref/results/ref_*.csv test_paper_trillium/results/

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/def-qcappart/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=test_paper_trillium_ref
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/romanlen/test_paper_trillium_ref/slurm_logs/slurm_%x_%A_%a.out
#SBATCH --error=/scratch/romanlen/test_paper_trillium_ref/slurm_logs/slurm_%x_%A_%a.err

set -euo pipefail

REPO="/project/def-qcappart/romanlen/Learning_bounds_CCCP"
cd "$REPO"

module load StdEnv/2023 gcc/12.3
source solver_bab/setup_gpu_env.sh

TESTDIR="$REPO/test_paper_trillium"
SCRATCH_OUT="/scratch/romanlen/test_paper_trillium_ref"
mkdir -p "$SCRATCH_OUT/logs" "$SCRATCH_OUT/results" "$SCRATCH_OUT/slurm_logs"

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TESTDIR/instances_list.txt")
read -r N_ITEMS N_SCEN M SEED RELPATH <<< "$LINE"
DATASET="$TESTDIR/$RELPATH"

if [[ ! -f "$DATASET" ]]; then
    echo "[ERREUR] instance introuvable : $DATASET" >&2
    exit 1
fi

BINARY="$REPO/solver_bab/build/bab_mkp_gecode"
EPS=0.10
T_ROOT=200
T_NODE=50
TIME_LIM_S=0   # illimité — mur SLURM (24h) comme seul filet de sécurité

LOGFILE="$SCRATCH_OUT/logs/ref_n${N_ITEMS}_N${N_SCEN}_seed${SEED}.log"
RESULT_CSV="$SCRATCH_OUT/results/ref_n${N_ITEMS}_N${N_SCEN}_seed${SEED}.csv"

echo "======================================================================" | tee "$LOGFILE"
echo "  test_paper_trillium ref — n=$N_ITEMS N=$N_SCEN m=$M seed=$SEED — $(date)" | tee -a "$LOGFILE"
echo "  Job : ${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}   Nœud : $(hostname)" | tee -a "$LOGFILE"
echo "======================================================================" | tee -a "$LOGFILE"

"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" ref 2>&1 | tee -a "$LOGFILE"

python3 "$TESTDIR/parse_result.py" ref "$N_ITEMS" "$N_SCEN" "$M" "$SEED" "$LOGFILE" > "$RESULT_CSV"

echo "" | tee -a "$LOGFILE"
echo "Terminé — $(date)" | tee -a "$LOGFILE"
