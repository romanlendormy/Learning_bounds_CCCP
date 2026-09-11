#!/bin/bash
# run_gnn_array.sh (Trillium) — mode gnn_batched (K=64, zeta=surrogate,
# stride=1, cumul complet, backend native, variant sd_edit) sur les 120
# instances de test_paper/mkp/trillium/sld/, une tâche d'array par instance
# (test_paper/mkp/trillium/sld/instances_list.txt, format : n N m seed path).
#
# Adaptation Trillium de la campagne équivalente Rorqual — déjà
# EXÉCUTÉE AVEC SUCCÈS le 2026-08-16 (job 776433, 120/120 COMPLETED). Voir
# test_paper/mkp/trillium/sld/results/gnn_*.csv pour les résultats déjà produits —
# ne pas relancer sans raison (recalcule 120 jobs GPU pour rien).
#
# Différences avec l'original Rorqual :
#   - --chdir : /project/def-qcappart (pas /project/6062881, Rorqual).
#   - --gpus-per-node=1 au lieu de --gres=gpu:h100:1 (syntaxe --gres refusée
#     par le slurm.conf de Trillium — ET --gpus-per-node est OBLIGATOIRE sur
#     CHAQUE job Trillium, même sans usage réel du GPU).
#   - pas de --mem (interdit sur Trillium : mémoire déduite du nb de GPU,
#     186GiB/GPU ici).
#   - logs/résultats écrits dans $SCRATCH (PAS test_paper_trillium/{logs,results},
#     sous /project qui est en LECTURE SEULE sur les nœuds de calcul Trillium,
#     contrairement à Rorqual) — copiés vers le repo après coup depuis le
#     login node (cf. commande de rapatriement en bas de fichier).
#   - module load StdEnv/2023 gcc/12.3 AVANT de sourcer setup_gpu_env.sh :
#     requis pour que `module load cuda/12.6` résolve (dépendance Lmod
#     hiérarchique non satisfaite par défaut dans l'environnement d'un job
#     sbatch sur Trillium, contrairement à une session interactive login node).
#   - pas d'activation de venv complet : parse_result.py est stdlib-only.
#     ATTENTION : backend=native N'EST PAS 100% sans Python pour autant —
#     ensure_scripted_batched_export() (gnn_sd_edit_scriptable_batched_export.hpp)
#     fork un python3 UNE SEULE FOIS par checkpoint pour produire
#     <ckpt>.scripted_batched.pt (TorchScript), puis réutilise ce cache (check
#     mtime) sans jamais rappeler Python. Ce cache a déjà été PRÉ-GÉNÉRÉ (venv
#     minimal torch==2.13.0 seul, wheelhouse Alliance Canada, cf.
#     /project/def-qcappart/romanlen/venv_export_only) — sans lui, chaque
#     tâche échouerait (ModuleNotFoundError: torch, python3 système n'a pas
#     torch). Si le checkpoint change, refaire cet export AVANT de resoumettre.
#
# Soumettre (déjà fait, pour référence) :
#   sbatch --array=1-120 test_paper/mkp/trillium/sld/run_gnn_array.sh
# Rapatrier les résultats depuis $SCRATCH vers le repo après coup (login node) :
#   cp /scratch/romanlen/test_paper_trillium_gnn/results/gnn_*.csv test_paper/mkp/trillium/sld/results/

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/def-qcappart/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=test_paper_trillium_gnn
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/romanlen/test_paper_trillium_gnn/slurm_logs/slurm_%x_%A_%a.out
#SBATCH --error=/scratch/romanlen/test_paper_trillium_gnn/slurm_logs/slurm_%x_%A_%a.err

set -euo pipefail

REPO="/project/def-qcappart/romanlen/Learning_bounds_CCCP"
cd "$REPO"

module load StdEnv/2023 gcc/12.3
source solver_bab/setup_gpu_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

TESTDIR="$REPO/test_paper/mkp/trillium/sld"
SCRATCH_OUT="/scratch/romanlen/test_paper_trillium_gnn"
mkdir -p "$SCRATCH_OUT/logs" "$SCRATCH_OUT/results" "$SCRATCH_OUT/slurm_logs"

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TESTDIR/instances_list.txt")
read -r N_ITEMS N_SCEN M SEED RELPATH <<< "$LINE"
DATASET="$TESTDIR/$RELPATH"

if [[ ! -f "$DATASET" ]]; then
    echo "[ERREUR] instance introuvable : $DATASET" >&2
    exit 1
fi

BINARY="$REPO/solver_bab/build/bab_mkp_gecode"
CKPT="gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
EPS=0.10
T_ROOT=200
T_NODE=50
TIME_LIM_S=0   # illimité — mur SLURM (24h) comme seul filet de sécurité
K=64

LOGFILE="$SCRATCH_OUT/logs/gnn_n${N_ITEMS}_N${N_SCEN}_seed${SEED}.log"
RESULT_CSV="$SCRATCH_OUT/results/gnn_n${N_ITEMS}_N${N_SCEN}_seed${SEED}.csv"

{
echo "======================================================================"
echo "  test_paper_trillium gnn_batched — n=$N_ITEMS N=$N_SCEN m=$M seed=$SEED — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}   Nœud : $(hostname)"
echo "  K=$K  zeta_method=surrogate  gnn_node_stride=1  backend=native"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
} | tee "$LOGFILE"

"$BINARY" "$DATASET" "$EPS" "$T_NODE" "$TIME_LIM_S" 0 "$T_ROOT" gnn_batched \
  non 0.1 "$CKPT" cuda python3 ipc sd_edit 0.0 0.0 1.0 "$K" 1.0 surrogate 1 native \
  2>&1 | tee -a "$LOGFILE"

python3 "$TESTDIR/parse_result.py" gnn_batched "$N_ITEMS" "$N_SCEN" "$M" "$SEED" "$LOGFILE" > "$RESULT_CSV"

echo "" | tee -a "$LOGFILE"
echo "Terminé — $(date)" | tee -a "$LOGFILE"
