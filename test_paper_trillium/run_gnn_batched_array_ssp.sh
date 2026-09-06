#!/bin/bash
# run_gnn_batched_array_ssp.sh — mode gnn_batched V1CLD (K=8, backend natif
# TorchScript, device cuda) sur les 270 instances de test_paper_trillium/ssp/,
# une tâche d'array par instance (instances_list_ssp.txt : E T N rep chemin_relatif).
#
# K=8 : valeur "de production" retenue le 20 août (sweep_ref_vs_gnn_batched_gpu.sh)
# précisément pour cette grille -- K=32 est plus rapide sur petites instances
# mais provoque des OOM CUDA à N=500 pour E>=6, ce qui touche directement
# cette grille (N va jusqu'à 500, E jusqu'à 8). Décision utilisateur du
# 2026-08-22 : garder K=8 partout plutôt que de risquer des échecs partiels.
#
# Deux binaires exécutés par tâche, dans le MÊME job (même allocation GPU,
# pas de job supplémentaire) :
#   1) test_gnn_v1cld_bab_batched -- le run PRINCIPAL (temps, nœuds, gap final,
#      répartition forward/graph_build/zeta_theta déjà imprimée par le binaire).
#   2) test_gnn_v1cld_bab (mode "gnn", séquentiel, PAS batché) -- run auxiliaire
#      COURT (time_lim=60s) dans le seul but de lire root_bound_value, imprimé
#      tel quel par ce binaire (capturé une seule fois, au tout premier nœud,
#      cf. gecode_bab_ssp.hpp:597 "is_first" -- une vraie borne à la racine,
#      pas un proxy). Ce binaire réutilise le même modèle exporté
#      (<ckpt>.scripted_batched.pt, unifié depuis la refonte du 20 août) que
#      le moteur batché -- même GNN, même borne, juste évaluée une fois au
#      lieu d'être batchée. 60s est largement suffisant : la valeur est
#      capturée dès le premier appel GNN, bien avant que la recherche
#      elle-même n'ait une chance de converger sur les grandes instances --
#      aucun besoin d'attendre la fin de cette recherche auxiliaire.
#
# Les deux sorties sont écrites dans le MÊME fichier log, séparées par un
# marqueur "=== SECTION: ... ===" que parse_result_ssp.py utilise pour ne pas
# confondre le status/nodes/solve_time_s du run principal avec ceux -- non
# pertinents -- du run auxiliaire.
#
# time_lim=0 (illimité) pour le run principal -- même convention que MKP et
# que run_ref_array_ssp.sh : le mur SLURM (24h) est le seul filet de sécurité.
#
# Hardware : cf. run_ref_array_ssp.sh (même profil, décision du 2026-08-22).
#
# Soumettre :
#   sbatch --array=1-270 test_paper_trillium/run_gnn_batched_array_ssp.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=trillium_ssp_gnn_batched
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --output=test_paper_trillium/logs/slurm_%x_%A_%a.out
#SBATCH --error=test_paper_trillium/logs/slurm_%x_%A_%a.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )}"
cd "$REPO"

module load StdEnv/2023
module load gcc/12.3
module load cmake/3.27.7
module load python/3.11
module load scipy-stack

source solver_bab/setup_gpu_env.sh
source "$HOME/envs/learning_bounds/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

TESTDIR="$REPO/test_paper_trillium"
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TESTDIR/instances_list_ssp.txt")
read -r E T N REP RELPATH <<< "$LINE"
DATASET="$TESTDIR/$RELPATH"

if [[ ! -d "$DATASET" ]]; then
    echo "[ERREUR] instance introuvable : $DATASET" >&2
    exit 1
fi

BAB_BUILD="$REPO/solver_bab/build_gpu_ssp"

# PAS de (re)build ici -- incident du 2026-08-28 : quand plusieurs tâches
# d'array démarrent en même temps (ex. après une file d'attente sur nœuds
# GPU), leurs "cmake --build" concurrents sur ce MÊME répertoire de build
# partagé se marchent dessus (un lien qui échoue dans une tâche supprime le
# binaire qu'une autre tâche vient de produire avec succès) -- 20 tâches
# FAILED d'un coup + 1 résultat dégradé (root_bound_value manquant, binaire
# disparu sous le run auxiliaire en cours). Le build doit être fait UNE
# SEULE FOIS, en amont, avant `sbatch --array` (cf.
# build_ssp_binaries_once.sh) -- ici on vérifie juste que les binaires
# existent, on échoue bruyamment sinon plutôt que de retenter un build.
BIN_BATCHED="$BAB_BUILD/test_gnn_v1cld_bab_batched"
BIN_SEQ="$BAB_BUILD/test_gnn_v1cld_bab"
if [[ ! -x "$BIN_BATCHED" || ! -x "$BIN_SEQ" ]]; then
    echo "[ERREUR] binaire manquant ($BIN_BATCHED ou $BIN_SEQ) -- lancer" \
         "test_paper_trillium/build_ssp_binaries_once.sh AVANT de soumettre ce job array." >&2
    exit 1
fi
CKPT="$REPO/gnn_ssp/runs/train_scaled/ckpt_scaled_v1cld_E3_T12_N20_E3_T12_N50_E4_T12_N20_E4_T12_N50_E4_T8_N20_E4_T8_N50_lr0.0003_bs32_h64_seed42_v6_periodpool_1k_ep0150.pt"
# illimité par défaut (cf. en-tête) ; SSP_SMOKE_TIME_LIM_GNN permet de forcer
# un budget court pour un smoke test (même logique que
# SSP_SMOKE_TIME_LIM_REF côté run_ref_array_ssp.sh) -- ne pas positionner
# cette variable pour la campagne réelle.
TIME_LIM_S="${SSP_SMOKE_TIME_LIM_GNN:-0}"
K=8
ROOT_GAP_TIME_LIM=60   # run auxiliaire court, cf. en-tête

LOGFILE="$TESTDIR/logs/gnn_batched_E${E}_T${T}_N${N}_rep${REP}.log"
RESULT_CSV="$TESTDIR/results/gnn_batched_E${E}_T${T}_N${N}_rep${REP}.csv"

{
echo "======================================================================"
echo "  test_paper_trillium ssp gnn_batched — E=$E T=$T N=$N rep=$REP — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}   Nœud : $(hostname)"
echo "  K=$K  backend=native  device=cuda"
echo "======================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=== SECTION: gnn_batched ==="
} | tee "$LOGFILE"

# "|| true" sur les deux invocations : avec expected_opt=0.0 (pas de
# vérification voulue), les deux binaires impriment "FAIL"/"INCOMPLET" et
# retournent un code non nul même sur un run réussi/attendu (bug de
# test_gnn_v1cld_bab[_batched].cpp : la branche finale ne traite pas
# expected_opt<=0.0 comme "vérification désactivée" côté code de sortie,
# seulement côté affichage). Sans ce garde, "set -e" tuerait la tâche avant
# le parsing. Le run sidecar est de toute façon attendu INCOMPLET (time_lim
# volontairement court, cf. en-tête) -- son propre statut n'est jamais lu,
# seul root_bound_value l'est. Le vrai statut du run principal est lu depuis
# "[test] status=" par parse_result_ssp.py, indépendamment du code de sortie.
"$BIN_BATCHED" "$DATASET" "$CKPT" "0.0" "$TIME_LIM_S" "$K" cuda 2>&1 | tee -a "$LOGFILE" || true

echo "=== SECTION: root_gap_sidecar ===" | tee -a "$LOGFILE"

"$BIN_SEQ" "$DATASET" "$CKPT" "0.0" "$ROOT_GAP_TIME_LIM" gnn cuda 2>&1 | tee -a "$LOGFILE" || true

python3 "$TESTDIR/parse_result_ssp.py" gnn_batched "$E" "$T" "$N" "$REP" "$LOGFILE" > "$RESULT_CSV"

echo "" | tee -a "$LOGFILE"
echo "Terminé — $(date)" | tee -a "$LOGFILE"
