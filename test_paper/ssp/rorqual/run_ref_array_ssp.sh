#!/bin/bash
# run_ref_array_ssp.sh — mode ref (squelette nu, sans borne) sur les 270
# instances de test_paper/ssp/rorqual/ssp/, une tâche d'array par instance
# (test_paper/ssp/rorqual/instances_list.txt, format : E T N rep chemin_relatif).
# CPU seul (ref n'appelle jamais le GNN) -- mais on réserve tout de même un
# GPU H100 dédié + le même profil CPU/RAM que le job gnn_batched
# (run_gnn_batched_array_ssp.sh), décision explicite de l'utilisateur du
# 2026-08-22 : le papier documente "This configuration was used identically
# for all evaluated methods, ensuring that computation times were compared
# on homogeneous hardware" (Trillium, 4 CPU / 187.5GB / 1 H100 dédié par
# tâche). Cette session tourne sur Rorqual (pas d'accès Trillium depuis ici) ;
# --mem=120G est le maximum raisonnable sur les nœuds H100 de Rorqual
# (512000MB / 4 GPU ≈ 128G par GPU dédié), pas les 187.5GB du papier.
#
# Binaire : test_gnn_v1cld_bab (mode "ref"), PAS bench_ssp_bab_file --
# même famille de binaire "[test] ..." que le mode gnn_batched (format de
# sortie unique, un seul parseur), et "ref" y est un squelette nu explicite
# (aucune borne, more() court-circuite), cf. docstring du fichier source.
#
# time_lim=0 (illimité) -- même convention que MKP (test_paper/run_ref_array.sh) :
# le mur SLURM (24h) est le seul filet de sécurité.
#
# Soumettre :
#   sbatch --array=1-270 test_paper/ssp/rorqual/run_ref_array_ssp.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=trillium_ssp_ref
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --output=test_paper/ssp/rorqual/logs/slurm_%x_%A_%a.out
#SBATCH --error=test_paper/ssp/rorqual/logs/slurm_%x_%A_%a.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../.." && pwd )}"
cd "$REPO"

module load StdEnv/2023
module load gcc/12.3
module load cmake/3.27.7
module load python/3.11
module load scipy-stack

source solver_bab/setup_gpu_env.sh
source "$HOME/envs/learning_bounds/bin/activate"

TESTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TESTDIR/instances_list.txt")
read -r E T N REP RELPATH <<< "$LINE"
DATASET="$TESTDIR/$RELPATH"

if [[ ! -d "$DATASET" ]]; then
    echo "[ERREUR] instance introuvable : $DATASET" >&2
    exit 1
fi

BAB_BUILD="$REPO/solver_bab/build_gpu_ssp"

# PAS de (re)build ici -- cf. incident du 2026-08-28 documenté dans
# run_gnn_batched_array_ssp.sh (des "cmake --build" concurrents sur ce
# répertoire de build PARTAGÉ, quand plusieurs tâches d'array démarrent en
# même temps, se corrompent mutuellement). Le build doit être fait UNE SEULE
# FOIS en amont (cf. build_ssp_binaries_once.sh) avant `sbatch --array`.
BINARY="$BAB_BUILD/test_gnn_v1cld_bab"
if [[ ! -x "$BINARY" ]]; then
    echo "[ERREUR] binaire manquant ($BINARY) -- lancer" \
         "test_paper/ssp/rorqual/build_ssp_binaries_once.sh AVANT de soumettre ce job array." >&2
    exit 1
fi
# illimité par défaut (cf. en-tête) ; SSP_SMOKE_TIME_LIM_REF permet de forcer
# un budget court pour un smoke test (valider la chaîne sans attendre une
# preuve d'optimalité complète sur les instances les plus dures), SANS dupliquer
# ce script -- ne pas positionner cette variable pour la campagne réelle.
TIME_LIM_S="${SSP_SMOKE_TIME_LIM_REF:-0}"

LOGFILE="$TESTDIR/logs/ref_E${E}_T${T}_N${N}_rep${REP}.log"
RESULT_CSV="$TESTDIR/results/ref_E${E}_T${T}_N${N}_rep${REP}.csv"

{
echo "======================================================================"
echo "  test_paper_trillium ssp ref — E=$E T=$T N=$N rep=$REP — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}   Nœud : $(hostname)"
echo "======================================================================"
} | tee "$LOGFILE"

# "|| true" : avec expected_opt=0.0 (pas de vérification voulue), le binaire
# imprime "FAIL" et retourne 1 même sur un run OPTIMAL réussi (bug de
# test_gnn_v1cld_bab.cpp : la branche finale ne traite pas expected_opt<=0.0
# comme "vérification désactivée" côté code de sortie, seulement côté
# affichage informel). Sans ce garde, "set -e" tuerait la tâche avant le
# parsing. Le vrai statut de la résolution est lu depuis "[test] status=" par
# parse_result_ssp.py, indépendamment du code de sortie du binaire.
"$BINARY" "$DATASET" "ignored_ckpt" "0.0" "$TIME_LIM_S" ref 2>&1 | tee -a "$LOGFILE" || true

python3 "$TESTDIR/parse_result_ssp.py" ref "$E" "$T" "$N" "$REP" "$LOGFILE" > "$RESULT_CSV"

echo "" | tee -a "$LOGFILE"
echo "Terminé — $(date)" | tee -a "$LOGFILE"
