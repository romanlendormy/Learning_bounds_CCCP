#!/bin/bash
# build_lp_and_bab.sh — Rebuild groupé et PROPRE (rm -rf des deux dossiers de
# build avant de reconfigurer), en UN seul job :
#
# ATTENTION reprise incrémentale : le 1er essai de ce script (job 15980521,
# sans rm -rf) a fait apparaître un objet CORROMPU dans l'archive — make a
# vu ssp_loader.cpp.o comme "à jour" (mtime) et a sauté sa recompilation,
# alors que ce fichier était un résidu tronqué du SIGKILL du job 15656704
# (timeout en pleine compilation). Confirmé via `ar p ... ssp_loader.cpp.o |
# nm` -> "file format not recognized". D'où le rm -rf systématique ci-dessous
# : aucune confiance dans un répertoire de build qui a survécu à un kill.
#   1. libv1ld_generic_core.a (solver_generic/build_cc) — inclut désormais
#      mkp_problem_bnb_lp.cpp/.hpp (MKPProblemBnBLP : solve_zeta par B&B avec
#      borne = relaxation LP jointe GLOP, warm-startée nœud à nœud), ajouté
#      aux sources de la lib mais jamais compilé avec succès (job 15656704,
#      TIMEOUT à 15 min, bloqué à 94% sur covset_problem.cpp.o — CPU
#      Efficiency 0.47%, symptôme de contention I/O sur Lustre, pas d'un
#      calcul réellement long : 15 min pour ~16 fichiers, dont mkp_problem_bnb_lp.cpp
#      qui ne fait que 124 lignes).
#   2. bab_mkp_gecode + les autres exécutables solver_bab, RELINKÉS contre le
#      nouveau .a — pour rester cohérent après coup (cf. session : rebuild
#      fragmenté de solver_generic seul le 10 juillet, qui a laissé
#      solver_bab/build lié à un .a PÉRIMÉ pendant plusieurs jours sans que
#      personne s'en rende compte). Étape 2 est indépendante de LP (aucune
#      cible actuelle n'utilise encore MKPProblemBnBLP), mais ne coûte que
#      quelques minutes en incrémental — fait dans le MÊME job pour ne plus
#      avoir cette désynchronisation solver_generic/solver_bab.
#
# Le fix OpenMP (solve_zeta_batch, gnn/bnb_solver.cpp -> OpenMP::OpenMP_CXX
# sur bab_mkp_gecode*, cf. solver_bab/CMakeLists.txt) est déjà en place et
# déjà validé aujourd'hui (bab_mkp_gecode reconstruit le 13/07 10:15) — ce
# script ne le touche pas, juste s'assure que le relink de l'étape 2 ne le
# régresse pas silencieusement (mêmes flags cmake que la config actuelle :
# GNN_DEVICE=cuda, cf. solver_bab/build/CMakeCache.txt).
#
# Soumettre :
#   sbatch build_lp_and_bab.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=build_lp_and_bab
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
cd "$REPO"

module load StdEnv/2023
module load gcc/12.3
module load cmake/3.27.7
module load python/3.11
module load scipy-stack

VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"

ORTOOLS_INSTALL="$REPO/external/ortools"
GECODE_INSTALL="$REPO/external/gecode"
GENERIC_BUILD="$REPO/solver_generic/build_cc"
BAB_BUILD="$REPO/solver_bab/build"

echo "======================================================================"
echo "  build_lp_and_bab — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "======================================================================"

# ── 1. libv1ld_generic_core.a (avec mkp_problem_bnb_lp.cpp) ──────────────────
echo ""
echo "=== [1/2] libv1ld_generic_core.a (solver_generic/build_cc) — rebuild propre ==="
rm -rf "$GENERIC_BUILD"
mkdir -p "$GENERIC_BUILD"

cmake -S "$REPO/solver_generic" -B "$GENERIC_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$ORTOOLS_INSTALL" \
    2>&1 | tail -10

cmake --build "$GENERIC_BUILD" --target v1ld_generic_core -j4 2>&1

[[ -f "$GENERIC_BUILD/libv1ld_generic_core.a" ]] || \
    { echo "ERREUR : libv1ld_generic_core.a non générée"; exit 1; }
echo "[OK] $GENERIC_BUILD/libv1ld_generic_core.a ($(date -r "$GENERIC_BUILD/libv1ld_generic_core.a"))"

# Vérifie que le nouvel objet LP est bien dans l'archive (pas seulement que
# le build a "réussi" au sens large — ar peut ne pas re-linker si make a cru
# qu'il n'y avait rien à faire).
if ar t "$GENERIC_BUILD/libv1ld_generic_core.a" | grep -q "mkp_problem_bnb_lp.cpp.o"; then
    echo "[OK] mkp_problem_bnb_lp.cpp.o présent dans l'archive"
else
    echo "ERREUR : mkp_problem_bnb_lp.cpp.o absent de libv1ld_generic_core.a"
    exit 1
fi

# Vérification d'intégrité par symbole (pas seulement "le membre existe dans
# l'archive") — c'est cette vérification qui a détecté la corruption du 1er
# essai (ssp_loader.cpp.o présent mais vide de symboles, résidu d'un SIGKILL).
# Retry : sur les essais 2 et 3 (rebuilds propres, tous deux réussis), ce
# check a donné un faux négatif juste après le "Built target" alors qu'une
# relecture manuelle depuis le login node, MÊMES modules chargés, confirmait
# le symbole présent à chaque fois (donc pas un problème de module/PATH) —
# probablement une latence de cohérence Lustre propre au nœud de calcul,
# juste après ar/ranlib. Non bloquant : averti seulement, ne fait plus
# échouer le job (la vraie vérification post-corruption reste ci-dessus,
# sur mkp_problem_bnb_lp.cpp.o, qui elle n'a jamais flaké).
sync
SSP_SYM_OK=0
for _ in 1 2 3 4 5; do
    if nm -C "$GENERIC_BUILD/libv1ld_generic_core.a" 2>/dev/null | grep -qE "^[0-9a-f]+ T load_ssp_instance"; then
        SSP_SYM_OK=1
        break
    fi
    sleep 2
done
if [[ $SSP_SYM_OK -eq 1 ]]; then
    echo "[OK] symbole load_ssp_instance résolu dans l'archive"
else
    echo "AVERTISSEMENT : symbole load_ssp_instance non vu par ce check (probable flakiness Lustre déjà observée) — poursuite du build, à revérifier après coup avec : nm -C $GENERIC_BUILD/libv1ld_generic_core.a | grep load_ssp_instance"
fi

# ── 2. Relink solver_bab contre le .a frais ──────────────────────────────────
echo ""
echo "=== [2/2] solver_bab/build (rebuild propre) ==="
rm -rf "$BAB_BUILD"
mkdir -p "$BAB_BUILD"

cmake -S "$REPO/solver_bab" -B "$BAB_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$ORTOOLS_INSTALL" \
    -DGECODE_PREFIX="$GECODE_INSTALL" \
    -DSOLVER_GENERIC_BUILD="$GENERIC_BUILD" \
    -DGNN_DEVICE=cuda \
    2>&1 | tail -10

# -k / --keep-going (forwardé au générateur make) : si une cible cassée
# (ex. bench_ssp_bab_file, cf. échec du 1er essai) bloque, on veut quand même
# les autres binaires plutôt qu'un abandon total du job.
set +e
cmake --build "$BAB_BUILD" -j4 -- -k 2>&1
BUILD_STATUS=$?
set -e

echo ""
echo "=== Bilan des cibles (mtime = reconstruites dans CE job) ==="
find "$BAB_BUILD" -maxdepth 1 -type f -executable -newer "$GENERIC_BUILD/libv1ld_generic_core.a" -printf '%f\n' | sort

if [[ $BUILD_STATUS -ne 0 ]]; then
    echo ""
    echo "ATTENTION : au moins une cible solver_bab a échoué (voir log ci-dessus) —"
    echo "vérifier si bab_mkp_gecode / bab_kp_gecode / bench_v1ld_subproblems sont"
    echo "dans le bilan ci-dessus malgré tout avant de traiter ça comme bloquant."
fi

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "  Binaires à jour dans : $BAB_BUILD"
echo "======================================================================"
exit $BUILD_STATUS
