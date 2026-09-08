#!/bin/bash
# rebuild_bab_mkp_gecode.sh — Rebuild INCRÉMENTAL (pas de rm -rf) de la seule
# cible bab_mkp_gecode, suite à l'instrumentation ajoutée dans
# gecode_bab_gnn_batched.hpp/.cpp et main_gecode.cpp (capture + impression du
# proxy "gap racine" en mode gnn_batched, cf. MANIFEST.md dans ce dossier).
#
# Contrairement à build_lp_and_bab.sh (rm -rf systématique des deux dossiers
# de build, cf. son en-tête sur la corruption post-SIGKILL du 15656704) :
# solver_bab/build n'a jamais été kill -9, dry-run (`make -n bab_mkp_gecode`)
# confirmé sain — seuls main_gecode.cpp.o et gecode_bab_gnn_batched.cpp.o +
# le relink sont reconstruits, aucune modification de solver_generic. Un
# rebuild ciblé est donc suffisant et bien plus rapide qu'un rebuild complet.
#
# Soumettre :
#   sbatch test_paper/mkp/trillium/v1ld/rebuild_bab_mkp_gecode.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=rebuild_bab_mkp_gecode
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../../.." && pwd )}"
cd "$REPO"

module load StdEnv/2023
module load gcc/12.3
module load cmake/3.27.7

BAB_BUILD="$REPO/solver_bab/build"

echo "======================================================================"
echo "  rebuild_bab_mkp_gecode — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "======================================================================"

BEFORE_MTIME=$(date -r "$BAB_BUILD/bab_mkp_gecode" +%s)

cmake --build "$BAB_BUILD" --target bab_mkp_gecode -j4 2>&1

AFTER_MTIME=$(date -r "$BAB_BUILD/bab_mkp_gecode" +%s)

if [[ "$AFTER_MTIME" -gt "$BEFORE_MTIME" ]]; then
    echo ""
    echo "[OK] bab_mkp_gecode reconstruit ($(date -r "$BAB_BUILD/bab_mkp_gecode"))"
else
    echo ""
    echo "ERREUR : mtime de bab_mkp_gecode inchangé — le binaire ne semble pas avoir été relinké" >&2
    exit 1
fi

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
