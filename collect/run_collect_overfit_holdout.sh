#!/bin/bash
# collect/run_collect_overfit_holdout.sh — Génère des instances CCMKP fraîches
# (n=20, N∈{50,500,1000}) pour ré-exécuter diag_overfit_single.py sur PLUSIEURS
# instances aléatoires plutôt qu'une seule (cf. jobs 15583299/300/301 : un seul
# point par N dans data/train_base_s10, dont un seul nœud dispo pour N=500/1000
# — donc aucune marge pour juger si le comportement observé est spécifique à
# l'instance ou général).
#
# Seeds 70001..70008 (8 = marge au-delà des 5 requis : certaines instances ont
# une racine déjà résolue par la seule propagation CP — SS_SOLVED avant tout
# branchement — auquel cas collect_gecode_base n'écrit aucun nœud ; cf.
# gecode_base/collect_gecode_base.cpp, le nœud racine n'est enregistré que s'il
# atteint SS_BRANCH). --max_depth 0 (ne garder que la racine) + --time_limit 90s
# (à N=1000 le B&B CP pur sans borne peut ne pas terminer vite — le nœud racine
# est de toute façon enregistré dès la 1ère itération de la boucle, avant tout
# risque de timeout).
#
# Sortie : data/overfit_holdout_train_base_s10/nodes_n20_m5_N{50,500,1000}.txt
# (dossier séparé de data/train_base_s10 — ne pas polluer les données d'entraînement
# réelles avec ces instances jetables).
#
# Soumettre :
#   sbatch collect/run_collect_overfit_holdout.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=collect_overfit_holdout
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )}"
cd "$REPO"

VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export LD_LIBRARY_PATH="$REPO/external/gecode/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

BINARY="$REPO/solver_bab/build/bab_mkp_gecode_base_collect"
OUTDIR="$REPO/data/overfit_holdout/train"
SEED_START=70001
N_INSTANCES=8

echo "======================================================================"
echo "  collect_overfit_holdout — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  seeds ${SEED_START}..$((SEED_START + N_INSTANCES - 1))  n=20  N in {50,500,1000}"
echo "======================================================================"

for N in 50 500 1000; do
    echo ""
    echo "=== N=$N ==="
    python collect/collect_large_base_samples.py \
        --n 20 --m 5 --N "$N" --alpha 0.5 --R 1000 --correlation weakly_correlated \
        --epsilon 0.1 \
        --n_instances "$N_INSTANCES" \
        --max_depth 0 \
        --time_limit 90 \
        --seed_start "$SEED_START" \
        --binary "$BINARY" \
        --outdir "$OUTDIR"
done

echo ""
echo "======================================================================"
echo "  Récapitulatif des nœuds racine collectés :"
for N in 50 500 1000; do
    f="${OUTDIR}_base_s10/nodes_n20_m5_N${N}.txt"
    n_lines=$(wc -l < "$f" 2>/dev/null || echo 0)
    echo "    N=$N : $n_lines instance(s) valide(s)"
done
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
