#!/bin/bash
# collect/run_collect_overfit_holdout_largeN.sh — Collecte "best-effort" d'instances
# CCMKP fraîches à grand N (500 ou 1000), n=20, pour re-tester diag_overfit_single.py
# sur plusieurs instances indépendantes (cf. run_collect_overfit_holdout.sh pour N=50,
# rapide et déjà fait séparément — ce script ne couvre QUE les N coûteux).
#
# Le B&B CP-pur (sans borne V1LD/GNN) de collect_gecode_base ne prouve
# l'optimalité — condition requise pour écrire quoi que ce soit, y compris la
# racine triviale, cf. gecode_base/collect_gecode_base.cpp — que très lentement
# à grand N (0/3 instances terminées en 30s à N=500 lors du test préalable ;
# à N=1000, la 1ère instance n'avait pas terminé après 2min sans plafond).
# Choix assumé : pas de --time_limit (illimité par instance), budget global de
# 24h par ce job, "best effort" — même convention que
# run_collect_base_n25_30_rorqual.sh ("cible 10000, best-effort en 24h").
# Le traitement des seeds est séquentiel : ce qui est déjà écrit avant la coupure
# à 24h reste valide sur disque (rien n'est perdu), seule l'instance en cours au
# moment de la coupure est perdue.
#
# Soumettre (un job par N, tourne en parallèle) :
#   sbatch --export=ALL,N_SCEN=500  collect/run_collect_overfit_holdout_largeN.sh
#   sbatch --export=ALL,N_SCEN=1000 collect/run_collect_overfit_holdout_largeN.sh
#
# BUG corrigé (2026-07-10) : les deux jobs (N=500 et N=1000) partageaient le même
# --outdir, donc le même fichier de staging générique instances_n20.txt/nodes_n20.txt
# (indexé par n SEUL, pas par N — cf. gecode_base/main_collect_gecode_base.cpp).
# Lancés en parallèle, les deux process C++ ont écrit dans le même fichier en
# même temps → lignes interleaved / corrompues (tailles de champs incohérentes,
# données N=1000 mélangées dans le fichier N=500 et vice-versa). Chaque N utilise
# maintenant son propre --outdir pour ne plus jamais partager de fichier de staging.

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=collect_overfit_holdout_largeN
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

N_SCEN="${N_SCEN:?Définir N_SCEN=500 ou 1000 via --export}"

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )}"
cd "$REPO"

VENV_PATH="$HOME/envs/learning_bounds"
source "$VENV_PATH/bin/activate"
export LD_LIBRARY_PATH="$REPO/external/gecode/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

BINARY="$REPO/solver_bab/build/bab_mkp_gecode_base_collect"
OUTDIR="$REPO/data/overfit_holdout_N${N_SCEN}/train"
SEED_START=70001
N_INSTANCES=10

echo "======================================================================"
echo "  collect_overfit_holdout_largeN — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  N_SCEN=$N_SCEN  seeds ${SEED_START}..$((SEED_START + N_INSTANCES - 1))  n=20"
echo "  time_limit par instance : illimité (best-effort, coupure au bout de 24h)"
echo "======================================================================"

python collect/collect_large_base_samples.py \
    --n 20 --m 5 --N "$N_SCEN" --alpha 0.5 --R 1000 --correlation weakly_correlated \
    --epsilon 0.1 \
    --n_instances "$N_INSTANCES" \
    --max_depth 0 \
    --seed_start "$SEED_START" \
    --binary "$BINARY" \
    --outdir "$OUTDIR"

echo ""
f="${OUTDIR}_base_s10/nodes_n20_m5_N${N_SCEN}.txt"
n_lines=$(wc -l < "$f" 2>/dev/null || echo 0)
echo "  N=$N_SCEN : $n_lines instance(s) valide(s) obtenue(s)"
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
