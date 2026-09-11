#!/bin/bash
# scratch/paper_campaign_ssp_sld_dld_vs_ref_full.sh — Job array : une
# tâche par instance de test_paper/ssp/rorqual/ssp/ (270 instances, cf.
# instances_list.txt), chacune exécutant le binaire de production
# solver_bab/ssp/campaign_sld_dld_vs_ref (REF + DLD + SLD, cf.
# campaign_sld_dld_vs_ref.cpp pour le détail et les décisions de
# conception -- restriction SLD à N<=100, écriture CSV en deux temps
# REF+DLD puis SLD pour ne pas perdre de données si SLD bloque).
#
# NE RECONFIGURE NI NE REBUILD RIEN : le binaire campaign_sld_dld_vs_ref
# doit être compilé AVANT de soumettre ce job (270 tâches concurrentes
# reconfigurant/rebuildant le même solver_bab/build_collect_cpu se
# marcheraient dessus). Construire avec :
#   cmake -S solver_bab -B solver_bab/build_collect_cpu \
#       -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$EXT/ortools \
#       -DGECODE_PREFIX=$EXT/gecode \
#       -DSOLVER_GENERIC_BUILD=$REPO/solver_generic/build_cc -DGNN_DEVICE=cpu
#   cmake --build solver_bab/build_collect_cpu --target campaign_sld_dld_vs_ref -j4
#
# Chaque tâche lit sa ligne dans instances_list.txt (index array = numéro de
# ligne, 1-indexé via sed -n Np) et laisse SLURM capturer stdout dans
# logs/campaign_sld_dld_vs_ref_%A_%a.out -- le binaire flush explicitement
# chaque ligne CSV_ROW, donc même une tâche tuée par --time garde sa ligne
# PENDING (REF+DLD) dans son .out. Le script de fusion
# (paper_campaign_ssp_sld_dld_vs_ref_merge.sh) relit ces .out et garde la
# DERNIÈRE ligne CSV_ROW de chacun.
#
# Budget par tâche : REF, DLD et SLD ont chacun leur propre plafond de
# 1800s (30 min) -- PAS un budget total partagé (décision utilisateur
# 2026-08-22) -- donc jusqu'à ~90 min dans le pire cas + marge de
# chargement/instance : --time=01:50:00.
#
# Soumission (après avoir construit le binaire, cf. ci-dessus) :
#   sbatch scratch/paper_campaign_ssp_sld_dld_vs_ref_full.sh

#SBATCH --account=def-qcappart_cpu
#SBATCH --partition=cpubase_bycore_b4
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=sld_dld_vs_ref
#SBATCH --output=logs/campaign_sld_dld_vs_ref_%A_%a.out
#SBATCH --error=logs/campaign_sld_dld_vs_ref_%A_%a.err
#SBATCH --time=01:50:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=1-270

set -uo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXT="$REPO/external"

module load StdEnv/2023
module load gcc/12.3
module load cmake/3.27.7

export LD_LIBRARY_PATH="$EXT/gecode/lib:$EXT/ortools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

BIN="$REPO/solver_bab/build_collect_cpu/campaign_sld_dld_vs_ref"
LIST="$REPO/test_paper/ssp/rorqual/instances_list.txt"

[[ -x "$BIN" ]] || { echo "ERREUR : $BIN absent -- construire AVANT de soumettre (cf. en-tête du script)"; exit 1; }

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
[[ -n "$LINE" ]] || { echo "ERREUR : pas de ligne $SLURM_ARRAY_TASK_ID dans $LIST"; exit 1; }

# Format d'une ligne : "E T N rep chemin_relatif" (chemin relatif à
# test_paper/ssp/rorqual/, cf. test_paper/ssp/rorqual/gen_instances_list_ssp.sh) --
# pas un chemin nu contrairement à instances_list_regenerated.txt.
REL_PATH=$(awk '{print $5}' <<< "$LINE")
INSTANCE_DIR="$REPO/test_paper/ssp/rorqual/$REL_PATH"
[[ -d "$INSTANCE_DIR" ]] || { echo "ERREUR : $INSTANCE_DIR introuvable"; exit 1; }

echo "[array $SLURM_ARRAY_TASK_ID] instance=$LINE"

"$BIN" "$INSTANCE_DIR" 100 1800 1800
