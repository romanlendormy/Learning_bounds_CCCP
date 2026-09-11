#!/bin/bash
# scratch/paper_campaign_ssp_sld_dld_vs_ref_regenerated.sh — Variante de
# paper_campaign_ssp_sld_dld_vs_ref_full.sh (job principal, 270 instances,
# job 19434333) mais restreinte aux 12 instances REGÉNÉRÉES (cf.
# test_paper/ssp/rorqual/regenerate_infeasible.sh) après que REF n'ait
# trouvé AUCUNE solution sur les originales (INFEASIBLE/TIMEOUT). Lance le
# même binaire de production solver_bab/ssp/campaign_sld_dld_vs_ref
# (REF + DLD + SLD).
#
# Volontairement TENU À PART du job principal (décision utilisateur
# 2026-08-22) : job/array/logs/résultats séparés, même si les 12 instances
# occupent les MÊMES chemins (test_paper/ssp/rorqual/ssp/E{E}_T{T}_N{N}/rep{r})
# que dans le run principal -- seul leur CONTENU a changé (nouvel ep_id).
# Ne pas fusionner automatiquement avec results/ssp_trillium_sld_dld_vs_ref_summary.csv
# tant que l'utilisateur n'a pas validé les 12 nouveaux résultats.
#
# NE RECONFIGURE NI NE REBUILD RIEN (même binaire déjà construit que le job
# principal, cf. son en-tête pour la commande de build si besoin).
#
# Soumission :
#   sbatch scratch/paper_campaign_ssp_sld_dld_vs_ref_regenerated.sh

#SBATCH --account=def-qcappart_cpu
#SBATCH --partition=cpubase_bycore_b4
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=sld_dld_vs_ref_regen
#SBATCH --output=logs/campaign_sld_dld_vs_ref_regen_%A_%a.out
#SBATCH --error=logs/campaign_sld_dld_vs_ref_regen_%A_%a.err
#SBATCH --time=01:50:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=1-12

set -uo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXT="$REPO/external"

module load StdEnv/2023
module load gcc/12.3
module load cmake/3.27.7

export LD_LIBRARY_PATH="$EXT/gecode/lib:$EXT/ortools/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

BIN="$REPO/solver_bab/build_collect_cpu/campaign_sld_dld_vs_ref"
LIST="$REPO/test_paper/ssp/rorqual/instances_list_regenerated.txt"

[[ -x "$BIN" ]] || { echo "ERREUR : $BIN absent -- construire AVANT de soumettre"; exit 1; }

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
[[ -n "$LINE" ]] || { echo "ERREUR : pas de ligne $SLURM_ARRAY_TASK_ID dans $LIST"; exit 1; }

INSTANCE_DIR="$REPO/$LINE"
[[ -d "$INSTANCE_DIR" ]] || { echo "ERREUR : $INSTANCE_DIR introuvable"; exit 1; }

echo "[array $SLURM_ARRAY_TASK_ID] instance=$LINE (régénérée)"

"$BIN" "$INSTANCE_DIR" 100 1800 1800
