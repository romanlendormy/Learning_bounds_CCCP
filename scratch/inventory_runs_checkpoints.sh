#!/bin/bash
# scratch/inventory_runs_checkpoints.sh — Inventaire par run_tag des
# checkpoints dans gnn_sd_edit/runs/train_scaled/ (SLD) et
# gnn_sd_edit/runs_dld/ (DLD), pour décider quoi garder/supprimer.
# CPU seul, aucun calcul lourd -- juste des du/ls sur Lustre.
#
# Usage : sbatch scratch/inventory_runs_checkpoints.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=inventory_runs
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0:10:00
#SBATCH --output=scratch/logs/inventory_runs_%j.out
#SBATCH --error=scratch/logs/inventory_runs_%j.err

set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )}"
cd "$REPO"
mkdir -p scratch/logs

echo "======================================================================"
echo "  SLD -- gnn_sd_edit/runs/train_scaled/ (par run_tag)"
echo "======================================================================"
for tag in "calib\." "v1_sdedit_light_100k" "n10_20_25_20k_100ep" "v1_sdedit_500ep"; do
    echo "--- $tag ---"
    n=$(ls gnn_sd_edit/runs/train_scaled/*"$tag"* 2>/dev/null | wc -l)
    sz=$(ls gnn_sd_edit/runs/train_scaled/*"$tag"* 2>/dev/null | xargs du -ch 2>/dev/null | tail -1)
    echo "  $n fichiers, $sz"
done

echo
echo "--- overfit_single/ (orpheline depuis suppression de diag_overfit_single.py) ---"
du -sh gnn_sd_edit/runs/overfit_single/ 2>/dev/null

echo
echo "======================================================================"
echo "  DLD -- gnn_sd_edit/runs_dld/ (par groupe)"
echo "======================================================================"
echo "--- calib ---"
ls gnn_sd_edit/runs_dld/*calib* 2>/dev/null | wc -l
ls gnn_sd_edit/runs_dld/*calib* 2>/dev/null | xargs du -ch 2>/dev/null | tail -1

echo "--- run court superseded (ep0010..ep0100, sans suffixe _500ep) ---"
ls gnn_sd_edit/runs_dld/*dld_n10_20_ep* 2>/dev/null | wc -l
ls gnn_sd_edit/runs_dld/*dld_n10_20_ep* 2>/dev/null | xargs du -ch 2>/dev/null | tail -1

echo "--- _500ep intermédiaires (ep0010 à ep0490) ---"
ls gnn_sd_edit/runs_dld/*_500ep_ep0[0-4]*.pt 2>/dev/null | grep -v "ep0500" | wc -l
ls gnn_sd_edit/runs_dld/*_500ep_ep0[0-4]*.pt 2>/dev/null | grep -v "ep0500" | xargs du -ch 2>/dev/null | tail -1

echo "--- _500ep final (ep0500, LE checkpoint de production) ---"
ls gnn_sd_edit/runs_dld/*_500ep_ep0500* 2>/dev/null
du -ch gnn_sd_edit/runs_dld/*_500ep_ep0500* 2>/dev/null | tail -1

echo
echo "======================================================================"
echo "  Totaux"
echo "======================================================================"
du -sh gnn_sd_edit/runs/ gnn_sd_edit/runs_dld/ 2>/dev/null
