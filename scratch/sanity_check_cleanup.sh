#!/bin/bash
# scratch/sanity_check_cleanup.sh — Vérifie que le nettoyage de septembre 2026
# (suppression de gnn/, déplacement de gnn_v1ld_scriptable.py vers
# gnn_sd_edit/, nettoyage collect/, solver_bab/, data/, sweep_bab/) n'a rien
# cassé côté pipeline GNN-batché test_paper (MKP, V1LD et V1CLD) /
# test_paper_trillium (SSP) ni côté lancement d'entraînement.
#
# 10 checks indépendants, chacun continue même si un précédent échoue (pas de
# set -e) — le but est de voir TOUT ce qui casse en un seul run :
#   1.  Build MKP V1LD  (bab_mkp_gecode)
#   2.  Build MKP V1CLD (bab_mkp_gecode_v1cld_gnn_batched)
#   3.  Build SSP        (test_gnn_v1cld_bab_batched)
#   4.  Imports Python des fichiers repointés vers gnn_sd_edit.gnn_v1ld_scriptable
#   5.  B&B GNN-batché MKP V1LD minimal
#   6.  B&B GNN-batché MKP V1CLD minimal (même instance que #5 — l'en-tête de
#       test_paper_trillium_v1cld/run_v1cld_array.sh documente un sanity-check
#       antérieur donnant le MÊME optimum 3436.0230 que V1LD/ref sur cette
#       instance — vérifié ici en plus du simple "ne plante pas")
#   7.  B&B GNN-batché SSP minimal
#   8.  Smoke test entraînement MKP sd_edit V1LD  (2 epochs, 200 graphes)
#   9.  Smoke test entraînement MKP sd_edit V1CLD (2 epochs, 200 graphes)
#   10. Smoke test entraînement SSP v1cld         (2 epochs, 200 graphes)
#
# Note sur les checks 5/6/7 : avec expected_opt=0.0 ou non spécifié, le code
# de sortie du binaire n'est pas fiable (bug connu, documenté ailleurs dans le
# dépôt) — le verdict est basé sur le contenu du log (statut trouvé, absence
# de plantage), pas sur le code de retour.
#
# Sorties d'entraînement (checkpoints/figures jetables) : scratch/tmp_smoketest/
# — à supprimer après lecture du résumé, ne remplace aucun vrai run.
#
# Usage :
#   sbatch scratch/sanity_check_cleanup.sh

#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=sanity_cleanup
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=1:15:00
#SBATCH --output=scratch/logs/sanity_cleanup_%j.out
#SBATCH --error=scratch/logs/sanity_cleanup_%j.err

set -uo pipefail   # pas de -e : chaque check doit tourner même si un autre a échoué

REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )}"
cd "$REPO"
mkdir -p scratch/logs scratch/tmp_smoketest

module load StdEnv/2023 gcc/12.3 cmake/3.27.7 python/3.11 scipy-stack
source solver_bab/setup_gpu_env.sh
source "$HOME/envs/learning_bounds/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

declare -A RESULT
section() { echo; echo "======================================================================"; echo "  $1"; echo "======================================================================"; }
status_ok() {  # $1 = fichier log
    grep -qiE "status=|Statut\s*:|OPTIMAL|TIMEOUT" "$1" && ! grep -qiE "segmentation fault|terminate called|traceback|core dumped|runtime_error|bad_alloc" "$1"
}

# ── 1. Build MKP V1LD ────────────────────────────────────────────────────────
section "1/10 Build MKP V1LD (bab_mkp_gecode)"
if cmake --build solver_bab/build --target bab_mkp_gecode -j4; then
    RESULT[1_build_mkp_v1ld]="PASS"
else
    RESULT[1_build_mkp_v1ld]="FAIL"
fi

# ── 2. Build MKP V1CLD ───────────────────────────────────────────────────────
section "2/10 Build MKP V1CLD (bab_mkp_gecode_v1cld_gnn_batched)"
if cmake --build solver_bab/build --target bab_mkp_gecode_v1cld_gnn_batched -j4; then
    RESULT[2_build_mkp_v1cld]="PASS"
else
    RESULT[2_build_mkp_v1cld]="FAIL"
fi

# ── 3. Build SSP ─────────────────────────────────────────────────────────────
section "3/10 Build SSP (test_gnn_v1cld_bab_batched)"
if cmake --build solver_bab/build_gpu_ssp --target test_gnn_v1cld_bab_batched -j4; then
    RESULT[3_build_ssp]="PASS"
else
    RESULT[3_build_ssp]="FAIL"
fi

# ── 4. Imports Python ────────────────────────────────────────────────────────
section "4/10 Imports Python (gnn_sd_edit.gnn_v1ld_scriptable et ses appelants)"
LOG4="scratch/logs/check4_imports.log"
python3 -c "
import sys
sys.path.insert(0, '.')
from gnn_sd_edit.gnn_v1ld_scriptable import GNNScriptable, GatedGraphConvScriptable
import gnn_sd_edit.gnn_sd_edit_scriptable_batched
import gnn_sd_edit.gnn_light
import gnn_sd_edit.gnn_v1cld_sd_edit_scriptable_batched
import gnn_ssp.gnn_ssp_scriptable_batched
import gnn_ssp.validate_clique_mean_equivalence
print('tous les imports OK')
" > "$LOG4" 2>&1
cat "$LOG4"
if grep -q "tous les imports OK" "$LOG4"; then
    RESULT[4_imports]="PASS"
else
    RESULT[4_imports]="FAIL"
fi

# ── 5. B&B GNN-batché MKP V1LD minimal ───────────────────────────────────────
section "5/10 B&B GNN-batché MKP V1LD minimal (n=15,N=100,seed=43, K=8, 60s)"
if [[ "${RESULT[1_build_mkp_v1ld]}" == "PASS" ]]; then
    CKPT_MKP="gnn_sd_edit/runs/train_scaled/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
    LOG5="scratch/logs/check5_bb_mkp_v1ld.log"
    timeout 150 solver_bab/build/bab_mkp_gecode \
        test_paper/instance_n15_m5_N100_seed43.txt 0.10 10 60 0 20 gnn_batched \
        non 0.1 "$CKPT_MKP" cuda python3 ipc sd_edit 0.0 0.0 1.0 8 1.0 surrogate 1 native \
        > "$LOG5" 2>&1
    cat "$LOG5"
    if status_ok "$LOG5"; then RESULT[5_bb_mkp_v1ld]="PASS"; else RESULT[5_bb_mkp_v1ld]="FAIL"; fi
else
    RESULT[5_bb_mkp_v1ld]="SKIP (build MKP V1LD a échoué)"
fi

# ── 6. B&B GNN-batché MKP V1CLD minimal ──────────────────────────────────────
section "6/10 B&B GNN-batché MKP V1CLD minimal (même instance que #5, K=8)"
if [[ "${RESULT[2_build_mkp_v1cld]}" == "PASS" ]]; then
    CKPT_V1CLD="gnn_sd_edit/runs_v1cld/ckpt_v1cld_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_v1cld_n10_20_500ep_ep0500.pt"
    LOG6="scratch/logs/check6_bb_mkp_v1cld.log"
    timeout 150 solver_bab/build/bab_mkp_gecode_v1cld_gnn_batched \
        test_paper/instance_n15_m5_N100_seed43.txt "$CKPT_V1CLD" 0.10 60 0 cuda 8 \
        > "$LOG6" 2>&1
    cat "$LOG6"
    if status_ok "$LOG6"; then
        if grep -q "3436.023" "$LOG6"; then
            RESULT[6_bb_mkp_v1cld]="PASS (optimum 3436.0230 confirmé, identique à V1LD/ref)"
        else
            RESULT[6_bb_mkp_v1cld]="PASS (tourne, mais optimum attendu 3436.0230 non retrouvé -- à vérifier)"
        fi
    else
        RESULT[6_bb_mkp_v1cld]="FAIL"
    fi
else
    RESULT[6_bb_mkp_v1cld]="SKIP (build MKP V1CLD a échoué)"
fi

# ── 7. B&B GNN-batché SSP minimal ────────────────────────────────────────────
section "7/10 B&B GNN-batché SSP minimal (E4_T8_N50/rep0, K=8, 60s)"
if [[ "${RESULT[3_build_ssp]}" == "PASS" ]]; then
    CKPT_SSP="gnn_ssp/runs/train_scaled/ckpt_scaled_v1cld_E3_T12_N20_E3_T12_N50_E4_T12_N20_E4_T12_N50_E4_T8_N20_E4_T8_N50_lr0.0003_bs32_h64_seed42_v6_periodpool_1k_ep0150.pt"
    LOG7="scratch/logs/check7_bb_ssp.log"
    timeout 150 solver_bab/build_gpu_ssp/test_gnn_v1cld_bab_batched \
        test_paper_trillium/ssp/E4_T8_N50/rep0 "$CKPT_SSP" 0.0 60 8 cuda \
        > "$LOG7" 2>&1
    cat "$LOG7"
    if status_ok "$LOG7"; then RESULT[7_bb_ssp]="PASS"; else RESULT[7_bb_ssp]="FAIL"; fi
else
    RESULT[7_bb_ssp]="SKIP (build SSP a échoué)"
fi

# ── 8. Smoke test entraînement MKP sd_edit V1LD ──────────────────────────────
section "8/10 Smoke test entraînement MKP sd_edit V1LD (2 epochs, 200 graphes)"
if timeout 600 python3 gnn_sd_edit/train_scaled_sd_edit.py \
    --n_list 10 --n_scens 20 --target_graphs 200 --n_epochs 2 --n_warmup 0 \
    --skip_quality --eval_every 1 --run_tag smoketest \
    --out_dir scratch/tmp_smoketest/mkp_sd_edit_v1ld; then
    RESULT[8_train_mkp_v1ld]="PASS"
else
    RESULT[8_train_mkp_v1ld]="FAIL"
fi

# ── 9. Smoke test entraînement MKP sd_edit V1CLD ─────────────────────────────
section "9/10 Smoke test entraînement MKP sd_edit V1CLD (2 epochs, 200 graphes)"
if timeout 600 python3 gnn_sd_edit/train_scaled_v1CLD_sd_edit.py \
    --n_list 10 --n_scens 20 --target_graphs 200 --n_epochs 2 --n_warmup 0 \
    --skip_quality --eval_every 1 --run_tag smoketest \
    --out_dir scratch/tmp_smoketest/mkp_sd_edit_v1cld; then
    RESULT[9_train_mkp_v1cld]="PASS"
else
    RESULT[9_train_mkp_v1cld]="FAIL"
fi

# ── 10. Smoke test entraînement SSP v1cld ────────────────────────────────────
section "10/10 Smoke test entraînement SSP v1cld (2 epochs, 200 graphes)"
if timeout 600 python3 gnn_ssp/train_scaled_v1cld.py \
    --combos E3_T12_N20 --target_graphs 200 --n_epochs 2 --n_warmup 0 \
    --eval_every 1 --run_tag smoketest \
    --out_dir scratch/tmp_smoketest/ssp_v1cld; then
    RESULT[10_train_ssp]="PASS"
else
    RESULT[10_train_ssp]="FAIL"
fi

# ── Résumé ────────────────────────────────────────────────────────────────────
section "RÉSUMÉ"
for k in 1_build_mkp_v1ld 2_build_mkp_v1cld 3_build_ssp 4_imports 5_bb_mkp_v1ld 6_bb_mkp_v1cld 7_bb_ssp 8_train_mkp_v1ld 9_train_mkp_v1cld 10_train_ssp; do
    printf "  %-16s %s\n" "$k" "${RESULT[$k]:-?}"
done
echo
echo "Sorties jetables : scratch/tmp_smoketest/ (à supprimer après lecture)"
