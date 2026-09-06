#!/bin/bash
# test_paper_trillium/ssp/regenerate_infeasible.sh — Régénère les 12
# instances où le B&B REF exact (campaign_v1ld_v1cld_vs_ref, job 19434333,
# budget 1800s) n'a trouvé AUCUNE solution (ref_status=INFEASIBLE ou
# TIMEOUT-sans-incumbent) : la calibration par bissection (ssp_probe, 5s,
# pruning CP pur) atteignait bien la fenêtre de difficulté visée, mais ces
# instances précises se sont révélées réellement infaisables (ou trop dures
# pour prouver quoi que ce soit en 1800s) une fois résolues à fond.
#
# Toutes les 12 sont E∈{6,8} -> fenêtre par défaut [1e4,1e6] (cf.
# run_gen_ssp_test_paper_trillium_rorqual.sh). ep_id = rep + 1000 (offset
# jamais utilisé par le script de génération original, qui va jusqu'à
# rep+400 pour ses propres replis) -- garantit un tirage DIFFÉRENT du
# précédent (même E/T/N/epsilon/master_seed), pas une simple re-génération
# à l'identique qui recréerait la même instance infaisable.
#
# Écrase le dossier repN existant à sa place (même chemin -- "mêmes données"
# pour le run de suivi), met à jour PROVENANCE.txt, et liste les 12 instances
# dans instances_list_regenerated.txt pour un run de campagne séparé.
#
# Usage : bash test_paper_trillium/ssp/regenerate_infeasible.sh

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GEN_SCRIPT="$REPO/gnn_ssp/gen_ssp_instance_custom.py"
PROBE_BIN="$REPO/solver_bab/build_collect_cpu/ssp_probe"
OUT_ROOT="$REPO/test_paper_trillium/ssp"
LIST_OUT="$OUT_ROOT/instances_list_regenerated.txt"
EPSILON="0.1"
MASTER_SEED="42"
EPID_OFFSET=1000

# (E T N rep) -- les 12 instances sans solution trouvée par REF (job 19434333)
INSTANCES=(
  "6 10 500 6"
  "6 12 500 1"
  "6 12 500 2"
  "6 12 500 6"
  "8 10 500 0"
  "8 10 500 2"
  "8 12 100 4"
  "8 12 100 5"
  "8 12 100 6"
  "8 12 500 4"
  "8 12 500 6"
  "8 12 50 6"
)

[[ -x "$PROBE_BIN" ]] || { echo "ERREUR : $PROBE_BIN introuvable"; exit 1; }
: > "$LIST_OUT"

N_OK=0
N_FAIL=0
for spec in "${INSTANCES[@]}"; do
    read -r E T N REP <<< "$spec"
    COMBO_DIR="$OUT_ROOT/E${E}_T${T}_N${N}"
    REP_DIR="$COMBO_DIR/rep${REP}"
    PROV_FILE="$COMBO_DIR/PROVENANCE.txt"
    EPID=$(( REP + EPID_OFFSET ))

    # Toutes E in {6,8} ici -> fenêtre par défaut.
    TARGET_LO="1e4"
    TARGET_HI="1e6"

    echo "==== E=$E T=$T N=$N rep=$REP -> $REP_DIR (ep_id=$EPID) ===="
    rm -rf "$REP_DIR"
    if python "$GEN_SCRIPT" \
            --E "$E" --T "$T" --N "$N" \
            --epsilon "$EPSILON" \
            --ep_id "$EPID" \
            --master_seed "$MASTER_SEED" \
            --target_lo "$TARGET_LO" --target_hi "$TARGET_HI" \
            --probe_bin "$PROBE_BIN" \
            --out_dir "$REP_DIR"; then
        echo "rep${REP}=regenerated_after_ref_infeasible ep_id=${EPID} date=$(date -Iseconds)" >> "$PROV_FILE"
        echo "test_paper_trillium/ssp/E${E}_T${T}_N${N}/rep${REP}" >> "$LIST_OUT"
        N_OK=$(( N_OK + 1 ))
    else
        echo "[regen] ÉCHEC E=$E T=$T N=$N rep=$REP ep_id=$EPID"
        echo "rep${REP}=REGEN_FAILED ep_id_tried=${EPID} date=$(date -Iseconds)" >> "$PROV_FILE"
        N_FAIL=$(( N_FAIL + 1 ))
    fi
done

echo "======================================================================"
echo "Régénération terminée : $N_OK/12 réussies, $N_FAIL échec(s)"
echo "Liste des instances régénérées : $LIST_OUT"
