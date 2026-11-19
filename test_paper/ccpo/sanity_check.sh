#!/bin/bash
# test_paper/ccpo/sanity_check.sh — Contrôle rapide de la chaîne CCPO sur une instance (n = 8, N = 100, idx 100 ;
# quelques secondes par bras) : LD0 et LD avec les réglages de la campagne, puis comparaison du statut, de l'optimum
# et du nombre de nœuds avec les résultats de l'article (results/ld0, results/ld).
# LD0 ne dépend d'aucun calcul flottant du réseau : ses nœuds doivent coïncider exactement. Pour LD, le forward sur
# CPU au lieu du GPU de l'article peut changer λ au dernier bit, donc l'arbre : seul un écart d'optimum est une erreur.
#
# Depuis la racine du dépôt (sur un nœud de calcul, ou un nœud de connexion pour ce seul test court) :
#   bash test_paper/ccpo/sanity_check.sh            # DEVICE=cpu par défaut
#   DEVICE=cuda bash test_paper/ccpo/sanity_check.sh

set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
BIN="$REPO/build/solver_bab/bab_ccpo"
source env/modules_trillium.sh
# Build CUDA (repéré au stub NVSHMEM) : bibliothèques CUDA nécessaires même avec DEVICE=cpu
[[ -e "$REPO/build/solver_bab/libnvshmem_host.so.3" ]] && source env/setup_gpu_env.sh
TD="$REPO/test_paper/ccpo"
FORWARD_DEVICE="${DEVICE:-cpu}"
source "$TD/campaign_config.env"   # réglages de la campagne (dont DEVICE=cuda, remplacé ci-dessous)
DEVICE=$FORWARD_DEVICE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

EP="${INSTANCES_DIR:-$TD/instances}/n8_N100/i100/ep_000000"
[[ -d "$EP" ]] || { echo "[ERREUR] instance introuvable : $EP (python3 test_paper/ccpo/generate_test_instances.py)" >&2; exit 1; }
OUT="$(mktemp -d)"
echo "# logs : $OUT (forward LD : $DEVICE)"
"$BIN" "$EP" "$TL" gnn_zero "$T_ROOT" "$T_NODE" "$K" "$OMP_NUM_THREADS" > "$OUT/ld0.log" 2>&1
"$BIN" "$EP" "$TL" gnn "$T_ROOT" "$T_NODE" "$K" "$OMP_NUM_THREADS" "$MODEL" "$TORCH_THREADS" "$DEVICE" > "$OUT/ld.log" 2>&1

fail=0
check() {  # mode log résultat_article nœuds_stricts
    local got want
    got=$(python3 "$TD/parse_result.py" "$1" 8 100 100 7000900100 "$2" | cut -d, -f8,9,11)
    want=$(cut -d, -f8,9,11 "$3")
    if [[ "$got" == "$want" ]]; then echo "OK      $1 : $got (statut, optimum, nœuds)"
    elif [[ $4 == 0 && "${got%,*}" == "${want%,*}" ]]; then echo "OK      $1 : $got (statut, optimum ; nœuds article ${want##*,})"
    else echo "ÉCART   $1 : obtenu $got, article $want (log : $2)"; fail=1; fi
}
check gnn_zero "$OUT/ld0.log" "$TD/results/ld0/ld0_n8_N100_i100.csv" 1
check gnn      "$OUT/ld.log"  "$TD/results/ld/ld_n8_N100_i100.csv"  0
exit $fail
