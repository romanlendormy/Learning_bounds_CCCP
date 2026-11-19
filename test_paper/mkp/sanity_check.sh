#!/bin/bash
# test_paper/mkp/sanity_check.sh — Contrôle rapide de la chaîne MKP sur une instance
# (n = 15, N = 100, graine 43 ; quelques secondes par mode) : REF, LD (SLD), LD0 et LD (DLD),
# puis comparaison du statut, de l'optimum et du nombre de nœuds avec les résultats de
# l'article (sld/results, lambda0/results, dld/results).
#
# À lancer AVANT les campagnes, depuis la racine du dépôt, sur un nœud de connexion :
# le premier appel GNN exporte chaque modèle en TorchScript à côté de son checkpoint
# (models/mkp/{sld,dld}/*scripted*.pt), ce qui demande un Python avec torch (venv du
# README) et un dépôt accessible en écriture — ce n'est pas le cas des nœuds de calcul
# de Trillium, où /project est en lecture seule.
#
#   bash test_paper/mkp/sanity_check.sh          # DEVICE=cpu par défaut
#   DEVICE=cuda bash test_paper/mkp/sanity_check.sh

set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"   # les binaires B&B doivent tourner depuis la racine du dépôt
DEVICE="${DEVICE:-cpu}"
BIN="$REPO/build/solver_bab"
source env/modules_trillium.sh
# Build CUDA (repéré au stub NVSHMEM) : bibliothèques CUDA nécessaires même avec DEVICE=cpu
[[ -e "$BIN/libnvshmem_host.so.3" ]] && source env/setup_gpu_env.sh
# Python avec torch pour l'export TorchScript (après les modules, qui fournissent aussi un python3)
VENV_PATH="${VENV_PATH:-$HOME/venvs/cccp}"
[[ -f "$VENV_PATH/bin/activate" ]] && source "$VENV_PATH/bin/activate"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

TD="$REPO/test_paper/mkp"
DATASET="${INSTANCES_DIR:-$TD/instances}/instance_n15_m5_N100_seed43.txt"
[[ -f "$DATASET" ]] || { echo "[ERREUR] instance introuvable : $DATASET (python3 test_paper/mkp/generate_test_instances.py)" >&2; exit 1; }
OUT="$(mktemp -d)"
echo "# logs : $OUT"
CKPT_SLD="models/mkp/sld/ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt"
CKPT_DLD="models/mkp/dld/ckpt_dld_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_dld_n10_20_500ep_ep0500.pt"
GNN_ARGS=(non 0.1 "$CKPT_SLD" "$DEVICE" python3 ipc sd_edit 0.0 0.0 1.0 64 1.0 surrogate 1 native)

"$BIN/bab_mkp_gecode" "$DATASET" 0.10 50 0 0 200 ref > "$OUT/ref.log" 2>&1
"$BIN/bab_mkp_gecode" "$DATASET" 0.10 50 0 0 200 gnn_batched "${GNN_ARGS[@]}" > "$OUT/gnn.log" 2>&1
GNN_ZERO_LAMBDA=1 "$BIN/bab_mkp_gecode" "$DATASET" 0.10 50 0 0 200 gnn_batched "${GNN_ARGS[@]}" > "$OUT/lambda0.log" 2>&1
"$BIN/bab_mkp_gecode_dld_gnn_batched" "$DATASET" "$CKPT_DLD" 0.10 0 0 "$DEVICE" 64 > "$OUT/dld.log" 2>&1

fail=0
check() {  # mode_parse log résultat_article
    local got want
    got=$(python3 "$TD/parse_result.py" "$1" 15 100 5 43 "$2" | cut -d, -f6,7,9)
    want=$(cut -d, -f6,7,9 "$3")
    if [[ "$got" == "$want" ]]; then echo "OK      $1 : $got (statut, optimum, nœuds)"
    else echo "ÉCART   $1 : obtenu $got, article $want (log : $2)"; fail=1; fi
}
check ref             "$OUT/ref.log"     "$TD/sld/results/ref_n15_N100_seed43.csv"
check gnn_batched     "$OUT/gnn.log"     "$TD/sld/results/gnn_n15_N100_seed43.csv"
check lambda0         "$OUT/lambda0.log" "$TD/lambda0/results/lambda0_n15_N100_seed43.csv"
check dld_gnn_batched "$OUT/dld.log"     "$TD/dld/results/dld_n15_N100_seed43.csv"
exit $fail
