#!/bin/bash
# aggregate_results_ssp.sh — Concatène les CSV par run
# (test_paper_trillium/results/*.csv, écrits par parse_result_ssp.py dans
# run_ref_array_ssp.sh / run_gnn_batched_array_ssp.sh) en un seul
# test_paper_trillium/results_ssp.csv, avec en-tête. À lancer une fois les
# 540 tâches des deux array jobs terminées (ou pour un sous-ensemble, ex.
# smoke test).
#
# Usage :
#   bash test_paper_trillium/aggregate_results_ssp.sh

set -euo pipefail

REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
TESTDIR="$REPO/test_paper_trillium"

N_REF=$(ls "$TESTDIR"/results/ref_*.csv 2>/dev/null | wc -l)
N_GNN=$(ls "$TESTDIR"/results/gnn_batched_*.csv 2>/dev/null | wc -l)
echo "Résultats trouvés : $N_REF/270 (ref)  $N_GNN/270 (gnn_batched)"

python3 "$TESTDIR/parse_result_ssp.py" --header > "$TESTDIR/results_ssp.csv"
cat "$TESTDIR"/results/ref_*.csv "$TESTDIR"/results/gnn_batched_*.csv 2>/dev/null >> "$TESTDIR/results_ssp.csv"

echo "-> $TESTDIR/results_ssp.csv ($(( $(wc -l < "$TESTDIR/results_ssp.csv") - 1 )) lignes de données)"

if [[ "$N_REF" -lt 270 || "$N_GNN" -lt 270 ]]; then
    echo ""
    echo "ATTENTION : incomplet. Instances manquantes :"
    awk '{print "ref_E" $1 "_T" $2 "_N" $3 "_rep" $4; print "gnn_batched_E" $1 "_T" $2 "_N" $3 "_rep" $4}' \
        "$TESTDIR/instances_list.txt" | sort > /tmp/expected_results_ssp_$$.txt
    ls "$TESTDIR"/results | sed 's/\.csv$//' | sort > /tmp/actual_results_ssp_$$.txt
    comm -23 /tmp/expected_results_ssp_$$.txt /tmp/actual_results_ssp_$$.txt || true
    rm -f /tmp/expected_results_ssp_$$.txt /tmp/actual_results_ssp_$$.txt
fi
