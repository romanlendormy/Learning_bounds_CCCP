#!/bin/bash
# aggregate_results.sh — Concatène les CSV par run (results/*.csv, écrits
# par parse_result.py dans run_ref_array.sh / run_gnn_array.sh) en un seul
# results.csv, avec en-tête. À lancer une fois les 240 tâches des deux array
# jobs terminées.
#
# Usage :
#   bash test_paper/mkp/trillium/sld/aggregate_results.sh

set -euo pipefail

TESTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

N_REF=$(ls "$TESTDIR"/results/ref_*.csv 2>/dev/null | wc -l)
N_GNN=$(ls "$TESTDIR"/results/gnn_*.csv 2>/dev/null | wc -l)
echo "Résultats trouvés : $N_REF/120 (ref)  $N_GNN/120 (gnn_batched)"

python3 "$TESTDIR/parse_result.py" --header > "$TESTDIR/results.csv"
cat "$TESTDIR"/results/ref_*.csv "$TESTDIR"/results/gnn_*.csv 2>/dev/null >> "$TESTDIR/results.csv"

echo "-> $TESTDIR/results.csv ($(( $(wc -l < "$TESTDIR/results.csv") - 1 )) lignes de données)"

if [[ "$N_REF" -lt 120 || "$N_GNN" -lt 120 ]]; then
    echo ""
    echo "ATTENTION : incomplet. Instances manquantes :"
    awk '{print "ref_n" $1 "_N" $2 "_seed" $4; print "gnn_n" $1 "_N" $2 "_seed" $4}' "$TESTDIR/instances_list.txt" | sort \
        > /tmp/expected_results_$$.txt
    ls "$TESTDIR"/results | sed 's/\.csv$//' | sort > /tmp/actual_results_$$.txt
    comm -23 /tmp/expected_results_$$.txt /tmp/actual_results_$$.txt || true
    rm -f /tmp/expected_results_$$.txt /tmp/actual_results_$$.txt
fi
