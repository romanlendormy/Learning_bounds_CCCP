#!/bin/bash
# aggregate_results.sh — Concatène les CSV par run (results/v1cld_*.csv,
# écrits par parse_result.py dans run_v1cld_array.sh) en un seul results.csv,
# avec en-tête. À lancer une fois les 120 tâches de l'array terminées.
#
# Usage :
#   bash test_paper/mkp/trillium/v1cld/aggregate_results.sh

set -euo pipefail

TESTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

N_V1CLD=$(ls "$TESTDIR"/results/v1cld_*.csv 2>/dev/null | wc -l)
echo "Résultats trouvés : $N_V1CLD/120 (v1cld)"

python3 "$TESTDIR/parse_result.py" --header > "$TESTDIR/results.csv"
cat "$TESTDIR"/results/v1cld_*.csv 2>/dev/null >> "$TESTDIR/results.csv"

echo "-> $TESTDIR/results.csv ($(( $(wc -l < "$TESTDIR/results.csv") - 1 )) lignes de données)"

if [[ "$N_V1CLD" -lt 120 ]]; then
    echo ""
    echo "ATTENTION : incomplet. Instances manquantes :"
    awk '{print "v1cld_n" $1 "_N" $2 "_seed" $4}' "$TESTDIR/instances_list.txt" | sort \
        > /tmp/expected_results_$$.txt
    ls "$TESTDIR"/results | sed 's/\.csv$//' | sort > /tmp/actual_results_$$.txt
    comm -23 /tmp/expected_results_$$.txt /tmp/actual_results_$$.txt || true
    rm -f /tmp/expected_results_$$.txt /tmp/actual_results_$$.txt
fi
