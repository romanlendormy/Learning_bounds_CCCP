#!/bin/bash
# test_paper/mkp/aggregate_results.sh — Concatène les CSV par instance (une ligne chacun,
# écrits par parse_result.py) d'un dossier de résultats en un seul CSV avec en-tête.
#
# Usage (depuis la racine du dépôt) :
#   bash test_paper/mkp/aggregate_results.sh test_paper/mkp/sld/results > sld_results.csv
#   bash test_paper/mkp/aggregate_results.sh test_paper/mkp/sld/results ref_ > ref_results.csv

set -euo pipefail
TD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR="${1:?dossier de résultats}"
PREFIX="${2:-}"

python3 "$TD/parse_result.py" --header
cat "$DIR"/"$PREFIX"*.csv
echo "$(ls "$DIR"/"$PREFIX"*.csv | wc -l) lignes depuis $DIR/${PREFIX}*.csv" >&2
