#!/bin/bash
# gen_instances_list_ssp.sh — (re)génère test_paper_trillium/instances_list.txt
# à partir des dossiers ssp/E{E}_T{T}_N{N}/rep{0..9} déjà présents sur disque.
# Format d'une ligne : "E T N rep chemin_relatif" (cf. run_ref_array_ssp.sh /
# run_gnn_batched_array_ssp.sh, même convention que test_paper/instances_list.txt
# côté MKP).
#
# Usage :
#   bash test_paper_trillium/gen_instances_list_ssp.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

OUT="instances_list.txt"
> "$OUT"
missing=0
for E in 4 6 8; do
    for T in 8 10 12; do
        for N in 50 100 500; do
            for rep in $(seq 0 9); do
                d="ssp/E${E}_T${T}_N${N}/rep${rep}"
                if [[ -d "$d" ]]; then
                    echo "$E $T $N $rep $d" >> "$OUT"
                else
                    echo "MANQUANT: $d" >&2
                    missing=$((missing + 1))
                fi
            done
        done
    done
done

n=$(wc -l < "$OUT")
echo "-> $OUT : $n instances listées ($missing manquantes)"
