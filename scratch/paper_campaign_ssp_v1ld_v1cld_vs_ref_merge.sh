#!/bin/bash
# scratch/paper_campaign_ssp_v1ld_v1cld_vs_ref_merge.sh — Fusionne les logs
# .out d'un job array solver_bab/ssp/campaign_v1ld_v1cld_vs_ref (REF + V1CLD
# + V1LD, lancé via paper_campaign_ssp_v1ld_v1cld_vs_ref_full.sh ou
# _regenerated.sh) en un seul CSV. Garde la DERNIÈRE ligne CSV_ROW de chaque
# fichier .out (cf. campaign_v1ld_v1cld_vs_ref.cpp : écriture en deux temps
# REF+V1CLD puis V1LD -- la dernière ligne est la plus complète disponible,
# même si la tâche a été tuée pendant V1LD).
#
# Usage: paper_campaign_ssp_v1ld_v1cld_vs_ref_merge.sh <job_id> [logs_dir=logs] [out_csv] [prefix=campaign_v1ld_v1cld_vs_ref]
#   prefix : préfixe des fichiers .out avant "_<job_id>_<array_idx>.out"
#            (ex. "campaign_v1ld_v1cld_vs_ref_regen" pour le run des instances
#            régénérées, cf. paper_campaign_ssp_v1ld_v1cld_vs_ref_regenerated.sh).

set -uo pipefail
JOB_ID="${1:?Usage: merge_campaign_results.sh <job_id> [logs_dir] [out_csv] [prefix]}"
LOGS_DIR="${2:-logs}"
OUT_CSV="${3:-results/ssp_trillium_v1ld_v1cld_vs_ref_summary.csv}"
PREFIX="${4:-campaign_v1ld_v1cld_vs_ref}"

mkdir -p "$(dirname "$OUT_CSV")"

echo "instance_dir,E,T,N,rep,ref_status,ref_proven,ref_objective,ref_time_s,ref_nodes,v1ld_status,v1ld_time_s,v1ld_iterations_run,v1ld_lb_salary,v1ld_gap_pct,v1ld_n_stationary,v1cld_time_s,v1cld_iterations_run,v1cld_lb_salary,v1cld_gap_pct,v1cld_n_stationary,v1cld_t_zeta_theta_s" > "$OUT_CSV"

n_ok=0
n_missing=0
for f in "$LOGS_DIR"/${PREFIX}_${JOB_ID}_*.out; do
    last_row=$(grep '^CSV_ROW,' "$f" | tail -1)
    if [[ -n "$last_row" ]]; then
        echo "${last_row#CSV_ROW,}" >> "$OUT_CSV"
        n_ok=$((n_ok + 1))
    else
        echo "AUCUNE ligne CSV_ROW dans $f" >&2
        n_missing=$((n_missing + 1))
    fi
done

echo "Fusion terminée : $n_ok lignes écrites, $n_missing fichiers sans CSV_ROW -> $OUT_CSV"
