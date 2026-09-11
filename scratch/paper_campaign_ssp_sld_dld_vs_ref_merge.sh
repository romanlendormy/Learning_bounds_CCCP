#!/bin/bash
# scratch/paper_campaign_ssp_sld_dld_vs_ref_merge.sh — Fusionne les logs
# .out d'un job array solver_bab/ssp/campaign_sld_dld_vs_ref (REF + DLD
# + SLD, lancé via paper_campaign_ssp_sld_dld_vs_ref_full.sh ou
# _regenerated.sh) en un seul CSV. Garde la DERNIÈRE ligne CSV_ROW de chaque
# fichier .out (cf. campaign_sld_dld_vs_ref.cpp : écriture en deux temps
# REF+DLD puis SLD -- la dernière ligne est la plus complète disponible,
# même si la tâche a été tuée pendant SLD).
#
# Usage: paper_campaign_ssp_sld_dld_vs_ref_merge.sh <job_id> [logs_dir=logs] [out_csv] [prefix=campaign_sld_dld_vs_ref]
#   prefix : préfixe des fichiers .out avant "_<job_id>_<array_idx>.out"
#            (ex. "campaign_sld_dld_vs_ref_regen" pour le run des instances
#            régénérées, cf. paper_campaign_ssp_sld_dld_vs_ref_regenerated.sh).

set -uo pipefail
JOB_ID="${1:?Usage: merge_campaign_results.sh <job_id> [logs_dir] [out_csv] [prefix]}"
LOGS_DIR="${2:-logs}"
OUT_CSV="${3:-results/ssp_trillium_sld_dld_vs_ref_summary.csv}"
PREFIX="${4:-campaign_sld_dld_vs_ref}"

mkdir -p "$(dirname "$OUT_CSV")"

echo "instance_dir,E,T,N,rep,ref_status,ref_proven,ref_objective,ref_time_s,ref_nodes,sld_status,sld_time_s,sld_iterations_run,sld_lb_salary,sld_gap_pct,sld_n_stationary,dld_time_s,dld_iterations_run,dld_lb_salary,dld_gap_pct,dld_n_stationary,dld_t_zeta_theta_s" > "$OUT_CSV"

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
