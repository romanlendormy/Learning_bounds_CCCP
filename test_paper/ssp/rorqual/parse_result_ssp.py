#!/usr/bin/env python3
"""
parse_result_ssp.py — Parse la sortie stdout de test_gnn_v1cld_bab (mode ref)
ou de run_gnn_batched_array_ssp.sh (test_gnn_v1cld_bab_batched + run
auxiliaire test_gnn_v1cld_bab mode gnn pour le gap racine) et extrait les
métriques pour la campagne test_paper/ssp/rorqual/ssp.

Format de sortie différent de test_paper/mkp/trillium/v1ld/parse_result.py (MKP) : binaires
différents ("[test] ...", "[gnn_timing] ..."), pas de mode string ni de
chaînes "Statut :"/"Valeur optimale :" côté SSP.

Pour le mode gnn_batched, le log contient DEUX sections séparées par le
marqueur "=== SECTION: root_gap_sidecar ===" (posé par
run_gnn_batched_array_ssp.sh) : la section principale (test_gnn_v1cld_bab_batched,
K=8) fournit status/objective/nodes/solve_time_s/répartition du temps ; la
section sidecar (test_gnn_v1cld_bab mode gnn, run court 60s) fournit
UNIQUEMENT root_bound_value (borne réelle au nœud racine, capturée une seule
fois au tout premier nœud -- cf. gecode_bab_ssp.hpp "is_first"). Le
status/nodes/solve_time_s de la section sidecar sont ignorés (run tronqué à
60s, non représentatif du run principal).

gap_root_pct = 100 * (objective_run_principal - root_bound_value) / objective_run_principal
(convention gap >= 0 pour une borne inférieure valide, problème SSP = minimisation ;
utilise l'objectif du run PRINCIPAL lui-même comme référence -- avec
time_lim=0 côté gnn_batched, ce run est exhaustif et son objectif à
convergence EST l'optimum prouvé, pas besoin de dépendre du job ref séparé).

Usage :
  python3 parse_result_ssp.py <mode: ref|gnn_batched> <E> <T> <N> <rep> <logfile> >> results.csv
  python3 parse_result_ssp.py --header
"""
import re
import sys

FIELDS = [
    "E", "T", "N", "rep", "mode", "status", "objective", "feasible", "optimal",
    "nodes", "fails", "solve_time_s",
    "K", "n_bucket_flushes", "mean_bucket_fill_at_flush", "max_pending_live",
    "t_graph_build_s", "t_forward_gnn_s", "t_zeta_theta_s", "t_lagrangian_rest_s",
    "pct_graph_build", "pct_forward_gnn", "pct_zeta_theta", "pct_lagrangian_rest",
    "root_bound_value", "root_bound_time_s", "gap_root_pct",
]

SIDECAR_MARKER = "=== SECTION: root_gap_sidecar ==="


def _search(pattern, text, cast=str, default=""):
    m = re.search(pattern, text)
    if not m:
        return default
    try:
        return cast(m.group(1))
    except (ValueError, TypeError):
        return default


def parse_primary(text: str) -> dict:
    row = {}
    # Champs indépendamment ancrés (PAS chaînés les uns après les autres) :
    # bug trouvé sur le smoke test (2026-08-22) -- un ancien regex exigeait
    # "objective=[0-9...]" juste avant "feasible=", et objective=nan (run
    # TIMEOUT/INFEASIBLE, aucune solution trouvée) ne matche pas cette classe
    # de caractères (pas de chiffre dans "nan") : tout le regex chaîné
    # échouait, laissant "feasible" vide au lieu de "0" alors que la valeur
    # est bien présente telle quelle dans le texte. status/objective restent
    # ancrés ensemble (peu de risque, "status=" est unique dans le bloc
    # [test]), mais feasible/optimal sont maintenant cherchés indépendamment.
    row["status"] = _search(r"\[test\] status=(\S+)", text)
    row["objective"] = _search(r"\[test\] status=\S+\s+objective=(\S+)", text, float, float("nan"))
    row["feasible"] = _search(r"\bfeasible=(\d)\b", text, int, "")
    row["optimal"] = _search(r"\boptimal=(\d)\b", text, int, "")
    row["nodes"] = _search(r"\[test\] nodes=(\d+)", text, int, "")
    row["fails"] = _search(r"\[test\] nodes=\d+\s+fails=(\d+)", text, int, "")
    row["solve_time_s"] = _search(r"\[test\] solve_time_s=(\S+)", text, float, float("nan"))

    # Champs présents seulement pour gnn_batched
    row["K"] = _search(r"\bK=(\d+)", text, int, "")
    row["n_bucket_flushes"] = _search(r"n_bucket_flushes=(\d+)", text, int, "")
    row["mean_bucket_fill_at_flush"] = _search(r"mean_fill=(\S+)", text, float, float("nan"))
    row["max_pending_live"] = _search(r"max_pending_live=(\d+)", text, int, "")

    row["t_graph_build_s"] = _search(r"\[gnn_timing\] graph_build\s*:\s*(\S+)s", text, float, float("nan"))
    row["t_forward_gnn_s"] = _search(r"\[gnn_timing\] forward_gnn\s*:\s*(\S+)s", text, float, float("nan"))
    row["t_zeta_theta_s"] = _search(r"\[gnn_timing\] zeta_theta\s*:\s*(\S+)s", text, float, float("nan"))
    row["t_lagrangian_rest_s"] = _search(r"\[gnn_timing\] lagrangian_rest\s*:\s*(\S+)s", text, float, float("nan"))

    row["pct_graph_build"] = _search(r"\[gnn_timing\] graph_build\s*:\s*\S+s\s*\((\S+)%\)", text, float, float("nan"))
    row["pct_forward_gnn"] = _search(r"\[gnn_timing\] forward_gnn\s*:\s*\S+s\s*\((\S+)%\)", text, float, float("nan"))
    row["pct_zeta_theta"] = _search(r"\[gnn_timing\] zeta_theta\s*:\s*\S+s\s*\((\S+)%\)", text, float, float("nan"))
    row["pct_lagrangian_rest"] = _search(r"\[gnn_timing\] lagrangian_rest\s*:\s*\S+s\s*\((\S+)%\)", text, float, float("nan"))

    return row


def parse_sidecar_root_bound(text: str):
    # \S+ (pas [\-0-9.eE]+) : root_bound_value peut valoir "nan" si le nœud
    # racine n'a jamais appelé le GNN (résolu par pure propagation CP avant
    # tout calcul de borne) -- cf. is_first dans gecode_bab_ssp.hpp.
    m = re.search(r"\[test\] root_bound_value=(\S+)\s+root_bound_time_s=(\S+)", text)
    if not m:
        return float("nan"), float("nan")
    try:
        return float(m.group(1)), float(m.group(2))
    except ValueError:
        return float("nan"), float("nan")


def parse(mode: str, text: str) -> dict:
    row = {k: "" for k in FIELDS}

    if mode == "ref":
        primary = parse_primary(text)
        row.update(primary)
        # ref n'a pas de borne -> pas de gap racine
    elif mode == "gnn_batched":
        if SIDECAR_MARKER in text:
            primary_text, sidecar_text = text.split(SIDECAR_MARKER, 1)
        else:
            primary_text, sidecar_text = text, ""
        primary = parse_primary(primary_text)
        row.update(primary)

        root_bound_value, root_bound_time_s = parse_sidecar_root_bound(sidecar_text)
        row["root_bound_value"] = root_bound_value
        row["root_bound_time_s"] = root_bound_time_s

        obj = row.get("objective")
        if (
            isinstance(obj, float) and obj not in ("", None) and obj == obj  # pas NaN
            and obj != 0.0
            and root_bound_value == root_bound_value  # pas NaN
        ):
            row["gap_root_pct"] = 100.0 * (obj - root_bound_value) / obj
        else:
            row["gap_root_pct"] = float("nan")
    else:
        raise ValueError(f"mode inconnu : {mode}")

    return row


def to_csv_row(row: dict) -> str:
    return ",".join(str(row.get(k, "")) for k in FIELDS)


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--header":
        print(",".join(FIELDS))
        return

    if len(sys.argv) != 7:
        print(f"Usage: {sys.argv[0]} <mode> <E> <T> <N> <rep> <logfile>", file=sys.stderr)
        print(f"       {sys.argv[0]} --header", file=sys.stderr)
        sys.exit(1)

    mode, E, T, N, rep, logfile = sys.argv[1:7]
    with open(logfile, "r", errors="replace") as f:
        text = f.read()

    row = parse(mode, text)
    row["E"], row["T"], row["N"], row["rep"], row["mode"] = E, T, N, rep, mode
    print(to_csv_row(row))


if __name__ == "__main__":
    main()
