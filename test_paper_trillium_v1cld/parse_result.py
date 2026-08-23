#!/usr/bin/env python3
"""
parse_result.py — Parse la sortie stdout de bab_mkp_gecode (mode ref ou
gnn_batched) et extrait les métriques du test final CCMKP pour le rapport.

Usage :
  python3 parse_result.py <mode: ref|gnn_batched> <n> <N> <m> <seed> <logfile> >> results.csv
  (n'écrit PAS l'en-tête CSV — utiliser --header une fois en tête de pipeline)

  python3 parse_result.py --header
"""
import re
import sys

FIELDS = [
    "n", "N", "m", "seed", "mode", "status", "objective", "greedy_value",
    "nodes", "naive_nodes", "fails", "incumbent_improvements",
    "solve_time_s", "t_bab_pure_s",
    "t_bound_total_s", "t_gnn_graph_build_s", "t_gnn_forward_s", "t_gnn_bound_compute_s",
    "first_level_bound_value", "gap_root_pct",
    "n_bucket_flushes", "mean_bucket_fill_at_flush", "max_pending_live",
    "gpu_model", "K",
]


def _search(pattern, text, cast=str, default=""):
    m = re.search(pattern, text)
    if not m:
        return default
    try:
        return cast(m.group(1))
    except (ValueError, TypeError):
        return default


def parse(mode: str, text: str) -> dict:
    row = {k: "" for k in FIELDS}

    # gpu_model : extrait de la ligne "nvidia-smi --query-gpu=name,..." écrite en
    # tête de log par le script de job (absente en mode ref, qui n'utilise jamais
    # le GPU même si Trillium force --gpus-per-node=1 sur tous les jobs) — permet
    # de distinguer les instances relancées sur H200 (VRAM insuffisante en H100
    # pour certaines configs V1CLD, cf. investigation OOM du 2026-08-21) des
    # instances tournées normalement sur H100.
    row["gpu_model"] = _search(r"NVIDIA\s+(H\d+)", text)

    # K : taille de bucket GNN batché, extraite du header imprimé par le binaire
    # lui-même ("(K=64)" / "(K=8, sd_edit, native)") — absent en mode ref. Suivi
    # explicitement car le K utilisé pour V1CLD n'est PAS uniforme (K=64 comme
    # V1LD là où ça tient en mémoire, K=8 là où le graphe O(N_z*m*n_g²) sature
    # la VRAM même après micro-batching interne — cf. investigation OOM du
    # 2026-08-21/22, gnn_v1cld_sd_edit_native_batched.cpp).
    row["K"] = _search(r"\(K=(\d+)", text, int, "")

    row["status"] = _search(r"Statut\s*:\s*(\S+)", text)
    row["objective"] = _search(r"Valeur optimale\s*:\s*([\-0-9.eE]+)", text, float, float("nan"))
    row["greedy_value"] = _search(r"Incumbent initial \(greedy\)\s*:\s*([\-0-9.eE]+)", text, float, float("nan"))
    row["fails"] = _search(r"[ÉE]checs totaux\s*:\s*(\d+)", text, int, "")
    row["incumbent_improvements"] = _search(r"Am[ée]liorations incumbent\s*:\s*(\d+)", text, int, "")

    if mode == "ref":
        row["nodes"] = _search(r"N[œo]uds Gecode\s*:\s*(\d+)", text, int, "")
        row["naive_nodes"] = _search(r"N[œo]uds Gecode\s*:\s*\d+\s*/\s*(\d+)\s*na[iï]f", text, int, "")
        row["solve_time_s"] = _search(r"Temps total\s*:\s*([\-0-9.eE]+)\s*s", text, float, float("nan"))
        row["t_bab_pure_s"] = row["solve_time_s"]
    elif mode in ("gnn_batched", "v1cld_gnn_batched"):
        row["nodes"] = _search(r"N[œo]uds\s*:\s*(\d+)", text, int, "")
        row["solve_time_s"] = _search(r"Temps total\s*:\s*([\-0-9.eE]+)\s*s", text, float, float("nan"))
        row["t_bound_total_s"] = _search(r"Temps borne \(GNN batch[ée]e?\)\s*:\s*([\-0-9.eE]+)\s*s", text, float, float("nan"))
        row["t_gnn_graph_build_s"] = _search(r"construction graphe\s*:\s*([\-0-9.eE]+)\s*s", text, float, float("nan"))
        row["t_gnn_forward_s"] = _search(r"forward GNN\s*:\s*([\-0-9.eE]+)\s*s", text, float, float("nan"))
        row["t_gnn_bound_compute_s"] = _search(r"calcul borne \(ζ\)\s*:\s*([\-0-9.eE]+)\s*s", text, float, float("nan"))
        row["t_bab_pure_s"] = _search(r"Temps BAB pur\s*:\s*([\-0-9.eE]+)\s*s", text, float, float("nan"))
        row["n_bucket_flushes"] = _search(r"Flushes de bucket\s*:\s*(\d+)", text, int, "")
        row["mean_bucket_fill_at_flush"] = _search(r"Remplissage moyen au flush\s*:\s*([\-0-9.eE]+)", text, float, float("nan"))
        row["max_pending_live"] = _search(r"Pic de n[œo]uds en attente\s*:\s*(\d+)", text, int, "")

        if re.search(r"Borne GNN 1er niveau\s*:\s*N/A", text):
            row["first_level_bound_value"] = ""
            row["gap_root_pct"] = ""
        else:
            row["first_level_bound_value"] = _search(
                r"Borne GNN 1er niveau\s*:\s*([\-0-9.eE]+)", text, float, float("nan"))
            row["gap_root_pct"] = _search(
                r"gap vs optimal\s*:\s*([\-0-9.eE]+)%\)\n", text, float, float("nan"))
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
        print(f"Usage: {sys.argv[0]} <mode> <n> <N> <m> <seed> <logfile>", file=sys.stderr)
        print(f"       {sys.argv[0]} --header", file=sys.stderr)
        sys.exit(1)

    mode, n, N, m, seed, logfile = sys.argv[1:7]
    with open(logfile, "r", errors="replace") as f:
        text = f.read()

    row = parse(mode, text)
    row["n"], row["N"], row["m"], row["seed"], row["mode"] = n, N, m, seed, mode
    print(to_csv_row(row))


if __name__ == "__main__":
    main()
