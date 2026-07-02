"""
plot_bench_ssp_compare.py — Compare la convergence v1LD et v1CLD sur le SSP.

Usage :
    python tests/plot_bench_ssp_compare.py v1ld.csv v1cld.csv [--out fig.pdf]

Chaque CSV est produit par bench_ssp (v1LD) ou bench_ssp_v1cld (v1CLD).
Format attendu : même schéma que plot_bench_ssp.py (colonnes iter,time_s,cost_bound,best_cost,gnorm).
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: str):
    meta, rows, col_names = {}, [], None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                for m in re.finditer(r"([\w_]+)=([\d.eE+\-]+)", line):
                    try:
                        meta[m.group(1)] = float(m.group(2))
                    except ValueError:
                        pass
            elif col_names is None:
                col_names = line.split(",")
            else:
                rows.append([float(v) for v in line.split(",")])
    if not rows:
        return {}, meta
    arr = np.array(rows)
    return {name: arr[:, i] for i, name in enumerate(col_names)}, meta


def plot(ld_path: str, cld_path: str, out_path: str | None):
    data_ld,  meta_ld  = load_csv(ld_path)
    data_cld, meta_cld = load_csv(cld_path)

    E       = int(meta_ld.get("E", 0))
    T       = int(meta_ld.get("T", 0))
    N       = int(meta_ld.get("N", 0))
    eps     = meta_ld.get("eps", float("nan"))
    optimal = meta_ld.get("optimal", float("nan"))

    ld_star  = meta_ld.get("cost_lower",  float("nan"))
    cld_star = meta_cld.get("cost_lower", float("nan"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Panneau gauche : borne vs itérations ────────────────────────────────
    ax = axes[0]
    ax.plot(data_ld["iter"],  data_ld["best_cost"],
            color="steelblue", linewidth=2.0, label=f"v1LD  (best={ld_star:.3f})")
    ax.plot(data_cld["iter"], data_cld["best_cost"],
            color="darkorange", linewidth=2.0, label=f"v1CLD (best={cld_star:.3f})")
    ax.axhline(ld_star,  color="steelblue",  linewidth=1.0, linestyle="--", alpha=0.6)
    ax.axhline(cld_star, color="darkorange", linewidth=1.0, linestyle="--", alpha=0.6)
    if not np.isnan(optimal):
        ax.axhline(optimal, color="seagreen", linewidth=1.5, linestyle="-",
                   label=f"Optimal Gecode = {optimal:.3f}")
    ax.set_xlabel("Itération de sous-gradient")
    ax.set_ylabel("Borne inf. coût salarial  (= −L₁)")
    ax.set_title(f"v1LD vs v1CLD — SSP  (E={E}, T={T}, N={N}, ε={eps:.2f})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panneau droit : borne vs temps ──────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(data_ld["time_s"],  data_ld["best_cost"],
             color="steelblue", linewidth=2.0, label="v1LD")
    ax2.plot(data_cld["time_s"], data_cld["best_cost"],
             color="darkorange", linewidth=2.0, label="v1CLD")
    ax2.axhline(ld_star,  color="steelblue",  linewidth=1.0, linestyle="--", alpha=0.6)
    ax2.axhline(cld_star, color="darkorange", linewidth=1.0, linestyle="--", alpha=0.6)
    if not np.isnan(optimal):
        ax2.axhline(optimal, color="seagreen", linewidth=1.5, linestyle="-",
                    label=f"Optimal = {optimal:.3f}")
    ax2.set_xlabel("Temps écoulé (s)")
    ax2.set_ylabel("Borne inf. coût salarial  (= −L₁)")
    ax2.set_title("Convergence vs temps")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Annotation du gap pour chaque solveur
    if not np.isnan(optimal):
        for ax_, star, label, color in [
            (axes[0], ld_star,  "v1LD",  "steelblue"),
            (axes[0], cld_star, "v1CLD", "darkorange"),
        ]:
            gap = optimal - star
            pct = 100.0 * gap / optimal if optimal != 0 else 0.0
            ax_.annotate(
                f"{label} gap = {gap:.3f}  ({pct:.1f} %)",
                xy=(float(data_ld["iter"][-1]), (star + optimal) / 2),
                fontsize=8, color=color, ha="right",
            )

    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée : {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ld_csv",  help="CSV v1LD  (bench_ssp)")
    parser.add_argument("cld_csv", help="CSV v1CLD (bench_ssp_v1cld)")
    parser.add_argument("--out", default=None,
                        help="Chemin de sortie (pdf/png). Défaut : affichage interactif.")
    args = parser.parse_args()

    data_ld, _ = load_csv(args.ld_csv)
    if not data_ld:
        print(f"CSV v1LD vide ou mal formé : {args.ld_csv}", file=sys.stderr)
        sys.exit(1)
    data_cld, _ = load_csv(args.cld_csv)
    if not data_cld:
        print(f"CSV v1CLD vide ou mal formé : {args.cld_csv}", file=sys.stderr)
        sys.exit(1)

    plot(args.ld_csv, args.cld_csv, args.out)


if __name__ == "__main__":
    main()
