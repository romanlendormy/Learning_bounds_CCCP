"""
plot_bench_ssp.py — Visualise la convergence de la borne v1LD sur le SSP.

Usage :
    python tests/plot_bench_ssp.py bench_ssp.csv [--out fig.pdf]

Le CSV est produit par ./build_cc/bench_ssp (bench_ssp.cpp).
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ── Lecture ────────────────────────────────────────────────────────────────────

def load_csv(path: str):
    """Lit le CSV et extrait les métadonnées depuis la ligne de commentaire."""
    meta = {}
    rows = []
    col_names = None

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
        return np.array([]), meta

    arr = np.array(rows)
    data = {name: arr[:, i] for i, name in enumerate(col_names)}
    return data, meta


# ── Figure ─────────────────────────────────────────────────────────────────────

def plot(data: dict, meta: dict, out_path: str | None):
    E       = int(meta.get("E",     0))
    T       = int(meta.get("T",     0))
    N       = int(meta.get("N",     0))
    eps     = meta.get("eps",        float("nan"))
    iters   = int(meta.get("iters", len(data)))
    total_s = meta.get("time_s",     float("nan"))
    cost_lb    = meta.get("cost_lower",    float("nan"))
    optimal    = meta.get("optimal",       float("nan"))
    gecode_s   = meta.get("gecode_time_s", float("nan"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # ── Panneau gauche : borne vs itérations ────────────────────────────────
    ax = axes[0]
    ax.plot(data["iter"], data["cost_bound"], color="steelblue", alpha=0.45,
            linewidth=1.0, label="Borne courante")
    ax.plot(data["iter"], data["best_cost"],  color="firebrick",
            linewidth=2.0, label="Meilleure borne")
    ax.axhline(cost_lb, color="firebrick", linewidth=1.0, linestyle="--",
               alpha=0.6, label=f"v1LD* = {cost_lb:.3f}")
    if not np.isnan(optimal):
        ax.axhline(optimal, color="seagreen", linewidth=1.5, linestyle="-",
                   label=f"Optimal Gecode = {optimal:.3f}  ({gecode_s:.1f} s)")

    ax.set_xlabel("Itération de sous-gradient")
    ax.set_ylabel("Borne inf. coût salarial  (= −L₁)")
    ax.set_title(f"Convergence v1LD — SSP  (E={E}, T={T}, N={N}, ε={eps:.2f})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panneau droit : borne vs temps ──────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(data["time_s"], data["cost_bound"], color="steelblue", alpha=0.45,
             linewidth=1.0, label="Borne courante")
    ax2.plot(data["time_s"], data["best_cost"],  color="firebrick",
             linewidth=2.0, label="Meilleure borne")
    ax2.axhline(cost_lb, color="firebrick", linewidth=1.0, linestyle="--", alpha=0.6,
                label=f"v1LD* = {cost_lb:.3f}")
    if not np.isnan(optimal):
        ax2.axhline(optimal, color="seagreen", linewidth=1.5, linestyle="-",
                    label=f"Optimal Gecode = {optimal:.3f}")

    ax2.set_xlabel("Temps écoulé (s)")
    ax2.set_ylabel("Borne inf. coût salarial  (= −L₁)")
    ax2.set_title(f"Convergence vs temps  ({iters} iter, {total_s:.1f} s total)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Annotation du resserrement
    first_cost = float(data["best_cost"][0])
    tightening = cost_lb - first_cost
    pct = 100.0 * tightening / abs(first_cost) if first_cost != 0 else 0.0
    axes[0].annotate(
        f"+{tightening:.2f}  (+{pct:.1f} %)",
        xy=(float(data["iter"][-1]), cost_lb),
        xytext=(float(data["iter"][len(data)//2]), first_cost * 0.92),
        fontsize=9, color="firebrick",
        arrowprops=dict(arrowstyle="->", color="firebrick", lw=1.0),
    )

    # Annotation du gap (optimal − v1LD*)
    if not np.isnan(optimal):
        gap = optimal - cost_lb
        gap_pct = 100.0 * gap / optimal if optimal != 0 else 0.0
        axes[0].annotate(
            f"gap = {gap:.3f}  ({gap_pct:.1f} %)",
            xy=(float(data["iter"][-1]), (cost_lb + optimal) / 2),
            fontsize=8, color="seagreen", ha="right",
        )

    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée : {out_path}")
    else:
        plt.show()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="Fichier CSV produit par bench_ssp")
    parser.add_argument("--out", default=None,
                        help="Chemin de sortie (pdf/png). Défaut : affichage interactif.")
    args = parser.parse_args()

    data, meta = load_csv(args.csv)
    if not data:
        print("CSV vide ou mal formé.", file=sys.stderr)
        sys.exit(1)

    plot(data, meta, args.out)


if __name__ == "__main__":
    main()
