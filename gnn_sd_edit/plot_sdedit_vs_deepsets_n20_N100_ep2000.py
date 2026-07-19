"""
Evolution de la borne L1(lambda) selon les epochs — sd_edit (hidden=128,
2 couches MP) vs DeepSetsGNN (hidden=64, aucun message passing), overfit_single
sur la meme instance (n=20, N=100), 2000 epochs, avec lignes de reference
pointillees pour la methode sous-gradient a T=50 et T=100 iterations.

Source des donnees : metrics.json ecrits par diag_overfit_single.py, non retouches.
    sd_edit   : gnn_sd_edit/runs/overfit_single/N100_n20_inst0_d0_idx0_sdedit_h128_ep2000/metrics.json
    deepsets  : gnn_sd_edit/runs/overfit_single/N100_n20_inst0_d0_idx0_deepsets_ep2000/metrics.json
Meme instance verifiee (instance.opt=5489.58 identique dans les deux fichiers).

Usage : python gnn_sd_edit/plot_sdedit_vs_deepsets_n20_N100_ep2000.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_SDEDIT = REPO_ROOT / "gnn_sd_edit" / "runs" / "overfit_single" / "N100_n20_inst0_d0_idx0_sdedit_h128_ep2000"
RUN_DEEPSETS = REPO_ROOT / "gnn_sd_edit" / "runs" / "overfit_single" / "N100_n20_inst0_d0_idx0_deepsets_ep2000"
OUT_PATH = REPO_ROOT / "gnn_sd_edit" / "runs" / "fig_sdedit_vs_deepsets_n20_N100_ep2000.png"

COLOR_SDEDIT = "#2a78d6"     # bleu — palette categorique slot 1
COLOR_DEEPSETS = "#4a3aa7"   # violet — palette categorique slot 7
COLOR_REF50 = "gray"
COLOR_REF100 = "dimgray"


def main():
    m_sdedit = json.loads((RUN_SDEDIT / "metrics.json").read_text())
    m_deepsets = json.loads((RUN_DEEPSETS / "metrics.json").read_text())

    assert m_sdedit["instance"]["opt"] == m_deepsets["instance"]["opt"], \
        "instances differentes — comparaison invalide"

    refs = m_sdedit["references"]  # identiques dans les deux fichiers (meme instance)

    fig, ax = plt.subplots(figsize=(9, 6.5))

    h_sdedit = m_sdedit["history"]
    ax.plot(h_sdedit["epochs"], h_sdedit["l1"], color=COLOR_SDEDIT, lw=1.6,
             marker=".", ms=4, label="L1(λ) — sd_edit (hidden=128, 2 couches)", zorder=3)
    ax.scatter([h_sdedit["epochs"][-1]], [m_sdedit["final_l1"]], color=COLOR_SDEDIT, s=50, zorder=4)
    ax.annotate(f"  {m_sdedit['final_l1']:.0f}",
                (h_sdedit["epochs"][-1], m_sdedit["final_l1"]),
                va="bottom", fontsize=9, color=COLOR_SDEDIT)

    h_deepsets = m_deepsets["history"]
    ax.plot(h_deepsets["epochs"], h_deepsets["l1"], color=COLOR_DEEPSETS, lw=1.6,
             marker=".", ms=4, label="L1(λ) — DeepSetsGNN (hidden=64, sans message passing)", zorder=3)
    ax.scatter([h_deepsets["epochs"][-1]], [m_deepsets["final_l1"]], color=COLOR_DEEPSETS, s=50, zorder=4)
    ax.annotate(f"  {m_deepsets['final_l1']:.0f}",
                (h_deepsets["epochs"][-1], m_deepsets["final_l1"]),
                va="top", fontsize=9, color=COLOR_DEEPSETS)

    ax.axhline(refs["l1_ref50"], color=COLOR_REF50, lw=1.4, ls="--",
               label=f"Sous-gradient T=50 = {refs['l1_ref50']:.0f}", zorder=2)
    ax.axhline(refs["l1_ref100"], color=COLOR_REF100, lw=1.4, ls="--",
               label=f"Sous-gradient T=100 = {refs['l1_ref100']:.0f}", zorder=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("L1(λ) résiduelle — borne (plus bas = mieux)")
    ax.set_title(
        "sd_edit (h=128, 2 couches) vs DeepSetsGNN (h=64, sans message passing) — overfit_single\n"
        f"n=20, N=100, même instance (opt={m_sdedit['instance']['opt']:.0f}), 2000 epochs"
    )
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.text(0.01, 0.01,
              "Overfit = mémorisation pure sur une seule instance (pas un test de généralisation).",
              fontsize=8, color="dimgray")
    plt.tight_layout(rect=(0, 0.02, 1, 1))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Figure sauvegardée : {OUT_PATH}")


if __name__ == "__main__":
    main()
