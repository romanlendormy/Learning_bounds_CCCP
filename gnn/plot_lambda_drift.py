"""
Génère six figures (PNG) :

  Diag overfit 100 instances :
    fig_diag_lambda_avg.png / fig_diag_lambda_max.png / fig_diag_loss.png
    tanh   → logs/slurm_diag_overfit100_16823759.out
    sans   → logs/slurm_diag_overfit100_16813199.out
    sg ref → logs/slurm_diag_overfit100_16813001.out

  Train scaled ~10k instances :
    fig_scaled_lambda_avg.png / fig_scaled_lambda_max.png / fig_scaled_loss.png
    tanh   → logs/slurm_train_scaled_16824232.out
    sans   → logs/slurm_train_scaled_16823445.out

Usage : python gnn/plot_lambda_drift.py
"""

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(__file__).parent.parent
LOGS = ROOT / "logs"
OUT  = LOGS


# ── Extraction depuis les logs ─────────────────────────────────────────────

def parse_log(path: Path) -> tuple[list, list, list, list]:
    """
    Retourne (epochs, loss, lam_avg, lam_max) depuis les lignes de résumé d'epoch.
    Supporte les deux formats :
      loss_norm=X … |λ|avg=Y(max=Z)   (diag_overfit)
      loss=X±…    … |λ|=Y(max=Z)      (train_scaled)
    """
    epochs, losses, avgs, maxs = [], [], [], []
    pat = re.compile(
        r"ep\s+(\d+)/\d+"
        r".*?loss(?:_norm)?=([\d.]+)"
        r".*?\|λ\|(?:avg)?=([\d.]+)\(max=([\d.]+)\)"
    )
    with open(path) as f:
        for line in f:
            if line.startswith("[warmup]"):
                continue
            m = pat.search(line)
            if m:
                epochs.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                avgs.append(float(m.group(3)))
                maxs.append(float(m.group(4)))
    return epochs, losses, avgs, maxs


def parse_subgrad_ref(path: Path) -> tuple[float, float]:
    """Retourne (|λ*|avg, |λ*|max_mean) depuis le bloc [subgrad_full]."""
    avg_ref = max_ref = None
    with open(path) as f:
        for line in f:
            m = re.search(r"\|λ\*\|avg\s*:\s*mean=([\d.]+)", line)
            if m:
                avg_ref = float(m.group(1))
            m = re.search(r"\|λ\*\|max\s*:\s*mean=([\d.]+)", line)
            if m:
                max_ref = float(m.group(1))
    if avg_ref is None or max_ref is None:
        sys.exit(f"Référence sous-gradient introuvable dans {path}")
    return avg_ref, max_ref


# ── Style ──────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linewidth":    0.5,
    "lines.linewidth":   1.8,
    "figure.dpi":        200,
})

BLUE   = "#1C3461"
ORANGE = "#B85C1A"
GRAY   = "#555555"


def save_fig(fig, fname):
    out = OUT / fname
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved → {out}")


def make_lambda_figures(ep_tanh, y_tanh_avg, y_tanh_max,
                        ep_notanh, y_notanh_avg, y_notanh_max,
                        sg_avg, sg_max, prefix: str, log_scale: bool = False):
    configs = [
        ("avg", y_tanh_avg, y_notanh_avg, sg_avg,
         r"$|\lambda|_{\mathrm{avg}}$", f"{prefix}_lambda_avg.png"),
        ("max", y_tanh_max, y_notanh_max, sg_max,
         r"$|\lambda|_{\mathrm{max}}$", f"{prefix}_lambda_max.png"),
    ]
    for metric, y_tanh, y_notanh, sg_ref, ylabel, fname in configs:
        fig, ax = plt.subplots(figsize=(5.0, 3.2))
        ax.plot(ep_notanh, y_notanh, color=ORANGE, label=r"sans $\mathtt{tanh}$")
        ax.plot(ep_tanh,   y_tanh,   color=BLUE,   label=r"avec $\mathtt{tanh}$ fix")
        if sg_ref is not None:
            ax.axhline(sg_ref, color=GRAY, linestyle="--", linewidth=1.3,
                       label=rf"s.-grad. complet  $|{{\lambda}}^*|_{{\mathrm{{{metric}}}}} = {sg_ref:.2f}$")
        if log_scale:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=10))
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, max(ep_tanh[-1], ep_notanh[-1]))
        ax.legend(loc="upper left", framealpha=0.92, edgecolor="#cccccc")
        fig.tight_layout(pad=0.6)
        save_fig(fig, fname)


def make_loss_figure(ep_tanh, loss_tanh, ep_notanh, loss_notanh,
                     prefix: str, ylabel: str = r"Loss normalisée"):
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(ep_notanh, loss_notanh, color=ORANGE, label=r"sans $\mathtt{tanh}$")
    ax.plot(ep_tanh,   loss_tanh,   color=BLUE,   label=r"avec $\mathtt{tanh}$ fix")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, max(ep_tanh[-1], ep_notanh[-1]))
    ax.legend(loc="upper right", framealpha=0.92, edgecolor="#cccccc")
    fig.tight_layout(pad=0.6)
    save_fig(fig, f"{prefix}_loss.png")


# ── Diag overfit ───────────────────────────────────────────────────────────

print("=== Diag overfit ===")
ep_t,      loss_t,      avg_t,      max_t      = parse_log(LOGS / "slurm_diag_overfit100_16823759.out")
ep_n_lam,  loss_n_lam,  avg_n,      max_n      = parse_log(LOGS / "slurm_diag_overfit100_16813199.out")
ep_n_loss, loss_n_loss, _,          _          = parse_log(LOGS / "slurm_diag_overfit100_16849109.out")
sg_avg, sg_max                                  = parse_subgrad_ref(LOGS / "slurm_diag_overfit100_16813001.out")
print(f"  tanh      : {len(ep_t)} pts, final loss={loss_t[-1]:.4f}  avg={avg_t[-1]:.2f}  max={max_t[-1]:.2f}")
print(f"  sans (λ)  : {len(ep_n_lam)} pts, final loss={loss_n_lam[-1]:.4f}  avg={avg_n[-1]:.2f}  max={max_n[-1]:.2f}")
print(f"  sans (loss): {len(ep_n_loss)} pts, final loss={loss_n_loss[-1]:.4f}")
print(f"  sg ref    : avg={sg_avg:.4f}  max={sg_max:.4f}")
make_lambda_figures(ep_t, avg_t, max_t, ep_n_lam, avg_n, max_n,
                    sg_avg, sg_max, prefix="fig_diag", log_scale=False)
make_loss_figure(ep_t, loss_t, ep_n_loss, loss_n_loss,
                 prefix="fig_diag", ylabel="Loss normalisée")

# ── Train scaled ───────────────────────────────────────────────────────────

print("=== Train scaled ===")
ep_t, loss_t, avg_t, max_t = parse_log(LOGS / "slurm_train_scaled_16824232.out")
ep_n, loss_n, avg_n, max_n = parse_log(LOGS / "slurm_train_scaled_16823445.out")
print(f"  tanh   : {len(ep_t)} pts, final loss={loss_t[-1]:.4f}  avg={avg_t[-1]:.2f}  max={max_t[-1]:.2f}")
print(f"  sans   : {len(ep_n)} pts, final loss={loss_n[-1]:.4f}  avg={avg_n[-1]:.2f}  max={max_n[-1]:.2f}")
make_lambda_figures(ep_t, avg_t, max_t, ep_n, avg_n, max_n,
                    sg_avg=None, sg_max=None, prefix="fig_scaled", log_scale=True)
make_loss_figure(ep_t, loss_t, ep_n, loss_n,
                 prefix="fig_scaled", ylabel="Loss")
