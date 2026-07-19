"""
Génère fig_quality_<run_name>.png depuis un checkpoint existant.
Usage :
    python gnn/gen_quality_fig.py <checkpoint.pt> [--data_dir <path>]
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import torch

from gnn.train_scaled_v1ld import (
    discover_pairs,
    load_and_merge,
    select_stratified,
    split_by_instance,
)
from gnn.train_gnn_v1ld import (
    LazyGraphDataset,
    EagerGraphDataset,
    GNN,
    lagrangian_v1ld,
    verify_v1ld,
)

_N_COLORS = {10: "#1f77b4", 15: "#ff7f0e", 20: "#2ca02c"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--data_dir", type=Path, default=None)
    p.add_argument("--device",   default="auto")
    p.add_argument("--split",    choices=["train", "test"], default="test",
                   help="Évaluer sur le split train ou test (défaut: test)")
    cli = p.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if cli.device == "auto" else torch.device(cli.device))

    print(f"Chargement checkpoint : {cli.checkpoint}")
    ckpt = torch.load(cli.checkpoint, map_location=device)
    a = argparse.Namespace(**ckpt["args"])

    data_dir = cli.data_dir if cli.data_dir else REPO_ROOT / "data"
    out_dir  = Path(a.out_dir)

    # Reconstruit run_name
    tag = f"scaled_n{'_'.join(str(n) for n in sorted(a.n_list))}"
    tag_suffix = f"_{a.run_tag}" if a.run_tag else ""
    run_name = (f"{tag}_ep{a.n_epochs}_lr{a.lr}_bs{a.batch_size}"
                f"_h{a.hidden}_seed{a.seed}{tag_suffix}")
    print(f"run : {run_name}")

    # Données — même pipeline que l'entraînement
    print("Découverte des données ...")
    pairs = discover_pairs(data_dir, a.n_list, a.n_scens_n10, a.m)
    if not pairs:
        sys.exit("ERREUR : aucun fichier trouvé — vérifiez --data_dir")
    print("Chargement ...")
    instances, all_records = load_and_merge(pairs, a.max_load)
    selected = select_stratified(all_records, a.target_graphs, a.seed)
    all_ds   = LazyGraphDataset(instances, selected)
    train_ds, test_ds = split_by_instance(instances, all_ds, a.train_frac, a.seed)

    src_ds = train_ds if cli.split == "train" else test_ds
    root_recs = [r for r in src_ds.raw_records if r["depth"] == 0]
    n_eval = min(a.eval_sample, len(root_recs))
    random.Random(a.seed).shuffle(root_recs)
    eval_lazy  = LazyGraphDataset(instances, root_recs[:n_eval])
    print(f"Construction eval set ({n_eval} racines) ...")
    eager_eval  = EagerGraphDataset(eval_lazy)
    eval_graphs = [eager_eval[i].to(device) for i in range(len(eager_eval))]

    # Modèle
    model = GNN(m=a.m, hidden=a.hidden, lam_max=a.lam_max,
                use_max_pool=a.use_max_pool).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Modèle chargé (epoch {ckpt['epoch']})")

    # Graphes éligibles
    quality_eligible = [
        (g, int(g.n_orig.item()) if hasattr(g, "n_orig") else 0)
        for g in eval_graphs
        if (int(g.n_orig.item()) if hasattr(g, "n_orig") else 0) <= a.quality_max_n
    ]
    print(f"[quality] {len(quality_eligible)} graphes (n≤{a.quality_max_n}), T={a.quality_T}")

    q_subgrad, q_gnn, q_n = [], [], []
    t0 = time.time()
    for j, (g, n_val) in enumerate(quality_eligible):
        g_cpu = g.cpu()
        sg = verify_v1ld(g_cpu, T=a.quality_T)
        g_dev = g.to(device)
        with torch.no_grad():
            gnn_l1 = lagrangian_v1ld(model(g_dev), g_dev).item()
        q_subgrad.append(sg)
        q_gnn.append(gnn_l1)
        q_n.append(n_val)
        if (j + 1) % 50 == 0 or j == len(quality_eligible) - 1:
            print(f"  [{j+1}/{len(quality_eligible)}]  t={time.time()-t0:.0f}s")

    diffs = [gnn - sg for sg, gnn in zip(q_subgrad, q_gnn)]
    print(f"[quality] GNN-subgrad: mean={np.mean(diffs):.4f}  "
          f"max={np.max(diffs):.4f}  "
          f"pct_better={100*np.mean(np.array(diffs)<0):.1f}%")

    fig_q_path = out_dir / f"fig_quality_{run_name}_{cli.split}.png"
    fig, ax = plt.subplots(figsize=(7, 6))
    for sg, gnn, nv in zip(q_subgrad, q_gnn, q_n):
        ax.scatter(sg, gnn, color=_N_COLORS.get(nv, "gray"), alpha=0.7, s=25, zorder=3)
    all_v = q_subgrad + q_gnn
    mn, mx = min(all_v), max(all_v)
    ax.plot([mn, mx], [mn, mx], "k--", lw=1.2, label="y = x (identique)")
    for nv, color in sorted(_N_COLORS.items()):
        if any(n == nv for n in q_n):
            ax.scatter([], [], color=color, s=40, label=f"n={nv}")
    ax.set_xlabel(f"L1 sous-gradient (T={a.quality_T})")
    ax.set_ylabel("L1 GNN (final)")
    ax.set_title(
        f"Qualité GNN vs sous-gradient [{cli.split}] — {run_name}\n"
        f"GNN−subgrad : mean={np.mean(diffs):.4f}  "
        f"pct_better={100*np.mean(np.array(diffs)<0):.1f}%"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_q_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure sauvegardée : {fig_q_path}")


if __name__ == "__main__":
    main()
