"""
diag_lambda_depth_mismatch.py — Test de l'hypothèse du centrage invalide
quand N_z<N dans le Lagrangien V1CLD (SSP).

Rappel (cf. diag_lambda_scale_T.py) : le facteur T manquant dans la tête du
GNN amplifie le gradient initial mais NE reproduit PAS, seul, l'explosion
catastrophique observée dans train_scaled_v1cld (job 16058124) — sur un
entraînement racines-seules (N_z=N), gnorm/|λ|_max restent bornés sur 30
epochs, avec ou sans le correctif T.

Nouvelle hypothèse : `_apply_center_ssp` retire la moyenne jointe sur (i,j)
— direction de jauge exacte uniquement quand N_z=N (aucun scénario z fixé,
cf. Λ_sum et cost_full dans lagrangian_v1cld_ssp : un décalage uniforme de λ
change le coût de δ·T·(N−N_z), nul seulement si N_z=N).

CORRECTIF DE PROTOCOLE (avant le run réel) : un premier sondage du fichier
E4_T12_N20 montre que "profondeur du B&B" ≠ "N_z<N" — la quasi-totalité de
la profondeur vient du branchement sur x (E*T paires), PAS sur z : même à
depth=10, 88% des nœuds ont encore N_z=N=20. Échantillonner par profondeur
(comme prévu initialement) aurait donc noyé le signal recherché. On
sélectionne donc directement sur N_z (indépendamment de la profondeur) :
  - root         : N_z = N        (jauge exacte, ~91% du fichier)
  - light_branch : N_z = N−1      (déviation minimale, majorité du reste)
  - strong_branch: N_z ≤ N/2      (déviation forte, ~5000 lignes dispo)

Test : un seul combo (E4_T12_N20), fix_T=False (comportement actuel,
inchangé), center_lam=True (comme en production) — même nombre de graphes
et d'epochs dans chaque condition.

Usage : python diag_lambda_depth_mismatch.py --n_graphs 150 --n_epochs 50
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_v1CLD_ssp import (
    GNN, lagrangian_v1cld_ssp, load_raw_nodes, LazyGraphDataset,
    EagerGraphDataset, _apply_center_ssp,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "train_ssp_base_s10"


CONDITIONS = {
    "root":          lambda nz, N: nz == N,
    "light_branch":  lambda nz, N: nz == N - 1,
    "strong_branch": lambda nz, N: nz <= N // 2,
}


def load_sample(combo: str, n_graphs: int, seed: int, condition: str, instances=None, records=None):
    if records is None:
        instances, records = load_raw_nodes(
            DATA_DIR / f"instances_{combo}.txt",
            DATA_DIR / f"nodes_{combo}.txt",
            max_records=None,
        )
    pred = CONDITIONS[condition]
    pool = [r for r in records
            if pred(int((r["z_mask"] == 2).sum()), instances[r["inst_id"]]["N"])]

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), min(n_graphs, len(pool)), replace=False)
    sample = [pool[i] for i in idx]
    lazy = LazyGraphDataset(instances, sample)
    eager = EagerGraphDataset(lazy)
    return instances, eager, records


def run(eager, instances, n_epochs: int, lr: float, max_norm: float, seed: int, tag: str):
    torch.manual_seed(seed)
    model = GNN(hidden=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"gnorm": [], "lam_max": [], "lam_mean": [], "loss": []}
    for ep in range(1, n_epochs + 1):
        ep_gnorm = 0.0
        ep_lam_max = 0.0
        ep_lam_mean_sum = 0.0
        ep_loss_sum = 0.0
        for i in range(len(eager)):
            g = eager[i]
            inst = instances[int(g.inst_id.item())]

            optimizer.zero_grad()
            lam = model(g)
            lam = _apply_center_ssp(lam, int(g.N_z), int(g.T), int(g.n_g))

            c_scale = max(float(g.c_scale), 1.0)
            loss = lagrangian_v1cld_ssp(lam, g, inst) / c_scale
            loss.backward()

            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            ep_gnorm = max(ep_gnorm, gn.item())
            lam_d = lam.detach().abs()
            ep_lam_max = max(ep_lam_max, lam_d.max().item())
            ep_lam_mean_sum += lam_d.mean().item()
            ep_loss_sum += loss.item()

            optimizer.step()

        history["gnorm"].append(ep_gnorm)
        history["lam_max"].append(ep_lam_max)
        history["lam_mean"].append(ep_lam_mean_sum / len(eager))
        history["loss"].append(ep_loss_sum / len(eager))
        print(f"  [{tag}] ep {ep:3d}/{n_epochs}  "
              f"loss={history['loss'][-1]:+.4f}  gnorm={ep_gnorm:12.3f}  "
              f"lam_mean={history['lam_mean'][-1]:.4f}  lam_max={ep_lam_max:10.3f}",
              flush=True)
    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--combo", default="E4_T12_N20")
    p.add_argument("--n_graphs", type=int, default=150)
    p.add_argument("--n_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_norm", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    results = {}
    instances = records = None
    for condition in ("root", "light_branch", "strong_branch"):
        print(f"\n=== Chargement {args.combo} (condition={condition}) ===", flush=True)
        t0 = time.time()
        instances, eager, records = load_sample(
            args.combo, args.n_graphs, args.seed, condition, instances, records)
        depths = eager._depths
        n_z_vals = [int(eager[i].N_z) for i in range(len(eager))]
        print(f"  {len(eager)} graphes chargés en {time.time()-t0:.1f}s  "
              f"depth: min={min(depths)} max={max(depths)} mean={np.mean(depths):.1f}  "
              f"N_z: min={min(n_z_vals)} max={max(n_z_vals)} mean={np.mean(n_z_vals):.1f}",
              flush=True)

        tag = f"{args.combo} condition={condition}"
        print(f"\n--- {tag} ---", flush=True)
        t0 = time.time()
        h = run(eager, instances, args.n_epochs, args.lr, args.max_norm, args.seed, tag)
        print(f"  fait en {time.time()-t0:.1f}s", flush=True)
        results[condition] = h

    print("\n" + "=" * 70)
    print("RÉSUMÉ — gnorm et |λ| par epoch")
    print("=" * 70)
    for condition, h in results.items():
        print(f"\n{args.combo}  condition={condition}")
        for ep, (gn, lm, la) in enumerate(zip(h["gnorm"], h["lam_max"], h["lam_mean"]), start=1):
            print(f"  ep{ep:3d}  gnorm={gn:12.3f}  lam_mean={la:.4f}  lam_max={lm:10.3f}")


if __name__ == "__main__":
    main()
