"""
diag_lambda_scale_T.py — Test de l'hypothèse d'amplification par T dans le
Lagrangien V1CLD (SSP).

Constat : lagrangian_v1cld_ssp divise λ par p_ij = 1/(N*T) (train_v1CLD_ssp.py
L742), donc multiplie son effet sur le coût par N*T. Mais la tête du GNN ne
normalise que par N : lam = h_out * c_scale / N (train_v1CLD_ssp.py L688,
identique à la formule MKP gnn/train_gnn_v1ld.py L711, où il n'y a pas de T).
Hypothèse : cet écart (facteur T manquant) amplifie le gradient et cause
l'explosion de gnorm/|λ| observée dans train_scaled_v1cld (job 16058124),
qui persiste malgré le centrage actif.

Test : sur des graphes RACINE (depth=0, donc N_z=N — pas de confondant de
branchement, cf. discussion sur la validité du centrage hors racine) de deux
combos ne différant que par T (E4_T8_N20 vs E4_T12_N20, même E=4, même N=20),
compare la croissance de gnorm/|λ|_max en entraînement, avec (fix_T=True,
lam /= T en plus) et sans (fix_T=False, comportement actuel) la correction.

Usage : python diag_lambda_scale_T.py --n_graphs 100 --n_epochs 25
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


def load_roots(combo: str, n_graphs: int, seed: int):
    instances, records = load_raw_nodes(
        DATA_DIR / f"instances_{combo}.txt",
        DATA_DIR / f"nodes_{combo}.txt",
        max_records=200_000,
    )
    roots = [r for r in records if r["depth"] == 0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(roots), min(n_graphs, len(roots)), replace=False)
    sample = [roots[i] for i in idx]
    lazy = LazyGraphDataset(instances, sample)
    eager = EagerGraphDataset(lazy)
    return instances, eager


def run(eager, instances, n_epochs: int, lr: float, max_norm: float,
        fix_T: bool, seed: int, tag: str):
    torch.manual_seed(seed)
    model = GNN(hidden=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"gnorm": [], "lam_max": [], "loss": []}
    for ep in range(1, n_epochs + 1):
        ep_gnorm = 0.0
        ep_lam_max = 0.0
        ep_loss_sum = 0.0
        for i in range(len(eager)):
            g = eager[i]
            inst = instances[int(g.inst_id.item())]

            optimizer.zero_grad()
            lam = model(g)
            if fix_T:
                lam = lam / float(g.T)
            lam = _apply_center_ssp(lam, int(g.N_z), int(g.T), int(g.n_g))

            c_scale = max(float(g.c_scale), 1.0)
            loss = lagrangian_v1cld_ssp(lam, g, inst) / c_scale
            loss.backward()

            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            ep_gnorm = max(ep_gnorm, gn.item())
            ep_lam_max = max(ep_lam_max, lam.detach().abs().max().item())
            ep_loss_sum += loss.item()

            optimizer.step()

        history["gnorm"].append(ep_gnorm)
        history["lam_max"].append(ep_lam_max)
        history["loss"].append(ep_loss_sum / len(eager))
        print(f"  [{tag}] ep {ep:3d}/{n_epochs}  "
              f"loss={history['loss'][-1]:+.4f}  gnorm={ep_gnorm:10.3f}  "
              f"lam_max={ep_lam_max:10.3f}", flush=True)
    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--combos", nargs="+", default=["E4_T8_N20", "E4_T12_N20"])
    p.add_argument("--n_graphs", type=int, default=100)
    p.add_argument("--n_epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_norm", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    results = {}
    for combo in args.combos:
        print(f"\n=== Chargement {combo} (racines seulement) ===", flush=True)
        t0 = time.time()
        instances, eager = load_roots(combo, args.n_graphs, args.seed)
        print(f"  {len(eager)} graphes racine chargés en {time.time()-t0:.1f}s", flush=True)

        for fix_T in (False, True):
            tag = f"{combo} fix_T={fix_T}"
            print(f"\n--- {tag} ---", flush=True)
            t0 = time.time()
            h = run(eager, instances, args.n_epochs, args.lr, args.max_norm,
                    fix_T, args.seed, tag)
            print(f"  fait en {time.time()-t0:.1f}s", flush=True)
            results[(combo, fix_T)] = h

    print("\n" + "=" * 70)
    print("RÉSUMÉ — gnorm et |λ|_max, dernier epoch vs premier epoch")
    print("=" * 70)
    for (combo, fix_T), h in results.items():
        print(f"{combo:15s} fix_T={fix_T!s:5s}  "
              f"gnorm[0]={h['gnorm'][0]:9.3f} -> gnorm[-1]={h['gnorm'][-1]:12.3f}   "
              f"lam_max[0]={h['lam_max'][0]:7.3f} -> lam_max[-1]={h['lam_max'][-1]:10.3f}")


if __name__ == "__main__":
    main()
