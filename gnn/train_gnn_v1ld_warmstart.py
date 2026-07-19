"""
train_gnn_v1ld_warmstart.py — Variante de train_gnn_v1ld.py où la loss est
calculee apres K iterations de sous-gradient a partir du lambda predit par
le GNN (warm start), au lieu du lambda one-shot brut.

Contexte (voir NOTES_N_scaling_investigation.md et memoire projet) : la
contrainte de production "one-shot" a ete explicitement levee pour cette
experience — le B&B appellera reellement K=10-15 iterations de sous-gradient
apres la prediction du GNN a chaque noeud. Le gain de vitesse vs la baseline
classique reste a rebenchmarker separement (reporte a plus tard) ; ce script
traite d'abord la question de qualite : un entrainement "K-step-aware"
produit-il un meilleur point apres K iterations qu'un entrainement one-shot
classique suivi des memes K iterations en post-hoc ?

Mecanisme de gradient (straight-through) :
  1. lam0 = model(data)                                  (differentiable)
  2. lam_K_np = verify_v1ld(data, T=K, lam_init=lam0, ...)[best_lam]
     — K iterations de sous-gradient en NumPy, IDENTIQUES a ce qui tournerait
       en B&B reel (memes appels au solveur knapsack C++), donc non
       differentiable (arg-selections combinatoires a chaque pas).
  3. lam_K_ste = lam0 + (lam_K_np - lam0).detach()
     — la valeur forward est lam_K (les K iterations ont bien eu lieu), mais
       le gradient remonte a lam0 par un raccourci identite : on ne deplie
       PAS le graphe de calcul sur les K pas (couteux, et de toute facon
       chaque pas contient un arg-select non-differentiable). C'est une
       approximation : le gradient reel de la trajectoire (Jacobien des K
       mises a jour) est ignore, seul le gradient de la loss finale (via le
       theoreme de l'enveloppe deja utilise dans lagrangian_v1ld) est reutilise
       tel quel. Cout ~= une passe forward GNN + K iterations NumPy (rapide),
       pas de surcout memoire d'un unroll Torch.
  4. loss = lagrangian_v1ld(lam_K_ste, data)              (differentiable)

Calibrage du pas de sous-gradient pour K petit : voir
gnn/diag_calibrate_warmstart_step.py. Constat empirique (15 instances,
N=N_z=500, depth=0, dump gnn/runs/ablation_hidden/ckpt_N500_h64) : le pas
historique 1/sqrt(t+1) (calibre pour un cold-start lambda=0) est trop
agressif pour un warm-start deja informatif — sur K=10-15 iterations, il ne
bat le lambda brut du GNN que sur 5-10/15 instances. Un pas fixe petit
(~0.05) ou un sqrt amorti (offset~50) bat le lambda brut sur 15/15 et
approche cold_T50 sur plusieurs instances. Valeurs par defaut choisies en
consequence ci-dessous (STEP_MODE/STEP_VALUE), mais ce n'est qu'un point de
depart raisonnable, pas un optimum global : le GNN va de toute facon
co-adapter lambda_0 au pas choisi pendant l'entrainement, donc l'enjeu de ce
calibrage etait surtout d'eviter un choix manifestement mauvais (comme le
montre sqrt_t0=1 ci-dessus), pas de trouver LE meilleur pas.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_gnn_v1ld import (
    GNN,
    load_dataset,
    lagrangian_v1ld,
    verify_v1ld,
    pyg_data,
)

# ── Hyperparametres K-step (voir diag_calibrate_warmstart_step.py) ───────────
K_DEFAULT          = 15
STEP_MODE_DEFAULT  = "fixed"
STEP_VALUE_DEFAULT = 0.05
STEP_OFFSET_DEFAULT = 1.0  # utilise seulement si step_mode="sqrt"


def lambda_k_step_straight_through(
    lam0: torch.Tensor,
    data: pyg_data.Data,
    K: int = K_DEFAULT,
    step_mode: str = STEP_MODE_DEFAULT,
    step_value: float | None = STEP_VALUE_DEFAULT,
    step_offset: float = STEP_OFFSET_DEFAULT,
) -> torch.Tensor:
    """
    Execute K iterations de sous-gradient (NumPy, non differentiable) a partir
    de lam0, et retourne un tenseur dont la VALEUR est lambda apres K
    iterations (le meilleur lambda vu sur la fenetre, meme convention que le
    B&B reel — cf. gecode_bab.hpp: res.lambda_star = best_lam) mais dont le
    GRADIENT est l'identite vers lam0 (straight-through estimator).

    lam0 : (N_z*n_g,) tenseur torch, sortie du GNN (requires_grad=True).
    """
    lam0_np = lam0.detach().cpu().numpy().reshape(data.N_z, data.n_g).astype(np.float64)
    _, lam_K_np = verify_v1ld(
        data, T=K, lam_init=lam0_np,
        step_mode=step_mode, step_value=step_value, step_offset=step_offset,
        return_lam=True,
    )
    lam_K_t = torch.from_numpy(lam_K_np).view(-1).to(dtype=lam0.dtype, device=lam0.device)
    return lam0 + (lam_K_t - lam0).detach()


# ─────────────────────────────────────────────────────────────────────────────
# Overfit sur un graphe — sanity check avant tout run a l'echelle
# ─────────────────────────────────────────────────────────────────────────────

def overfit_one_graph_warmstart(
    data: pyg_data.Data,
    n_epochs: int = 1000,
    lr: float = 1e-3,
    K: int = K_DEFAULT,
    step_mode: str = STEP_MODE_DEFAULT,
    step_value: float | None = STEP_VALUE_DEFAULT,
    step_offset: float = STEP_OFFSET_DEFAULT,
    seed: int = 0,
):
    """
    Meme protocole que overfit_one_graph (train_gnn_v1ld.py) mais la loss est
    calculee sur lambda_K (apres K iterations de sous-gradient) au lieu de
    lambda_0. Suit aussi L1(lambda_0) en monitoring (jamais optimise
    directement) pour voir si la qualite one-shot derive en sous-produit.
    """
    torch.manual_seed(seed)
    device = torch.device("cpu")
    data   = data.to(device)

    model     = GNN(m=data.m).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)

    opt_val = data.opt.item()
    with torch.no_grad():
        lam0_init = torch.zeros(data.N_z * data.n_g)
        l1_init   = lagrangian_v1ld(lam0_init, data).item()
        cold_K    = verify_v1ld(data, T=K, step_mode=step_mode,
                                 step_value=step_value, step_offset=step_offset)

    print(f"\n{'='*76}")
    print(f"Overfit warm-start K={K} (step_mode={step_mode}, step_value={step_value}) "
          f"— n_g={data.n_g}  N_z={data.N_z}  N={data.N}  m={data.m}")
    print(f"  opt={opt_val:.2f}   L1(λ=0)={l1_init:.2f}   cold_K={cold_K:.2f} (reference, sans GNN)")
    print(f"{'='*76}")
    print(f"{'epoch':>6}  {'L1(λ_0)':>10}  {'L1(λ_K)':>10}  {'gap_K_vs_opt%':>13}  {'gap_K_vs_cold%':>14}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        lam0 = model(data)
        lam_K_ste = lambda_k_step_straight_through(
            lam0, data, K=K, step_mode=step_mode,
            step_value=step_value, step_offset=step_offset)
        loss = lagrangian_v1ld(lam_K_ste, data)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 50 == 0 or epoch == 1:
            with torch.no_grad():
                l1_lam0 = lagrangian_v1ld(lam0, data).item()
            b     = loss.item()
            g_opt  = (b - opt_val) / opt_val * 100 if opt_val > 0 else float("nan")
            g_cold = (b - cold_K) / abs(cold_K) * 100 if cold_K != 0 else float("nan")
            print(f"{epoch:6d}  {l1_lam0:10.4f}  {b:10.4f}  {g_opt:+12.2f}%  {g_cold:+13.2f}%")

    print(f"{'='*76}")
    print(f"Borne finale L1(λ_K) : {loss.item():.4f}  (opt={opt_val:.2f}, cold_K={cold_K:.2f})")
    with torch.no_grad():
        l1_lam0_final = lagrangian_v1ld(model(data), data).item()
    print(f"Borne one-shot finale L1(λ_0) (jamais optimisee directement) : {l1_lam0_final:.4f}")


if __name__ == "__main__":
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/train_v1LD")
    n_val    = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    K_arg    = int(sys.argv[3]) if len(sys.argv) > 3 else K_DEFAULT

    inst_file  = data_dir / f"instances_n{n_val}.txt"
    nodes_file = data_dir / f"nodes_n{n_val}.txt"
    if not inst_file.exists():
        inst_file  = data_dir / f"instances_n{n_val}_gbab.txt"
        nodes_file = data_dir / f"nodes_n{n_val}_gbab.txt"

    graphs = load_dataset(inst_file, nodes_file, max_graphs=5)
    if not graphs:
        print("Aucun graphe charge — verifier data_dir/n_val.")
        sys.exit(1)

    overfit_one_graph_warmstart(graphs[0], K=K_arg)
