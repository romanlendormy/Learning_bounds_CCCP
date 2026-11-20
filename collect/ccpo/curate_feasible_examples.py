"""
curate_feasible_examples.py — Élimine du jeu d'entraînement les exemples
résiduels PROUVÉS INFAISABLES (indépendamment de λ).

Pourquoi une étape séparée plutôt qu'un skip silencieux à l'entraînement
(déjà en place dans train_ccpo_gnn.py) : la détection d'infaisabilité coûte
un evaluate_ccpo_lagrangian complet (zeta_solve sur tous les scénarios) --
un skip répété à CHAQUE epoch refait ce calcul en pure perte pour les mêmes
exemples, à chaque fois. Ici on le fait UNE SEULE FOIS, on écrit un fichier
propre, et l'entraînement n'a plus jamais besoin de tester la faisabilité.

Coût borné : on ne teste PAS tout le pool (des millions de lignes/cellule) --
on tire les candidats dans un ordre aléatoire et on s'arrête dès que la
cible `--target_per_cell` d'exemples FAISABLES est atteinte pour cette
cellule (ou que le pool est épuisé). Les racines (depth==0) sont testées
comme les autres (rigueur : ne pas supposer leur faisabilité même si elle
est quasi garantie par construction) mais suivent leur
propre quota (`--root_cap_per_cell`, plus petit, cf. root_frac de
train_ccpo_gnn.py).

Usage :
    python3 collect/ccpo/curate_feasible_examples.py --data_dir <dir> \
        --target_per_cell 3000 --root_cap_per_cell 300
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gnn" / "ccpo"))

from ccpo_dataset import load_instances_ccpo, load_nodes_ccpo, reconstruct_residual_instance
from ccpo_lagrangian_autograd import evaluate_ccpo_lagrangian_raw


def discover_cells(data_dir: Path) -> list[tuple[int, int]]:
    cells = []
    for f in sorted(data_dir.glob("instances_n*_N*.txt")):
        stem = f.stem.replace("instances_n", "")
        n, N = stem.split("_N")
        cells.append((int(n), int(N)))
    return cells


def instance_scales(inst) -> tuple[float, float]:
    c_scale = max(float(np.max(np.abs(inst.c_int))) if inst.c_int.size else 1.0, 1.0)
    eligible = np.where(~inst.infeasible)[0]
    if len(eligible) == 0:
        return c_scale, 1.0
    diags = [np.diagonal(inst.Sigma_dense[j]).mean() for j in eligible]
    return c_scale, max(float(np.mean(diags)), 1e-6)


def is_feasible(inst, node) -> bool:
    c_scale, sigma_scale = instance_scales(inst)
    ri = reconstruct_residual_instance(inst, node, c_scale, sigma_scale)
    if ri is None:
        return False  # feuille (n_free=0) -- rien a predire, traitee comme non retenue
    order = np.argsort(-ri.cost_free).astype(np.int32)
    infeasible_u8 = inst.infeasible.astype(np.uint8)
    lam0 = np.zeros((ri.N, ri.n_free), dtype=np.float64)
    _, _, feasible = evaluate_ccpo_lagrangian_raw(
        ri.n_free, ri.N, ri.k_res, ri.Q_res, ri.L, ri.U, inst.min_feas,
        ri.cost_free, ri.Sigma_free, ri.B_res, infeasible_u8, lam0, order, ri.R_ext,
    )
    return feasible


def _write_rec(f, rec) -> None:
    q_mask_str = ",".join(str(v) for v in rec.q_mask)
    f.write(f"{rec.instance_id},{rec.depth},{rec.curr_cost:.6f},{q_mask_str}\n")
    f.flush()


def _load_existing(out_path: Path) -> tuple[int, int, set]:
    """Relit ce qui est deja ecrit (reprise apres coupure d'un job) -- (n_racines,
    n_non_racines, cles_deja_vues) ou une cle = (instance_id, depth, curr_cost)
    -- suffisant en pratique pour eviter de re-tester/redoublonner un nœud
    deja retenu, meme si ce n'est pas une garantie d'unicite absolue."""
    if not out_path.exists():
        return 0, 0, set()
    n_roots = n_non_roots = 0
    seen = set()
    with open(out_path) as f:
        for line in f:
            parts = line.rstrip("\n").split(",")
            inst_id, depth, curr_cost = int(parts[0]), int(parts[1]), parts[2]
            seen.add((inst_id, depth, curr_cost))
            if depth == 0:
                n_roots += 1
            else:
                n_non_roots += 1
    return n_roots, n_non_roots, seen


def curate_cell(data_dir: Path, n: int, N: int, target_per_cell: int,
                 root_cap_per_cell: int, rng: random.Random) -> None:
    inst_path = data_dir / f"instances_n{n}_N{N}.txt"
    nodes_path = data_dir / f"nodes_n{n}_N{N}.txt"
    out_path = data_dir / f"nodes_n{n}_N{N}_feasible.txt"

    n_kept_roots, n_kept_non_root, seen = _load_existing(out_path)
    if n_kept_roots >= root_cap_per_cell and n_kept_non_root >= target_per_cell:
        print(f"n={n} N={N} : deja complet ({n_kept_roots} racines, {n_kept_non_root} "
              f"non-racines) -- rien a faire", flush=True)
        return
    if seen:
        print(f"n={n} N={N} : reprise -- {n_kept_roots} racines, {n_kept_non_root} "
              f"non-racines deja presentes", flush=True)

    instances = load_instances_ccpo(str(inst_path))
    nodes = load_nodes_ccpo(str(nodes_path))
    roots = [r for r in nodes if r.depth == 0 and (r.instance_id, 0, f"{r.curr_cost:.6f}") not in seen]
    non_roots = [r for r in nodes
                 if r.depth != 0 and (r.instance_id, r.depth, f"{r.curr_cost:.6f}") not in seen]
    rng.shuffle(roots)
    rng.shuffle(non_roots)

    # Ecriture INCREMENTALE (append, flush apres chaque ligne) : une coupure ne perd
    # pas la progression deja faite.
    out = open(out_path, "a")
    tested_roots = 0
    for rec in roots:
        if n_kept_roots >= root_cap_per_cell:
            break
        tested_roots += 1
        if is_feasible(instances[rec.instance_id], rec):
            _write_rec(out, rec)
            n_kept_roots += 1

    tested_non_root = 0
    for rec in non_roots:
        if n_kept_non_root >= target_per_cell:
            break
        tested_non_root += 1
        if is_feasible(instances[rec.instance_id], rec):
            _write_rec(out, rec)
            n_kept_non_root += 1
        if tested_non_root % 20 == 0:
            print(f"  [n={n} N={N}] {tested_non_root} testes (session), "
                  f"{n_kept_non_root}/{target_per_cell} faisables au total", flush=True)
    out.close()

    print(f"n={n} N={N} : racines {n_kept_roots}/{root_cap_per_cell}, "
          f"non-racines {n_kept_non_root}/{target_per_cell} "
          f"({tested_roots} racines + {tested_non_root} non-racines testees cette session) "
          f"-> {out_path.name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--target_per_cell", type=int, default=2000)
    ap.add_argument("--root_cap_per_cell", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only_cell", default=None,
                     help="Restreint a une seule cellule 'n,N' (ex. '8,20') -- "
                          "pour paralleliser en lancant un process par cellule.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    rng = random.Random(args.seed)
    cells = discover_cells(data_dir)
    if args.only_cell:
        n_f, N_f = (int(x) for x in args.only_cell.split(","))
        cells = [c for c in cells if c == (n_f, N_f)]
    for n, N in cells:
        curate_cell(data_dir, n, N, args.target_per_cell, args.root_cap_per_cell, rng)


if __name__ == "__main__":
    main()
