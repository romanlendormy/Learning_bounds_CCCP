"""
subsample_nodes.py — Sous-échantillonnage de nodes_n{n}_m{m}.txt.

Deux modes :
  --rate   R  : conserve R% des nœuds non-racines par instance (comportement original)
  --target T  : conserve T nœuds au total, répartis uniformément entre les valeurs
                de N présentes dans le fichier. Les racines (depth=0) sont toujours
                conservées ; les non-racines sont samplées pour atteindre T/nb_N.

Les racines sont TOUJOURS conservées dans les deux modes.
Le fichier instances est laissé inchangé (les inst_ids orphelins sont ignorés à l'entraînement).

Usage — mode cible :
  python collect/subsample_nodes.py \\
      --nodes     data/train_base_s30/nodes_n15_m5.txt \\
      --n         15 \\
      --target    30000 \\
      --inplace

Usage — mode taux :
  python collect/subsample_nodes.py \\
      --nodes     data/train_base_s30/nodes_n15_m5.txt \\
      --rate      0.3 \\
      --inplace
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nodes",    required=True,  help="nodes_n{n}_m{m}.txt à sous-échantillonner")
    p.add_argument("--n",        type=int, default=None,
                   help="Taille n du problème (nécessaire pour --target, pour déduire N depuis les lignes)")
    p.add_argument("--target",   type=int, default=None,
                   help="Nombre total de nœuds à conserver (réparti uniformément par N)")
    p.add_argument("--rate",     type=float, default=None,
                   help="Fraction de nœuds non-racines à conserver (par instance)")
    p.add_argument("--inplace",  action="store_true",
                   help="Écrase le fichier source (sauvegarde .bak créée automatiquement)")
    p.add_argument("--out",      default=None,
                   help="Fichier de sortie (si --inplace non spécifié)")
    p.add_argument("--seed",     type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    if args.target is None and args.rate is None:
        raise ValueError("Spécifie --target ou --rate.")
    if args.target is not None and args.rate is not None:
        raise ValueError("--target et --rate sont mutuellement exclusifs.")
    if args.target is not None and args.n is None:
        raise ValueError("--n est requis avec --target (pour déduire N depuis les lignes).")

    rng = random.Random(args.seed)
    nodes_src = Path(args.nodes)

    # ── Lecture et groupement ─────────────────────────────────────────────────
    print(f"Lecture de {nodes_src} …", flush=True)
    # roots[N][inst_id]  = [lignes]
    # others[N][inst_id] = [lignes]
    roots:  dict[int, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    others: dict[int, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    total_lines = 0

    with open(nodes_src) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts   = line.split(",")
            inst_id = int(parts[0])
            depth   = int(parts[1])
            if args.target is not None:
                N = len(parts) - 3 - args.n
            else:
                N = 0   # pas utilisé en mode --rate
            if depth == 0:
                roots[N][inst_id].append(line)
            else:
                others[N][inst_id].append(line)
            total_lines += 1

    n_values = sorted(set(roots) | set(others))
    print(f"  {total_lines} nœuds lus")
    for N in n_values:
        n_roots  = sum(len(v) for v in roots[N].values())
        n_others = sum(len(v) for v in others[N].values())
        print(f"  N={N:4d} : {len(roots[N])} instances  {n_roots} racines  {n_others} non-racines")

    # ── Calcul des budgets ────────────────────────────────────────────────────
    if args.target is not None:
        n_groups     = len(n_values)
        per_N_budget = args.target // n_groups
        print(f"\nCible : {args.target} nœuds  →  {per_N_budget} par N ({n_groups} valeurs de N)")

    # ── Sélection des lignes à conserver ─────────────────────────────────────
    kept_lines: list[str] = []

    for N in n_values:
        n_root_total = sum(len(v) for v in roots[N].values())

        if args.target is not None:
            budget_non_root = max(0, per_N_budget - n_root_total)
            # Collecter tous les non-racines pour ce N, puis sampler globalement
            all_non_roots = [l for lines in others[N].values() for l in lines]
            k = min(budget_non_root, len(all_non_roots))
            sampled_non_roots = rng.sample(all_non_roots, k) if k > 0 else []

            kept_N  = [l for lines in roots[N].values() for l in lines]
            kept_N += sampled_non_roots
            kept_lines.extend(kept_N)

            print(f"  N={N} : {n_root_total} racines + {k} non-racines = {len(kept_N)} conservés")

        else:   # mode --rate
            for inst_id in sorted(set(roots[N]) | set(others[N])):
                kept_lines.extend(roots[N].get(inst_id, []))
                non_root = others[N].get(inst_id, [])
                if non_root:
                    k = max(0, round(len(non_root) * args.rate))
                    if k > 0:
                        kept_lines.extend(rng.sample(non_root, k))

    # ── Écriture ──────────────────────────────────────────────────────────────
    if args.inplace:
        bak = nodes_src.with_suffix(".txt.bak")
        shutil.copy(nodes_src, bak)
        print(f"\nSauvegarde originale → {bak}")
        out_path = nodes_src
    elif args.out:
        out_path = Path(args.out)
    else:
        raise ValueError("Spécifie --inplace ou --out.")

    with open(out_path, "w") as f:
        for line in kept_lines:
            f.write(line + "\n")

    print(f"→ {len(kept_lines)} nœuds écrits dans {out_path}")
    print(f"  (réduction {total_lines} → {len(kept_lines)}, "
          f"{len(kept_lines)/total_lines*100:.1f} % conservés)")


if __name__ == "__main__":
    main()
