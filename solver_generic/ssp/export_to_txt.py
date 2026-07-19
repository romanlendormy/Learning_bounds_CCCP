"""
export_to_txt.py — Exporte un épisode ssp_stochastic (HDF5) vers le format
texte 4-fichiers attendu par ssp_loader.cpp (cf. ssp_loader.hpp).

Référencé depuis ssp_loader.hpp/README.md mais jamais écrit — reconstruit
lors de la résurrection de ssp_stochastic (module supprimé au commit
e2f7b37, remplacé par rien de fonctionnel côté génération d'instances).

Usage :
    python export_to_txt.py dataset.h5 --ep 0 --out solver_bab/ssp/instances/ep0
    python export_to_txt.py dataset.h5 --ep 0 --ep 3 --out-dir instances/  # plusieurs épisodes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ssp_stochastic.writers import open_dataset, read_episode, SYM_TO_INT


def write_instance(ep: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dfa = ep["dfa"]
    E, T, N = ep["E"], ep["T"], ep["N"]
    demand_TN = ep["demand"]          # (T, N) — convention ssp_stochastic
    salary = ep["salary"]

    (out_dir / "instance.txt").write_text(
        f"E {E}\nT {T}\nN {N}\nEPSILON {ep['epsilon']}\nINTERCHANGEABLE 1\n"
    )

    accepting_sorted = sorted(dfa.accepting)
    trans_sorted = sorted(dfa.delta.items())  # ((src, sym), dst), ordre déterministe
    lines = [
        f"NUM_STATES {dfa.num_states}",
        f"INITIAL {dfa.initial}",
        f"N_ACCEPTING {len(accepting_sorted)}",
        " ".join(str(q) for q in accepting_sorted),
        f"N_TRANSITIONS {len(trans_sorted)}",
    ]
    for (src, sym), dst in trans_sorted:
        lines.append(f"{src} {SYM_TO_INT[sym]} {dst}")
    (out_dir / "automaton.txt").write_text("\n".join(lines) + "\n")

    (out_dir / "cost.txt").write_text(
        f"T {T}\nSALARY " + " ".join(f"{s:.10g}" for s in salary) + "\n"
    )

    # demand.txt : N lignes de T entiers, demand[scénario][période]
    # -> transposer depuis la convention (T, N) de ssp_stochastic
    demand_NT = demand_TN.T
    demand_lines = [f"T {T}", f"N {N}"]
    for i in range(N):
        demand_lines.append(" ".join(str(int(x)) for x in demand_NT[i]))
    (out_dir / "demand.txt").write_text("\n".join(demand_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5_path", help="Fichier dataset.h5 (ssp_stochastic)")
    parser.add_argument("--ep", type=int, action="append", required=True,
                        help="ID(s) d'épisode à exporter (répétable)")
    parser.add_argument("--out-dir", default=".",
                        help="Répertoire parent — un sous-dossier ep_XXXXX/ par épisode")
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    with open_dataset(args.h5_path, mode="r") as h5f:
        for ep_id in args.ep:
            ep = read_episode(h5f, ep_id)
            out_dir = out_root / f"ep_{ep_id:05d}"
            write_instance(ep, out_dir)
            print(f"ep_{ep_id:05d} -> {out_dir}  "
                  f"(E={ep['E']} T={ep['T']} N={ep['N']} eps={ep['epsilon']} "
                  f"n_states={ep['dfa'].num_states})")


if __name__ == "__main__":
    main()
