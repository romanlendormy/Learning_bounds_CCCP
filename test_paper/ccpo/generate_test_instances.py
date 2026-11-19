#!/usr/bin/env python3
"""
Génère les 120 instances CCPO de test de l'article (campagnes LD et LD0).

Grille : n ∈ {8, 10, 12, 15} × N ∈ {100, 500, 1000}, 10 instances faisables par cellule,
listées dans instances_list.txt (n N idx graine certificat). Chaque instance est produite par
collect/gen_bab_test_episode.py (même générateur que les données d'entraînement, graines
disjointes : graine = 7e9 + 100000 n + 1000 N + idx). Le choix des indices retenus est
documenté dans selection/.

La génération est déterministe ; les md5 attendus de instance.txt et scenarios.txt (les deux
fichiers lus par le solveur) sont dans instances_md5.txt
(vérification : cd <sortie> && md5sum -c <dépôt>/test_paper/ccpo/instances_md5.txt).

Usage (depuis la racine du dépôt) :
  python3 test_paper/ccpo/generate_test_instances.py [dossier_sortie]
  (défaut : test_paper/ccpo/instances/ ; instance i -> n<n>_N<N>/i<idx>/ep_000000)
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GEN = REPO / "collect" / "gen_bab_test_episode.py"


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "instances"
    n_written = 0
    for line in (HERE / "instances_list.txt").read_text().split("\n"):
        if not line.strip():
            continue
        n, N, idx, seed, _cert = line.split()
        dst = out_dir / f"n{n}_N{N}" / f"i{idx}"
        subprocess.run([sys.executable, str(GEN), n, N, idx, str(dst)], check=True, stdout=subprocess.DEVNULL)
        n_written += 1
    print(f"{n_written} instances écrites dans {out_dir}")


if __name__ == "__main__":
    main()
