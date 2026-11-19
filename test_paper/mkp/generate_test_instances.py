#!/usr/bin/env python3
"""
Génère les 120 instances CC-MKP de test de l'article (communes aux campagnes SLD, DLD,
λ = 0, sensibilité et GNN vs sous-gradient).

Grille : n ∈ {15, 20, 25, 30} × N ∈ {100, 500, 1000}, m = 5, graines 43..52
(10 instances par cellule). Même générateur que la collecte d'entraînement
(collect/collect_large_base_samples.py::generate_song_tempfile : instance Pisinger
weakly_correlated, R = 1000, alpha = 0.5, scénarios de Song et al.).
ε = 0,1 n'est pas dans les fichiers : c'est un paramètre du solveur.

La génération est déterministe ; les md5 attendus sont dans instances_md5.txt
(vérification : cd <sortie> && md5sum -c <dépôt>/test_paper/mkp/instances_md5.txt).

Usage (depuis la racine du dépôt) :
  python3 test_paper/mkp/generate_test_instances.py [dossier_sortie]
  (défaut : test_paper/mkp/instances/)
"""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "collect"))
from collect_large_base_samples import generate_song_tempfile  # noqa: E402

N_VALUES = [15, 20, 25, 30]
N_SCEN_VALUES = [100, 500, 1000]
M = 5
SEEDS = list(range(43, 53))
ALPHA = 0.5
R = 1000
CORRELATION = "weakly_correlated"


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "instances"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for n in N_VALUES:
        for N_scen in N_SCEN_VALUES:
            for seed in SEEDS:
                tmp_path = generate_song_tempfile(
                    n, M, N_scen, seed, alpha=ALPHA, R=R, correlation=CORRELATION,
                )
                shutil.move(tmp_path, out_dir / f"instance_n{n}_m{M}_N{N_scen}_seed{seed}.txt")
                n_written += 1
    print(f"{n_written} instances écrites dans {out_dir}")


if __name__ == "__main__":
    main()
