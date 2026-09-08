# Instances de test — CC-MKP (rapport de recherche)

Générées le 2026-08-13 avec `generate_test_instances.py` (appelle
`collect/collect_large_base_samples.py::generate_song_tempfile`, même
distribution Song/Pisinger que la collecte d'entraînement).

## Paramètres

- n ∈ {15, 20, 25, 30}
- N (nb. scénarios) ∈ {100, 500, 1000}
- m = 5
- alpha = 0.5, R = 1000, correlation = weakly_correlated
- seeds = 43..52 (10 instances par config)
- epsilon = 0.1 — **non présent dans les fichiers** : c'est un paramètre de
  résolution (chance constraint), à passer au solveur au moment du benchmark,
  pas à la génération de l'instance.

12 configs × 10 seeds = 120 fichiers, nommés
`instance_n{n}_m5_N{N}_seed{seed}.txt`.

## ⚠️ Contamination train/test connue (décision utilisateur du 2026-08-13)

Les seeds 43..52 ne sont **pas** garanties hors-entraînement pour toutes les
configs. Vérifié contre `data/train_base_s10/seeds_done_n{n}_m5_N{N}.txt` :

| config      | seeds vues à l'entraînement | seeds 43-52 concernées |
|-------------|------------------------------|--------------------------|
| n15 (*)     | aucune donnée                | — (sûr) |
| n20, N100   | 0–5171                       | **43-52 vues** |
| n20, N500   | 0–1041                       | **43-52 vues** |
| n20, N1000  | seed 0 seulement             | — (sûr) |
| n25, N100   | 0–1333                       | **43-52 vues** |
| n25, N500/N1000 | aucune donnée            | — (sûr) |
| n30, N100   | 0–26                         | — (sûr, 43>26) |
| n30, N500/N1000 | aucune donnée            | — (sûr) |

Décision explicite : l'utilisateur a choisi de garder seed=43 (et suite
43..52) malgré ce risque pour n20-N100, n20-N500 et n25-N100.

Voir aussi `compare_holdout_multiinstance.sh` (racine du projet), qui
documente exactement le même piège pour seed=42 sur n∈{10,20} et l'a corrigé
en passant à seed≥50001 — précédent à considérer si ce risque devient
gênant pour l'interprétation des résultats du rapport.
