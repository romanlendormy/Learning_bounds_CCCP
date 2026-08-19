# Note — Exploitation des z résolus pendant le x-branching (V1LD gnn_batched)

**Date** : 2026-08-19
**Statut** : expérience terminée, **non déployée** — résultat négatif/neutre documenté ci-dessous.
**Réversibilité** : tous les fichiers créés sont additifs, séparés du code de production. Aucune modification de `gecode_bab.hpp/.cpp`, `gecode_bab_gnn_batched.hpp/.cpp`, `gnn_sd_edit_native_batched.hpp/.cpp`, `gnn_sd_edit_scriptable_batched.py`. Revenir en arrière = ne pas utiliser les binaires/fichiers listés en fin de note.

## Question initiale

`extract_pending()` (`gecode_bab_gnn_batched.hpp`) n'extrait que `x_mask`/`fp` — jamais l'état de `z`. Le GNN calcule donc toujours sur les N scénarios complets, même quand certains `z_i` sont déjà résolus par la propagation CP pendant le x-branching (avant que le branchement natif sur z ne commence). Question : peut-on exploiter cette info pour resserrer la borne et/ou accélérer le B&B ?

## Vérifications théoriques (avant tout code)

- **Fréquence mesurée** (seed=43, n=20/N=500, instrumentation ad hoc dans le driver) : 33.4% des nœuds de x-branching ont ≥1 z déjà fixé ; en moyenne 5.05 z fixés/nœud sur l'ensemble de la population (donc ~15/500 quand ça se déclenche) ; max observé 49 sur un seul nœud ; **z=1 jamais résolu par propagation seule** (0.000 exactement — démontrable : `z_i=1 → contrainte` (RM_IMP) ne peut être déduit que par branchement explicite, jamais par propagation dans le sens infaisabilité).
- **Validité** : exclure un scénario à z fixé (0 ou 1) de la somme lagrangienne est une **exclusion exacte**, pas une approximation — sa contribution à l'objectif (qui ne dépend que de x) est nulle une fois z fixé, et sa contrainte de capacité est déjà gérée nativement par la propagation CP de Gecode (dure si z=1, désactivée si z=0). La formule de référence (`lagrangian_v1ld`/`lagrangian_v1cld`, `train_v1LD_sd_edit.py`/`train_v1CLD_sd_edit.py`) le fait déjà via `N_z`/`n_fixed_z1` — ce n'est pas une nouvelle dérivation.
- **Modèle GNN déjà entraîné** : le `forward()` scripté déduit `n_z` dynamiquement de la taille du tenseur reçu (`n_z = v_total // B // n_g`) — **aucun réentraînement nécessaire** pour accepter un N_z différent de N. Seule contrainte : dans un appel batché, tous les membres du batch doivent partager le MÊME N_z (reshape dense `.view(B, n_z, ...)`).

## Tentative 1 — bucketing par `(n_g, N_z)` : ÉCHEC (régression nette)

Pour respecter la contrainte "même N_z par batch", le premier essai a bucketé les nœuds en attente par `(n_g, N_z)` au lieu de `n_g` seul.

**Résultat (seed=43, n=20/N=500, K=64, cuda)** :

| | Origine | (n_g,N_z) |
|---|---|---|
| Nœuds | 12 975 | 15 839 (+22%) |
| Flushes / remplissage moyen | 375 / 56.31 | 1 086 / **22.66** |
| Temps total | 66.8 s | 74.4 s (+11%) |

**Cause** : `N_z` varie quasi continûment (dépend du chemin exact, cf. plus bas) — la clé `(n_g,N_z)` fragmente en une multitude de petits paniers qui ne se remplissent jamais jusqu'à K=64, forçant des flushs prématurés. **Abandonné.**

## Tentative 2 — batch RAGGED (pooling scatter, sans padding) : quasi neutre

Le message passing (`GatedGraphConvScriptable`, `index_add_` générique) traite déjà un batch bloc-diagonal comme des graphes disconnectés, quelle que soit leur taille — seul le **pooling** (`.view(B,n_z,n_g,dim).mean(...)`) exigeait un N_z uniforme. Solution : un second modèle exporté (mêmes poids, aucun réentraînement), `GNNSdEditScriptableBatchedRagged` (`gnn_sd_edit/gnn_sd_edit_scriptable_batched_ragged.py`), avec pooling par `scatter`/`index_add_` paramétré par des index par nœud (`graph_id`, `item_group`, `scen_group`) — accepte nativement des N_z hétérogènes dans un même batch, sans padding. Validé numériquement contre le modèle dense : max|diff|≈3e-8 (cas uniforme ET hétérogène, `test_paper/validate_gnn_sd_edit_ragged.py`).

Driver revenu au bucketing par `ng_depth` seul (plus de fragmentation).

**Résultats (même job, même charge GPU — comparaison propre)** :

| | seed=43, n=20/N=500 | seed=43, n=20/N=1000 |
|---|---|---|
| Nœuds | 12 975 → 12 912 (−0.5%) | 12 352 → 12 272 (−0.6%) |
| Forward GNN | 14.00s → 13.93s (≈0) | 25.24s → 25.95s (≈0) |
| Calcul ζ | 18.59s → 18.29s (≈0) | 35.73s → 35.18s (≈0) |
| Construction graphe | 11.69s → **15.03s** (+29%) | 23.53s → **29.41s** (+25%) |
| **Temps total** | 65.3s → 67.6s (**+3.5%**) | 142.6s → 147.2s (**+3.2%**) |

Correction validée : même optimum exact dans tous les cas (sanity check vs mode `v1ld` exact).

### Cause du ralentissement (mesurée, pas supposée)

```
                          hits    misses   hit rate
Origine   (n20/N500)   :  20361      33      99.8%
Z-aware   (n20/N500)   :  18104    2175      89.3%
Origine   (n20/N1000)  :  21084      33      99.8%
Z-aware   (n20/N1000)  :  19200    1880      91.1%
```

Le cache `ActiveXInvariant` (dantzig_rank + A_mean_scen/A_std_scen — tris et agrégations, pas de simples lectures) est indexé par `active_x` côté origine. Le branchement x suit un **ordre fixe, indépendant des valeurs assignées** (`branch_order`) — l'ensemble des items encore libres à une profondeur donnée est donc presque toujours le même quelles que soient les valeurs choisies en amont, d'où 99.8% de hits (33 calculs coûteux amortis sur ~20 000 appels).

Côté z-aware, la clé devient `active_x ⊕ free_z`. Quels scénarios deviennent infaisables dépend entièrement des poids `A[i][j][k]` des items *précis* fixés à 1 — pas seulement de leur nombre — donc `free_z` ne recur presque jamais entre branches différentes. Le nombre de calculs coûteux passe de 33 à ~2000 (×60) : le gain marginal par graphe (un peu plus petit) est totalement noyé par la perte d'amortissement du cache.

## Découverte additionnelle — OOM sur instance plus grande

Sur seed=43, n=25/N=500 (278 576 nœuds côté origine, ~23 min), le run z-aware a été tué par OOM après 14 min (MaxRSS ~33.5 Go, limite 32 Go). Cause probable : `active_x_cache`/`edge_cache`, indexés par des combinaisons `(active_x, free_z)` quasi-uniques, accumulent un nombre d'entrées non borné proportionnel au nombre de nœuds explorés (pas à la taille de l'instance) — pas de politique d'éviction. **Non résolu, non contourné** — le run n'a pas été refait avec plus de mémoire, l'expérience s'arrête ici.

## Conclusion

- Correction mathématique : validée (démontrée, pas approximative).
- Modèle GNN : réutilisable sans réentraînement.
- Fragmentation du batching : corrigée (approche ragged).
- Gain net mesuré : **quasi neutre à légèrement négatif** (+3-3.5% de temps, −0.5% de nœuds) sur les tailles testées, à cause de la perte d'amortissement du cache `ActiveXInvariant`.
- Risque mémoire non résolu sur les instances plus grandes/runs plus longs.

**Décision** : ne pas déployer sur V1LD, ne pas porter sur V1CLD. La campagne réelle continue avec les binaires d'origine (`bab_mkp_gecode` / mode `gnn_batched`), inchangés.

## Fichiers créés (tous additifs, aucun impact sur la production)

- `solver_bab/mkp/gecode_bab_gnn_batched_zaware.hpp` / `.cpp` — driver (bucketing `ng_depth` seul, extraction `z_mask`).
- `solver_bab/mkp/gnn_sd_edit_native_batched_zaware.hpp` / `.cpp` — backend (batch ragged, exclusion des scénarios résolus).
- `solver_bab/mkp/main_bab_gnn_batched_zaware.cpp` — CLI standalone (`bab_mkp_gecode_gnn_batched_zaware`).
- `solver_bab/mkp/gnn_sd_edit_scriptable_batched_ragged_export.hpp` — cache d'export TorchScript (`.ragged_scripted_batched.pt`).
- `gnn_sd_edit/gnn_sd_edit_scriptable_batched_ragged.py` — modèle avec pooling scatter (mêmes poids que le checkpoint de production).
- `test_paper/validate_gnn_sd_edit_ragged.py` — validation numérique ragged vs dense.
- `test_paper/run_sanity_gnn_batched_zaware.sh`, `run_compare_zaware_n20_N500.sh`, `run_compare_zaware_n20_N1000.sh`, `run_zaware_n25_N500.sh`, `run_zstats_v1cld_n20_N500.sh` — scripts SLURM utilisés pour cette expérience.
- Cible CMake `bab_mkp_gecode_gnn_batched_zaware` (`solver_bab/CMakeLists.txt`).
