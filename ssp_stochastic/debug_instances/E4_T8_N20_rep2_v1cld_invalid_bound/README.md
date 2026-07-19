# Instance de débogage : borne v1CLD invalide (E=4, T=8, N=20, rep=2)

## Origine

Générée par `ssp_stochastic.generator_adaptive.generate_episode_fixed_E` avec
`ep_id=8, E=4, T=8, N=20, epsilon=0.1, master_seed=42` (paramètres par défaut
sinon). C'est l'instance `[9/18]` du sweep `sweep_bound_efficacy.py`
(job SLURM 15505881) qui a échoué au sanity check.

Régénération :
```python
from pathlib import Path
from ssp_stochastic.generator_adaptive import generate_episode_fixed_E
from export_to_txt import write_instance  # solver_generic/ssp/export_to_txt.py

ep = generate_episode_fixed_E(ep_id=8, E=4, T=8, N=20, epsilon=0.1,
                               probe_bin=Path("solver_bab/build/ssp_probe"),
                               master_seed=42)
write_instance(ep, Path("out_dir"))
```

## Fichiers

- `automaton.txt`, `cost.txt`, `demand.txt`, `instance.txt` : l'instance SSP
  (format `ssp_loader.hpp`), chargeable directement par `bench_ssp_bab_file`
  ou les outils `debug_*` de `solver_bab/ssp/`.
- `mask_lambda_at_violation.txt` : masque (E×T) + λ (N×m×d, mode v1CLD)
  capturés en flagrant délit — c'est le nœud exact où `more()` a produit
  `lb_int=257552 > sal_ub=37756` (donc invalide, cf. section Diagnostic).
  Format lisible par `debug_replay_violation`.

## Symptôme

```
bench_ssp_bab_file <ce dossier> 30 300 -1
```
donne :

| variante     | objectif | nœuds  | statut  |
|---|---|---|---|
| Gecode base  | 17.867   | 15 885 | OPTIMAL |
| B&B base_ref | 17.867   | 28 483 | OPTIMAL |
| B&B v1CLD    | **20.889** | 12 401 | OPTIMAL (faux) |

Sanity check `[FAIL]`, écart 3.022 — v1CLD se déclare optimal sur une valeur
strictement plus grande que l'optimum réel prouvé par les deux autres
méthodes.

## Diagnostic établi (session du 2026-07-08)

1. **Borne empiriquement invalide** : `lb_int` dépasse `sal_ub` (coût si tous
   les employés travaillent toutes les périodes — plafond physique absolu,
   indépendant du nœud) des centaines de fois pendant le run. Test :
   `SSP_DEBUG_BOUND=1 SSP_DEBUG_DUMP=<path> bench_ssp_bab_file ...` (dump du
   masque+λ au premier dépassement, cf. `gecode_bab_ssp.hpp::more()`).

2. **Bug réel trouvé et corrigé** : `dp_one_employee` (dans
   `ssp_problem.cpp` ET `ssp_cld_problem.cpp`, même défaut dupliqué) avait un
   fallback qui, quand un masque rend le sous-problème d'un employé
   *réellement infaisable* (aucun chemin de longueur T dans l'automate),
   ignorait silencieusement le masque et renvoyait une valeur bidon au lieu
   de `−∞`. Corrigé (commit à faire — actuellement modifié en place, pas
   encore committé). Root cause confirmée par force brute (voir
   `solver_bab/ssp/debug_verify_oracle.cpp`).

3. **Après correctif, le symptôme persiste à l'identique** (mêmes 20.889,
   mêmes 12 401 nœuds). Diagnostic poussé plus loin :
   - θ et ζ sont **prouvés exacts** par force brute (énumération DFS
     indépendante du DP testé) sur le masque+λ exact de
     `mask_lambda_at_violation.txt` (320 vérifications, 0 écart, cf.
     `solver_bab/ssp/debug_replay_violation.cpp`).
   - En recalculant `L1` à la main avec CES θ/ζ garantis exacts, en suivant
     exactement la formule de `v1cld_solver.cpp`, on retrouve
     `lb_salary = 257.552253` — identique à la valeur invalide réelle.
   - **Conclusion : le bug (ou la question ouverte) est dans la formule
     d'agrégation elle-même**, pas dans les oracles θ/ζ. Suspect précis :
     `Δ_i = (1/m) Σ_j max(0, θ_{i,j} − ζ_{i,j})` comme substitut de
     `Δ_i = θ_i − ζ_i` (v1LD, exact et prouvé valide sur papier par
     l'utilisateur). Le `max(0,·)` terme-à-terme avant la moyenne, comparé
     à la borne algébrique `θ_i − ζ_i ≥ moyenne_j(θ_{i,j} − ζ_{i,j})`
     (sans clamp), ne permet pas de conclure dans quel sens va l'inégalité
     une fois le clamp appliqué — point resté ouvert, nécessite la
     dérivation mathématique complète de v1CLD (pas juste v1LD).

## Outils de diagnostic (dans `solver_bab/ssp/`, tous gated/sans impact par défaut)

- `debug_verify_oracle.cpp` : vérifie θ/ζ vs force brute sous masques et
  coûts adverses synthétiques.
- `debug_replay_violation.cpp` : rejoue un (masque, λ) capturé (format de
  `mask_lambda_at_violation.txt`) et recalcule `L1` à la main.
- `debug_root_bound.cpp`, `debug_warmstart_rounds.cpp` : tests de stabilité
  de la descente de sous-gradient (racine seule / rafales répétées).
- `SSP_DEBUG_BOUND=1` (env var) : active un garde-fou dans `more()` qui
  signale `lb_int > sal_ub`.
- `SSP_DEBUG_DUMP=<path>` : combiné au précédent, dump le masque+λ au
  premier dépassement dans `<path>`.

## Prochaine étape

Revoir la dérivation mathématique complète de v1CLD (pas seulement v1LD) —
en particulier si `Δ_i = (1/m)Σ_j max(0,θ_ij−ζ_ij)` est bien un majorant
valide du vrai gap `θ_i − ζ_i`, condition nécessaire pour que soustraire le
knapsack_val construit dessus donne toujours une borne inférieure valide.
