# NOTE — Génération d'instances SSP-CCP dures pour REF (2026-08-20)

## Contexte

Objectif : construire des instances SSP-CCP où REF (Gecode CP pur, sans
borne) galère vraiment, pour pouvoir juger honnêtement si une borne
(V1CLD/GNN) apporte quelque chose. Un tirage aléatoire "brut" de la demande
est presque toujours soit trivialement facile, soit infaisable — la zone
"dure mais faisable" est étroite et doit être calibrée, pas laissée au
hasard (cf. `ssp_stochastic/calibrate_adaptive.py`, qui existe déjà pour ça).

Cette note documente ce qui caractérise VRAIMENT la difficulté (pas juste
"une grosse instance"), suite à l'investigation du 20 août sur l'instance
E10_T8_N100 (REF : 944s, conservée dans
`gnn_ssp/test_data/kept_hard_instances/E10_T8_N100_ref944s_hard/`).

## Ce qui NE caractérise PAS la difficulté

- **La taille brute (E, T, N) seule** : un balayage complet E∈{4,6,8}×T∈{8,10,12}×N∈{50,100,500}
  n'a montré AUCUNE tendance propre de difficulté REF avec la taille — parce
  que la calibration (bissection sur l'échelle de la demande, ciblant un
  nombre de nœuds fixe) neutralise activement cet effet par construction.
- **L'automate seul** : réutiliser EXACTEMENT le même automate (+ salaire)
  de l'instance dure E10_T8_N100 à E=4 donne une instance **INFEASIBLE**
  (0.38s), à E=6 donne une instance **triviale** (2.60s, OPTIMAL). L'automate
  n'est donc pas "dur" en soi.
- **Un ratio de tension scalaire simple** (Q_peak/E, quantile (1-ε) de la
  demande par période / nombre d'employés) : vaut 0.5 pour l'instance dure
  E10_T8_N100 — une valeur modérée, pas extrême. Cohérent avec le docstring
  de `calibrate_adaptive.py` : ce ratio n'a pas d'effet universel selon T,
  encore moins selon E.

## Ce qui CARACTÉRISE la difficulté (établi empiriquement, 4 points de données)

**C'est la combinaison (automate donné) × (E) qui compte, pas l'un sans l'autre.**
Pour un automate fixé (celui de E10_T8_N100, 19 états, validé pour T=8),
en balayant E (N et l'automate inchangés) :

| E | échelle demande calibrée | résultat REF |
|---|---|---|
| 4 | 0.3126 (aucune fenêtre cible trouvée en 12 essais de bissection) | 0.38s **INFEASIBLE** |
| 6 | 0.3158 | 2.60s **OPTIMAL** (trivial) |
| 10 | 0.48–0.52 (selon N) | **300s+ TIMEOUT, 15-77 MILLIONS de nœuds**, pas résolu |

Confirmé sur **trois N différents** (50, 100, 500) à E=10 : tous durs,
malgré des N très différents — **N n'est presque pas un facteur ici**, c'est
E qui l'est. Il y a donc une vraie **transition de phase en E**, spécifique
à cet automate : trop peu d'employés (E=4) → infaisable ; assez d'employés
(E=6) → trivial ; **pile à la frontière (E=10 pour CET automate) → dur**.

Note : l'échelle de demande calibrée grimpe nettement avec E (0.31→0.48-0.52)
parce que la bissection cherche toujours la même fenêtre de nœuds cible —
mais la valeur d'échelle en elle-même n'est qu'un sous-produit, pas la cause
: c'est la position de E par rapport à la structure combinatoire de CET
automate qui détermine si on tombe dans la bande dure.

## Recette reproductible pour un automate donné

1. **Fixer l'automate** (et le T pour lequel il a été validé non-vide —
   changer de T demande de re-vérifier `is_nonempty_at_length`, pas testé).
2. **Fixer N** à une valeur pratique (50-100, peu d'effet observé sur la
   difficulté pour cet automate — mais pas garanti pour un autre automate).
3. **Balayer E** (recalibrer l'échelle de demande par bissection à chaque E
   via `calibrate_demand_scale`), et classer chaque résultat REF :
   INFEASIBLE / trivial (résolu en quelques secondes) / dur (timeout, des
   dizaines de millions de nœuds).
4. La bande dure se trouve à la frontière entre "trivial" et "infaisable" —
   affiner par bissection sur E lui-même (comme on bissecte déjà l'échelle)
   pour la localiser précisément.
5. Chaque automate a probablement SA PROPRE frontière en E — pas de recette
   universelle "E=10 est toujours dur", ce résultat est spécifique à
   l'automate testé.

## Outils

- `gnn_ssp/gen_ssp_instance_custom.py` — tire un NOUVEL automate (+ calibre
  la demande) pour un (E,T,N,ε) donné. Utilisé pour explorer plusieurs
  automates.
- `gnn_ssp/gen_ssp_instance_fixed_automaton.py` — **réutilise** l'automate
  ET le salaire d'une instance déjà générée (`--source_dir`), ne redessine
  que la demande (nécessaire, sa taille dépend de N) et la recalibre pour un
  nouveau (E,N). C'est l'outil pour la recette ci-dessus (étape 3).
  Limite connue : rho/sigma_dem (forme de la demande) ne sont pas
  récupérables depuis le format 4-fichiers déjà écrit, donc réinitialisés au
  milieu des plages par défaut (0.55, 0.20) plutôt que les valeurs exactes
  d'origine — confond légèrement la forme de la demande, mais pas l'automate
  (ce qu'on isole).
- `solver_bab/ssp/test_fixed_automaton_ref_only.sh` — script sbatch
  (CPU, pas besoin de GPU pour cette question) qui illustre l'usage complet.

## Instances dures conservées

Dans `gnn_ssp/test_data/kept_hard_instances/` :
- `E10_T8_N100_ref944s_hard/` — l'originale (REF : 944.41s, OPTIMAL, objectif=51.254).
- `E10_T8_N50_ref300s_timeout_77Mnodes/` — même automate/salaire, N=50 (REF :
  300s TIMEOUT, 77 108 923 nœuds, FEASIBLE seulement, objectif partiel=51.254).
- `E10_T8_N500_ref300s_timeout_16Mnodes/` — même automate/salaire, N=500
  (REF : 300s TIMEOUT, 15 847 761 nœuds, FEASIBLE seulement, objectif
  partiel=50.770).

Ces deux dernières sont en fait PLUS dures que l'originale (timeout sans
résolution du tout, contre résolution complète en 944s) — candidates de
choix pour évaluer GNN batché GPU sans retomber sur l'artefact "REF trop
facile".

Deux instances supplémentaires, issues du balayage 27 combos (E∈{4,6,8}×T∈{8,10,12}×N∈{50,100,500},
job 19324305, cf. discussion 2026-08-20) où REF lui-même a buté sur le
plafond de 90s de ce balayage sans prouver l'optimalité — automates
DIFFÉRENTS de celui de la famille E10_T8 ci-dessus, tirés indépendamment
pendant le balayage :
- `E6_T10_N500_ref90s_timeout_notoptimal/` — REF : 90.00s TIMEOUT, FEASIBLE
  seulement, **optimum réel INCONNU** (jamais prouvé, contrairement à la
  famille E10_T8 ci-dessus).
- `E8_T8_N500_ref90s_timeout_notoptimal/` — idem, REF : 90.00s TIMEOUT,
  FEASIBLE seulement, **optimum réel INCONNU**.

Ces deux-là n'étaient PAS protégées à l'origine (elles vivaient dans des
dossiers `ssp_native_test_*_sweep` génériques, vulnérables à un futur
balayage réutilisant les mêmes noms) — copiées ici le 2026-08-20 après coup.
Si on veut un optimum de référence pour elles, il faudra relancer REF avec
un budget bien plus généreux (comme fait pour E10_T8_N100 : 3600s a suffi,
944.57s réel) — pas encore fait, coût inconnu a priori pour CES deux-là.

## Conclusion provisoire pour le rapport (2026-08-20)

Après plusieurs pistes d'optimisation (batching GPU, clique→moyenne,
saut des employés fixés dans ζ/θ) qui donnent des gains réels mais modestes
(quelques dizaines de %, jamais un changement de régime), et un balayage
27 combos qui ne montre AUCUNE tendance de difficulté propre avec la taille
nominale (E,T,N) -- le message principal n'est probablement pas "l'optimisation
n'a pas marché", mais **"ce problème est trop sensible au tirage précis
(automate + demande) pour qu'une taille nominale garantisse une comparaison
significative REF vs GNN/V1CLD"**. Cohérent avec la calibration par bissection
qui doit chercher activement une fenêtre de difficulté plutôt que la laisser
émerger naturellement de la taille (cf. section "pourquoi calibrer" plus haut).
