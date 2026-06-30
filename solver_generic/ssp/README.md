# ssp — Oracles v1LD pour le SSP shift scheduling multi-employés

## Problème

**Shift Scheduling Problem (SSP) chance-contraint** :
- E employés, T périodes, alphabet Σ = {O, P, R, W} (O=0, P=1, R=2, W=3)
- Chaque employé doit respecter un DFA de quarts légaux partagé
- Objectif : **min** Σ_e Σ_t salary[t] · 1[x[e,t]=W]
- Contrainte stochastique : Pr[ Σ_e 1[x[e,t]=W] ≥ d[t] ∀t ] ≥ 1−ε

## Représentation des variables

```
n_vars = E × T × 4
idx(e, t, σ) = e*T*4 + t*4 + σ    (σ ∈ {O=0, P=1, R=2, W=3})
x[e, t, σ] ∈ {0, 1}               (one-hot par (e, t))
```

Coût de base : `c[e,t,W] = -salary[t]`, `c[e,t,autre] = 0`
(maximiser −salary ≡ minimiser salary)

## Architecture

```
ssp/
├── ssp_problem.hpp        Classe SSPShiftProblem (hérite V1LDProblem)
├── ssp_problem.cpp        Oracles solve_theta et solve_zeta
├── ssp_cld_problem.hpp    Stub — v1CLD non implémenté (TODO)
├── ssp_cld_problem.cpp    Stub
├── ssp_loader.hpp         Chargeur d'instances depuis fichiers texte
├── ssp_loader.cpp         Implémentation du loader
├── test_ssp.cpp           Tests unitaires (§8 du cahier des charges)
└── README.md              Ce fichier
```

## Graphe en couches (DAG déroulé — §3)

Construit **une seule fois** dans le constructeur de `SSPShiftProblem` :
- T+1 couches (t = 0 … T)
- Q nœuds par couche (états du DFA)
- Arc (t, q) → (t+1, q') étiqueté σ : transition DFA δ(q, σ) = q'
- Stocké en liste d'adjacence `adj_[q] = [(σ, q')]` triée

À chaque appel oracle, seuls les **poids d'arcs** (dérivés de `cost`) changent — jamais la structure du graphe.

## solve_theta (§4)

```
max cost^T x   s.t.  regular(x[e,·], DFA) ∀e
```

- **Décompose** en E sous-problèmes indépendants (aucune couverture dans le domaine de base)
- Chaque employé : **DP topologique** sur le DAG, couche par couche
- Complexité : O(E · T · Q · 4)
- Poids négatifs : gérés nativement (DAG acyclique, pas de Bellman-Ford)

## solve_zeta (§5)

```
max cost^T x   s.t.  regular(x[e,·], DFA) ∀e,  Σ_e x[e,t,W] ≥ d[t,i] ∀t
```

- La couverture **couple les E employés** → NP-difficile
- Résolution **exacte** via **OR-Tools CP-SAT** (déjà dans le build)
  - E contraintes `AddAutomaton(shift[e], DFA)` — transitions pré-calculées
  - Indicateurs de travail `work[e][t] ∈ {0,1}`, liés à `shift[e][t] == W`
  - Couverture : `Σ_e work[e][t] ≥ d[t,i]`
  - Objectif : `AddElement` pour `cost_et[shift[e][t]]` (pas de BoolVar par σ)
- **Exactitude requise** : une solution approchée rendrait la borne v1LD invalide

## Convention de signe (§6)

Le framework est **max-natif** (L1 = borne supérieure).

- SSP min → negate : `c[e,t,W] = -salary[t]`
- `cost` passé aux oracles peut être de signe quelconque (λ ajusté)
- `δ_i = max(0, θ_i − ζ_i) ≥ 0` car X^i ⊆ domaine de base ✓

## Format d'instance texte (§7)

Exporter depuis HDF5 avec `export_to_txt.py` (script Python utilitaire) puis charger avec `load_ssp_instance(dir)` :

| Fichier | Contenu |
|---------|---------|
| `instance.txt` | `E T N EPSILON INTERCHANGEABLE` |
| `automaton.txt` | États, initial, acceptants, transitions (src sym dst) |
| `cost.txt` | `T SALARY s0 s1 … sT-1` |
| `demand.txt` | N lignes de T entiers : `d[t,0] … d[t,T-1]` par scénario |

## Tests (§8)

```bash
# Depuis solver_generic/
cmake --build build_cc --target test_ssp_shift
./build_cc/test_ssp_shift
```

Tests couverts : découplage theta, exactitude zeta (d=0), δ≥0 ∀i, graphe stable, déterminisme, poids négatifs, couverture effective.
