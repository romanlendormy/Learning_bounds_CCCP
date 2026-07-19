# SSP Stochastic — Générateur de datasets

Générateur de données pour le problème de **shift scheduling chance-constrained stochastique (SSP)**.
Chaque épisode contient : un automate DFA de quarts légaux, des coûts salariaux, des scénarios
de demande in-sample et out-of-sample, et optionnellement une solution de vérité-terrain (CP-SAT).

---

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install numpy ortools pyyaml pytest
```

---

## Utilisation rapide

```bash
# 10 épisodes avec la config par défaut
python -m ssp_stochastic.generator --output dataset/

# Config personnalisée + graine reproductible
python -m ssp_stochastic.generator --config ma_config.yaml --n 50 --seed 2024 --output dataset/

# Charger un épisode en Python
from ssp_stochastic.loader import load_episode, build_bipartite_graph
inst  = load_episode("dataset/episodes/ep_00000")
graph = build_bipartite_graph(inst)
```

---

## Structure du dataset

```
dataset/
├── dataset.h5       # Toutes les données (HDF5, un groupe par épisode)
├── config.yaml      # Configuration utilisée pour la génération
├── seeds.txt        # Graine entière de chaque épisode
└── manifest.txt     # Récapitulatif tabulaire (texte, lisible sans h5py)
```

---

## Format HDF5

Le fichier `dataset.h5` contient un groupe HDF5 par épisode, nommé `ep_XXXXX`
(zero-padded sur 5 chiffres).

### Datasets par groupe

| Nom           | Shape         | dtype   | Description |
|---------------|---------------|---------|-------------|
| `salary`      | `(T,)`        | float64 | Coût salarial par période |
| `demand`      | `(T, N)`      | int32   | Scénarios in-sample |
| `demand_oos`  | `(T, K_oos)`  | int32   | Scénarios out-of-sample |
| `d_bar`       | `(T,)`        | int32   | Demande nominale bimodale |
| `aut_accepting` | `(n_acc,)`  | int32   | États acceptants du DFA |
| `aut_trans`   | `(n_trans, 3)`| int32   | Transitions `[src, sym_int, dst]` |

Encodage alphabet dans `aut_trans` : O=0, P=1, R=2, W=3 (ordre alphabétique).

### Attributs scalaires par groupe

`ep_id`, `E`, `T`, `N`, `K_oos`, `epsilon`, `quantile`, `rho`, `sigma_dem`,
`aut_num_states`, `aut_initial` + champs de meta préfixés `meta_`
(`meta_L_min`, `meta_dfa_num_states`, `meta_ratio_tightness`, …).

### Lecture en Python

```python
from ssp_stochastic.writers import open_dataset, read_episode, list_episodes
from ssp_stochastic.loader  import load_episode, build_bipartite_graph

# Lire un épisode brut
with open_dataset("dataset/dataset.h5") as h5f:
    ids  = list_episodes(h5f)          # [0, 1, 2, ...]
    data = read_episode(h5f, ep_id=0)  # dict avec arrays + DFA

# Charger comme Instance (objet structuré)
with open_dataset("dataset/dataset.h5") as h5f:
    inst = load_episode(h5f, ep_id=0)

# Raccourci pour un épisode isolé
from ssp_stochastic.loader import load_episode_from_file
inst = load_episode_from_file("dataset/dataset.h5", ep_id=0)

# Graphe bipartite pour GNN
graph = build_bipartite_graph(inst)
```

---

## Chaîne mathématique de génération de la demande

### 1. Demande nominale bimodale

Un profil journalier bimodal `d̄ ∈ ℕᵀ` est construit par superposition de deux gaussiennes
puis normalisation :

```
g(t) = A₁·exp(-(t - μ₁T)²/(2σ₁²T²))  +  A₂·exp(-(t - μ₂T)²/(2σ₂²T²))
d̄[t] = round(max_demand · g(t) / max g)    (entier, ≥ 1)
```

Paramètres par défaut : pic matin à `μ₁ = 0.35`, pic soir à `μ₂ = 0.75`, `max_demand = 8`.

### 2. Processus AR(1) log-normal

Pour chaque scénario `k = 1, …, N`, on génère une trajectoire de T périodes via un AR(1)
gaussien transformé en log-normal :

**Étape 1 — Chaîne AR(1) :**

```
η[0, k]  = ε[0, k]
η[t, k]  = ρ · η[t-1, k]  +  √(1 - ρ²) · ε[t, k]        ε[t,k] ~ N(0, 1) i.i.d.
```

Le facteur `√(1 - ρ²)` garantit que `Var(η[t, k]) = 1` en régime stationnaire.

**Étape 2 — Transformation log-normale avec correction de biais :**

```
d[t, k]  = max(0, ceil( d̄[t] · exp(σ · η[t, k]  −  σ²/2) ))
```

Le terme `-σ²/2` est la correction de Jensen assurant `E[exp(σ·η - σ²/2)] = 1`,
soit `E[d[t, k]] ≈ d̄[t]`.

**Paramètres :**

| Paramètre | Plage par défaut | Description |
|-----------|-----------------|-------------|
| `ρ`       | [0.3, 0.8]      | Autocorrélation temporelle |
| `σ`       | [0.1, 0.3]      | Volatilité log-normale |
| N         | 50              | Scénarios in-sample |
| K_oos     | 2000            | Scénarios out-of-sample |

### 3. Propriétés garanties

- `d[t, k] ≥ 0` pour tout `(t, k)`
- `E[d[t, k]] ≈ d̄[t]` (biais ≤ 5% vérifié par test)
- `Corr(d[t, k], d[t-1, k]) ≈ ρ` (vérifié par test à ±0.05)
- Entièrement déterministe pour une graine fixée

---

## Formulation mathématique (SAA)

### Variables

- `x[e, t] ∈ {O, P, R, W}` : activité de l'employé `e` à la période `t`
- `z[k] ∈ {0, 1}` : indicateur de couverture du scénario `k`

### Contraintes

**Automate (quarts légaux) :** pour chaque employé `e`,

```
(x[e, 0], x[e, 1], …, x[e, T-1])  est accepté par le DFA
```

Le DFA encode : durée de service [L_min, L_max], au plus `m` W consécutifs, exactement
`n_P` pauses P, un repas de `meal_len` symboles R, fenêtre de placement du repas [win_start, win_end].

**Couverture (chance constraint SAA) :**

```
∑_{e} 𝟏[x[e,t] = W]  ≥  d[t, k]  si  z[k] = 1    ∀ t, k
∑_k z[k]  ≥  ⌈(1 - ε) · N⌉
```

**Objectif :**

```
min  ∑_e  ∑_t  s[t] · 𝟏[x[e,t] = W]
```

---

## Modules Python

| Module | Rôle |
|--------|------|
| `demand.py` | Génération de la demande (AR(1) log-normal) |
| `cost.py` | Génération des coûts salariaux |
| `automaton.py` | Construction et minimisation du DFA |
| `calibrate.py` | Calibration de E par quantile de demande |
| `writers.py` | Lecture/écriture des fichiers texte |
| `loader.py` | Chargement + graphe bipartite pour GNN |
| `groundtruth.py` | Solveur CP-SAT (OR-Tools) |
| `generator.py` | Générateur en lot (CLI) |

---

## Reproductibilité

Toutes les graines sont dérivées hiérarchiquement via `numpy.random.SeedSequence` :

```
master_seed
  └── ep_seqs[i]  (une séquence par épisode, spawn de master)
        ├── sub_seqs[0]  → rng automate + paramètres épisode (ρ, σ, aut_params)
        ├── sub_seqs[1]  → rng coûts
        ├── sub_seqs[2]  → rng demande in-sample
        └── sub_seqs[3]  → rng demande OOS
```

Aucun état global n'est modifié (`numpy.random.seed` jamais appelé).

---

## Tests

```bash
pytest ssp_stochastic/tests/ -v
# 47 passent, 1 skippé (micro-énumération : instance micro infaisable par construction)
```
