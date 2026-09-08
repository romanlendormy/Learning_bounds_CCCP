# SSP Stochastic — Générateur de datasets

Générateur de données pour le problème de **shift scheduling chance-constrained stochastique (SSP)**.
Chaque épisode contient : un automate DFA de quarts légaux, des coûts salariaux, des scénarios
de demande in-sample et out-of-sample.

---

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install numpy h5py pytest
```

---

## Utilisation rapide

Génération à `(E, T, N, ε)` fixés (pas de `tightness_target` universel — cf.
`calibrate_adaptive.py` : le ratio Q_peak/E n'a pas d'effet stable selon T,
validé empiriquement) :

```python
from pathlib import Path
from ssp_stochastic.generator_adaptive import generate_episode_fixed_E

ep = generate_episode_fixed_E(ep_id=0, E=15, T=12, N=50, epsilon=0.1,
                               probe_bin=Path("solver_bab/build/ssp_probe"))
# ep : dict compatible avec writers.write_episode / export_to_txt.write_instance
```

Points d'entrée réels côté génération d'instances pour le B&B GNN batché :
`gnn_ssp/gen_ssp_instance_custom.py` (via `generate_episode_fixed_E`) et
`gnn_ssp/gen_ssp_instance_fixed_automaton.py` (réutilise un automate fixé
+ `calibrate_adaptive.calibrate_demand_scale`, cf.
`gnn_ssp/NOTE_ssp_hard_instance_generation.md`).

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

with open_dataset("dataset/dataset.h5") as h5f:
    ids  = list_episodes(h5f)          # [0, 1, 2, ...]
    data = read_episode(h5f, ep_id=0)  # dict avec arrays + DFA
```

Conversion vers le format texte 4-fichiers attendu par `ssp_loader.cpp`
(côté B&B Gecode) : `solver_generic/ssp/export_to_txt.py`.

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
| `calibrate_adaptive.py` | Calibration de l'échelle de demande par bissection (cible : fenêtre de nœuds B&B, cf. `gnn_ssp/NOTE_ssp_hard_instance_generation.md`) |
| `writers.py` | Lecture/écriture du dataset HDF5 |
| `generator_adaptive.py` | Génération d'un épisode à (E,T,N,ε) fixés |

Modules retirés (2026-09, pipeline pré-adaptatif jamais reconstruit après la
suppression de `ssp_stochastic` au commit `e2f7b37` — cf.
`solver_generic/ssp/export_to_txt.py`) : `generator.py`, `calibrate.py`,
`loader.py`, `groundtruth.py`, `config_default.yaml`.

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
```
