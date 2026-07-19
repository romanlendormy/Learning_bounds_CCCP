# Snapshot architecture GNN V1LD — avant refonte (2026-07-06)

But : garder une trace fidèle de l'architecture "v21" actuelle (les deux
variantes, homogène et bipartite) avant une refonte simplifiée visant
l'agnosticité à n (items) et N (scénarios). Ce document est factuel
(section 1-2) puis analytique (section 3 : blocages identifiés).

Référence code : `gnn/train_gnn_v1ld.py` (classe `GNN`, homogène) et
`gnn_bipartite/train_gnn_bipartite.py` (classe `GNNBipartite`).
Config par défaut testée en overfit_single (`gnn/diag_overfit_single.py`) :
`hidden=64, use_max_pool=True, use_theta_attn=True, use_film=True,
use_rich_pool=False, use_factored_head=False`.

## 1. Architecture homogène (`gnn/train_gnn_v1ld.py::GNN`)

**Graphe** : un nœud par paire (scénario libre i, item actif k), donc
`N_z · n_g` nœuds. Nœuds indexés `i_idx * n_g + k_idx`.

**Features nœud (7m+5, m=nb contraintes)** — construites dans `build_graph`
(L119-289) :
| Bloc | Dim | Description |
|---|---|---|
| `f_c` | 1 | `c_k / c_scale` |
| `f_A` | m | `A_ijk / b_orig_j` (capacité *originale*, pas résiduelle) |
| `f_frac` | m | `clip(A_ijk / b_res_j, 10)` |
| `f_eff` | m | `log1p(f_c / f_A)` |
| `f_rank` | m | rang de Dantzig normalisé par `(n_g-1)` ∈ [0,1] |
| `f_cap` | m | `b_res_j/b_j` du scénario i (même valeur ∀k du scénario) |
| `f_scen_min/mean` | 1+1 | min/mean sur j de `f_cap`, niveau scénario |
| `f_A_mean/std_scen` | m+m | moyenne/écart-type de `f_A` **sur les N_z scénarios**, niveau item |
| `f_cap_std/sat_scen` | 1+1 | écart-type / taux de saturation (<0.1) de `f_cap` sur j, niveau scénario |

**Arêtes** (L232-257) :
- `item_varies` : clique complète entre les `n_g` items d'un même scénario, poids `1/n_g` — `N_z · C(n_g,2) · 2` arêtes.
- `scenario_varies` : clique complète entre les `N_z` scénarios d'un même item, poids `1/N_z` — `n_g · C(N_z,2) · 2` arêtes.
- **Total : O(n_g·N_z·(n_g + N_z))** arêtes.

**Contexte (9-dim)**, normalisé par `LayerNorm` : `[depth/n, frac_fixed,
tightest, curr_cost/c_scale, epsilon, n_fixed_z1/N, log(n_g), log(N_z),
N_z/n_g]`.

**Couches** (L631-681) :
1. `embed` : Linear(7m+5→h) → ReLU → Linear(h→h) → ReLU → LayerNorm
2. `[FiLM]` (si `use_film`) : `γ(ctx)⊙h + β(ctx)`, init identité
3. `conv1` : `GatedGraphConv(h, num_layers=2)` (aggr='add' + edge_weight ≈ moyenne pondérée) → LayerNorm
4. `conv2` : idem + skip `h1`
5. `proj` : Linear(h→4h) → ReLU → Linear(4h→2h) = `h_v`
6. Pool scénario : `mean` (+ `max` concaténé si `use_max_pool`) sur les `n_g` nœuds du scénario → `h_scen`
7. Pool graphe `g_pool` : `mean` uniforme, OU attention `θ_i` (si `use_theta_attn`), OU les deux + max concaténés (si `use_rich_pool`) — **un seul vecteur pour les N_z scénarios**
8. Tête : `[h_v | h_scen | g_pool | ctx] → h → ReLU → h → ReLU → 1` (ou tête factorisée si `use_factored_head`, cf. ci-dessous)
9. `λ = h_out · c_scale / N` (tanh-clip optionnel si `lam_max>0`)

**Tête factorisée** (`use_factored_head=True`, L650-661, 739-761 — implémentée
mais jamais testée à l'échelle) : `λ_ik = gauge_k(h_item_k, ctx) +
resid_ik(h_v, h_scen_i, ctx)`, avec `h_item_k = mean_i(h_v)` sur les N_z
scénarios partageant l'item k (taille indépendante de N_z), et `resid_ik`
centré à moyenne nulle par item. **Seul composant conçu explicitement pour
découpler la charge de la tête de N_z.**

## 2. Architecture bipartite (`gnn_bipartite/train_gnn_bipartite.py::GNNBipartite`)

**Graphe** : `HeteroData` avec deux types de nœuds — `item` (n_g) et `scen`
(N_z) — reliés par des arêtes `item→scen` / `scen→item`, une par paire (i,k).
**O(n_g·N_z) arêtes** (pas de clique item-item ni scénario-scénario — plus
léger que l'homogène).

**Features nœud item (1+2m)** : `[c_k/c_scale, A_mean_scen(m), A_std_scen(m)]`.
**Features nœud scénario (2m)** : `[b_res_i(m), b_res_i/b_i(m)]`.
**Features arête (4m)** : `[A_ijk, A_ijk/b_res_i, (c_k/c_scale)/max(A_ijk,1), dantzig_rank]`.
Même contexte 9-dim que l'homogène.

**Couches** : `embed_item`/`embed_scen` (Linear+ReLU) → `[FiLM]` optionnel →
3 aller-retours `GeneralConv` (item→scen puis scen→item, avec edge_attr,
skip depuis r>0) → `proj_item`/`proj_scen` (Linear→2h) → pooling global
`g_pool` sur les N_z nœuds-scénario (mean/max/attn, mêmes variantes que
l'homogène) → tête `[h_k | h_i | edge_attr | g_pool | ctx] → h → h → 1`.
**Pas de tête factorisée implémentée côté bipartite.**

## 3. Blocages identifiés pour l'agnosticité n / N

Par ordre de confiance décroissante :

**(1) Coût et goulot de message-passing dépendant de N_z, structurel.**
Homogène : O(n_g·N_z²) arêtes scénario-scénario — la clique complète des
scénarios pour chaque item aggrège l'info des N_z-1 voisins en **une seule**
opération `GatedGraphConv` (aggr='add' pondéré 1/N_z ≈ moyenne). C'est un
champ moyen figé dans la topologie du graphe, pas un vrai passage de message
multi-hop : à N_z=500, chaque nœud voit une moyenne sur 499 scénarios en un
seul saut, ce qui écrase le signal spécifique-scénario. Cohérent avec
`NOTES_N_scaling_investigation.md` ("le goulot est spécifiquement le nombre
de scénarios N_z que la tête doit différencier/pooler").
Bipartite : évite la clique (O(n_g·N_z) arêtes), mais le même goulot
réapparaît au pooling global (`g_pool` = un seul vecteur mean/max/attn sur
les N_z nœuds-scénario) — déplacé, pas résolu.

**(2) Le contexte injecte `log(n_g), log(N_z), N_z/n_g` en dur dans la tête
(et dans FiLM).** Ces 3 scalaires sont vus directement par les dernières
couches (`head`, ligne 665/810 ; `film`, ligne 678/701-705). Un modèle censé
être agnostique à n/N ne devrait pas apprendre à conditionner sa sortie sur
la valeur numérique brute de n_g/N_z — c'est le mécanisme le plus direct par
lequel un (n,N) jamais vu à l'entraînement sort de la plage connue du réseau
(extrapolation d'un scalaire, pas généralisation structurelle). Piste
plausible pour expliquer l'échec de généralisation n=10→n=20 déjà observé
(`test_inference_checkpoint.py`, jobs 15355556/57 : 0% de généralisation,
9/10 instances pires que L1(λ=0)).

**(3) Hypothèse de batch à n_g uniforme.** `train_gnn_v1ld.py:712` et
`train_gnn_bipartite.py:453` lisent `data.n_g` comme un **scalaire unique**
pour tout le batch (commentaire explicite L724-726 : "suppose n_g uniforme
dans le batch"). Impossible en l'état de mélanger des graphes n=10 et n=20
dans un même mini-batch — blocage pipeline pur, indépendant de
l'architecture, mais qui empêche directement l'objectif "un seul modèle
entraîné sur n=10 et n=20 ensemble" sans un vrai batching par graphe (comme
c'est déjà fait pour N_z via `scen_batch`/`n_scen_per`).

**(4) Features agrégées sur N_z dont la distribution statistique dépend de
N_z.** `f_A_mean_scen`/`f_A_std_scen` (homogène) et `A_mean`/`A_std` (item,
bipartite) sont des moyennes/écarts-types calculés sur les N_z scénarios
libres. Par loi des grands nombres, leur variance d'échantillonnage diminue
quand N_z grandit : le réseau voit une feature statistiquement plus
"bruitée" à N_z=20 qu'à N_z=500 pour un même signal sous-jacent. Pas
nécessairement fatal, mais un candidat à vérifier en premier si un composant
spécifique s'avère mal généraliser entre tailles de N.

**(5) Complexité déjà démontrée non pertinente, toujours active par
défaut.** L'ablation N (`[[project_ablation_N_gnn_v1ld]]`, 2026-07-03) a
déjà tranché :
- `use_theta_attn` : **indiscernable de la baseline** sur toutes les métriques.
- `use_film` : **écarté**, quasi identique à 4 décimales près (le module
  ne représente que ~0.8% des paramètres).
Or `diag_overfit_single.py` active les deux **par défaut**
(`--no_theta_attn`/`--no_film` pour les désactiver). `use_max_pool` n'a
jamais été isolé dans l'ablation (confondu avec les deux autres dans la
config "baseline"). Ce sont les candidats les plus sûrs à retirer en premier
dans une refonte simplifiée — gain de simplicité sans perte de performance
déjà démontrée par l'ablation.

**(6) Résolution du rang de Dantzig dépend de n_g.** `f_rank` est normalisé
par `(n_g-1)` (L188-191) : la granularité du rang change avec n_g (ex.
résolution 1/9 à n_g=10 vs 1/19 à n_g=20). Effet probablement mineur par
rapport à (1)-(3), à garder en tête seulement.

## Pistes déjà notées ailleurs (non dupliquées ici)

Voir `NOTES_N_scaling_investigation.md` et
`[[project_overfit_single_n_scaling]]` / `[[project_oneshot_bound_constraint]]`
pour l'historique complet des expériences de scaling en N et en n déjà
menées (toutes en overfit single-instance) et la contrainte one-shot de
production (pas de raffinement itératif post-GNN).
