# Note : pourquoi le GNN SLD dégrade fortement à grand N (scénarios), et pistes explorées

Rédigé le 2026-07-03. Objectif : pouvoir reprendre cette investigation sans avoir
à relire toute la conversation qui l'a produite.

## Contexte et symptôme

`gnn/diag_overfit_100.py` entraîne un GNN à prédire, en **un seul forward pass**,
les multiplicateurs de Lagrange λ_ik (un par paire scénario libre × item actif,
donc N_z·n_g valeurs) pour une borne de relaxation lagrangienne (SLD) utilisée
comme borne à chaque nœud d'un Branch & Bound (B&B).

**Le GNN performe très bien à N=50 mais s'effondre à N=500** (n=10 fixe dans
toute cette investigation) :

| | N=50 (job 15080032) | N=500 (job 15081369) |
|---|---|---|
| `pct_better_T50` (% d'instances où le GNN bat la référence sous-gradient T=50) | **24.0%** | **2.2%** |
| L1 final (moyenne train) | 2216.2 | 2241.8 |

Contrainte de production **non négociable** (confirmée explicitement par
l'utilisateur) : le calcul de la borne doit rester **one-shot**. Le B&B appelle
la borne à chaque nœud ; une dizaine d'itérations de sous-gradient est déjà plus
lente que résoudre le problème avec la baseline classique. **Toute solution
hybride avec raffinement itératif post-GNN est donc écartée pour la production**
(elle reste utile comme diagnostic, voir plus bas).

## Historique : d'où part cette investigation

Elle fait suite à une première étude d'ablation (multi-N mixing, 7 runs,
`gnn/runs/ablation_N/`, rapport complet déjà publié en artifact) qui cherchait
à savoir si **mélanger** plusieurs N à l'entraînement dégradait l'apprentissage.
Résultats de cette première étude (résumé) :

- (c) attention θ_i et (d) FiLM : **écartées**, aucun effet mesurable.
- (b) warmup supervisé : effet réel mais **contre-productif** (le retirer aide
  N=50 mais dégrade N=200/500 et augmente le déséquilibre inter-N).
- (e) normalisation par `c_scale` : **inconcluant**, confondu par le clipping de
  gradient par-instance (max_norm=2.0 identique dans tous les runs — sans
  normalisation, la norme de gradient brute explose ×1000, le clip masque tout).
- (a) budget de données mal réparti entre N : **jamais testé**.
- Limite méthodologique découverte : les baselines single-N (1000 graphes) et
  les runs multi-N (3000 graphes) tournent le même nombre d'epochs → 3× plus de
  gradient steps côté multi-N, comparaison confondue.

C'est en creusant cette limite méthodologique que l'utilisateur a noté que
**le vrai signal n'est pas le mélange multi-N mais la difficulté intrinsèque à
grand N**, visible en comparant directement les deux baselines single-N
(tableau ci-dessus). Toute la suite de cette note porte sur **cette
deuxième question**.

## Hypothèses testées dans cette investigation (N=50 vs N=500)

### 1. Sur-lissage du pooling global (oversmoothing) — ÉCARTÉE

**Idée** : `g_pool` (résumé global du graphe, dimension fixe 2×hidden) est une
moyenne sur N_z scénarios — peut-être de plus en plus diluée quand N_z grandit,
empêchant le head de différencier les scénarios entre eux.

**Test** : script diagnostic (poids du GNN **aléatoires**, non entraînés — teste
la propriété structurelle indépendamment de l'apprentissage), mesurant la
similarité cosinus moyenne inter-scénarios de `h_scen_mean`, couche par couche
(x → h_embed → h1 → h2 → h_v).

**Résultat** : le collapse existe (très fort, cosinus ~0.99+ dès `h_scen_mean`)
mais est **quasi identique entre N=50 et N=500**, légèrement *moindre* à N=500 :

| couche | cos N=50 | cos N=500 | delta |
|---|---|---|---|
| x (features brutes) | 0.7875 | 0.7525 | -0.035 |
| h_embed | 0.7962 | 0.7578 | -0.038 |
| h1 | 0.8455 | 0.8070 | -0.038 |
| h2 | 0.8772 | 0.8558 | -0.022 |
| h_v | 0.9103 | 0.8964 | -0.014 |
| h_scen_mean | 0.9964 | 0.9952 | -0.001 |

**Conclusion** : le sur-lissage existe mais n'explique pas le gap différentiel
N=50/N=500 — écarté comme cause principale.

### 2. Dilution du gradient par nœud (facteur 1/N via Λ_sum) — ÉCARTÉE

**Idée** : `lam = h_out * c_scale / N` (ligne 735 de `train_gnn_sld.py`), et
`Λ_sum[k] = Σ_i λ_ik`, donc `∂Λ_sum[k]/∂h_out_i,k = c_scale/N` — la contribution
de gradient de chaque nœud individuel à travers ce terme est mathématiquement
atténuée en 1/N.

**Test** : analyse mathématique/empirique menée par un autre agent en parallèle
(résultat rapporté dans la conversation, pas re-vérifié en détail ici).

**Résultat** : confirmée **au niveau du nœud individuel**, mais **s'annule dans
l'agrégat** — il y a proportionnellement plus de nœuds à grand N, donc la somme
des contributions de gradient vers les poids partagés reste du même ordre de
grandeur. Ne prive pas le réseau de signal d'apprentissage.

**Conclusion** : écartée.

### 3. Biais de mesure : la référence T=50 est-elle comparable entre N ? — CONFIRMÉE MAIS PARTIELLE

**Idée** : le sous-gradient converge dans un espace dual de dimension N_z·n_g —
10× plus grand à N=500. Si T=50 itérations rapprochent bien plus l'optimum à
N=50 qu'à N=500, `pct_better_T50` compare le GNN à des barres de dureté
différentes selon N.

**Test** : `verify_sld(g, T=50)` vs `verify_sld(g, T=1000)` sur 8 instances
profondeur 0 par groupe N, mesure du gap relatif (/c_scale) restant à combler.

**Résultat** :

| | gap relatif L1(T=50)→L1(T=1000), % de c_scale |
|---|---|
| N=50 | 2.41% ± 0.33% |
| N=500 | **1.60%** ± 0.31% |

Le sous-gradient est **relativement plus convergé (barre plus dure) à N=500**
— logique, chaque itération moyenne sur plus de scénarios donc la direction de
mise à jour est moins bruitée.

**Conclusion** : biais réel et confirmé, mais probablement insuffisant à lui
seul pour expliquer le facteur ×11 observé sur `pct_better_T50`. Impossible de
quantifier précisément la part sans un checkpoint du GNN entraîné (voir
section "Travail en cours").

**Point clé qui en découle** (assemblé a posteriori, voir section Synthèse) :
la méthode **itérative** (sous-gradient) ne se dégrade pas à grand N — elle
se comporte même relativement mieux. C'est la régression **one-shot** (GNN) qui
s'effondre.

### 4. Capacité du réseau (dimension `hidden`) — CONFIRMÉE COMME FACTEUR SECONDAIRE, PAS PRINCIPAL

**Test A (poids aléatoires)** : même diagnostic de collapse cosinus que
l'hypothèse 1, en variant `hidden` ∈ {64, 128, 256, 512}.

| hidden | cos N=50 | cos N=500 | écart (N500−N50) |
|---|---|---|---|
| 64 | 0.9962 | 0.9952 | -0.0011 |
| 128 | 0.9954 | 0.9938 | -0.0016 |
| 256 | 0.9949 | 0.9933 | -0.0016 |
| 512 | (timeout, coût O(N_z²·hidden) trop élevé sur CPU) | | |

Augmenter `hidden` ne réduit pas l'écart à l'initialisation — mais ce test est
peu concluant pour une question de capacité (qui se manifeste à l'entraînement,
pas à l'init aléatoire).

**Test B (entraînement réel, 200 epochs, GPU H100)** — jobs
`gnn/runs/ablation_hidden/{N50,N500}_h{64,128}` :

| | hidden=64 | hidden=128 | effet relatif |
|---|---|---|---|
| N=50 (job 15112612/13) | **18.0%** | 9.9% | h128 **-45%** (pénalisé) |
| N=500 (job 15112614/15) | 1.9% | **2.4%** | h128 **+26%** (aide) |

**L'effet s'inverse selon N** : plus de capacité pénalise à N=50 (converge
plus lentement à budget d'epochs fixe) mais aide légèrement à N=500. C'est un
signal réel et cohérent, pas du bruit — mais l'ampleur reste très faible
(2.4% très loin de 18-24%).

**Conclusion** : un effet de capacité existe bel et bien à N=500, mais explique
au mieux une petite fraction du gap. Pas la cause principale.

### 5. Vérification tanh (question ponctuelle, pas une hypothèse de dégradation)

`--lam_max` vaut 0.0 par défaut dans `diag_overfit_100.py`, et
`run_ablation_N_rorqual.sh` ne le passe jamais explicitement → **tanh n'a été
activé sur aucun run** (ablation_N, ni ablation_hidden) : sortie λ toujours
linéaire non bornée. Vérifié dans le code (`train_gnn_sld.py:735`), pas une
piste de dégradation en soi.

## Synthèse : la conclusion qui recombine tout ça

Deux faits mis côte à côte :

1. Le **sous-gradient (itératif)** converge relativement **mieux** à N=500
   qu'à N=50 (hypothèse 3).
2. Le **GNN (one-shot)** s'effondre à N=500 alors qu'il performe bien à N=50.

**Le problème d'optimisation sous-jacent n'est pas intrinsèquement plus dur à
grand N pour une méthode qui itère.** Ce qui ne passe pas à l'échelle, c'est
spécifiquement le **paradigme one-shot** — prédire directement les N_z·n_g
duals en un seul forward pass, sans mécanisme de correction. Ni l'attention, ni
FiLM, ni le pooling, ni (de façon significative) la largeur du réseau ne
changent cette nature fondamentalement one-shot du problème.

## Piste retenue (motivée par les maths, pas encore implémentée)

Dans `verify_sld`/`lagrangian_sld`, la mise à jour du sous-gradient est
`λ_ik += step·(x̄_k − x_ik)`, où `x̄_k` est un **consensus global par item**
(taille n_g, indépendante de N) et `x_ik` la décision propre du scénario i. La
structure de λ* est donc **factorisée** : une petite partie globale + une
correction locale par scénario. Le GNN actuel demande à une tête partagée
d'apprendre implicitement cette structure à partir d'un vecteur concaténé
énorme (`h_v | h_scen | g_pool | ctx`) — et on a montré (hypothèse 1) que le
résumé global lui-même ne se dégrade pas avec N. Donc le goulot est
probablement dans la **tête par-scénario**, pas dans le pooling.

**Proposition** : restructurer la tête pour calculer explicitement
`λ_ik = g(x̄_k, features_locales_du_scénario_i)`, où `x̄_k` (taille n_g,
indépendante de N) est prédit une fois par instance, et `g` est une fonction
légère partagée, appliquée à chaque scénario indépendamment (pas de B&B, pas
d'itération — un simple calcul, donc toujours one-shot et rapide). **Pas encore
implémentée ni testée** au moment de la rédaction.

## Test post-hoc : le λ du GNN est-il un bon point de départ ? — TERMINÉ, résultat négatif et informatif

**But** : mesurer, à titre purement diagnostique (pas pour la production —
la contrainte one-shot reste absolue), à quel point le λ appris par le GNN est
déjà un bon point de départ, en le comparant à un raffinement post-hoc par
sous-gradient (`verify_sld(g, T=K, lam_init=λ_GNN)` pour K=5/10/20/50 vs la
référence froide T=50 et T=1000).

- Code modifié : `gnn/diag_overfit_100.py` (nouveau flag `--save_checkpoint` :
  sauvegarde `checkpoint.pt` + `lambdas_final.pt`, ce dernier contenant le λ
  final **et** toutes les données du graphe — `A_g`, `b_res_g`, `c_g`,
  `c_scale`, `epsilon`, `n_fixed_z1`, `opt` — pour pouvoir ré-évaluer
  `lagrangian_sld`/`verify_sld` hors-ligne sans dépendre des données
  d'origine, souvent sur `SLURM_TMPDIR` donc éphémères après la fin du job).
  `gnn/train_gnn_sld.py` : `GNN.forward()` accepte maintenant
  `return_hidden=True` (retourne aussi `h_v`, `h_scen_mean`, `g_pool`,
  `scen_batch`, `x`, `h_embed`, `h1`, `h2` — utilisé pour l'hypothèse 1/4).
  `run_ablation_N_rorqual.sh` : flag `SAVE_CHECKPOINT=1` câblé.
- **Premier essai (job 15119168) raté** : le dump ne contenait pas
  `A_g`/`b_res_g`/`c_g`, donc inutilisable pour ré-évaluer le λ. Tué et
  corrigé.
- **Job 15120757** (relancé), `gnn/runs/ablation_hidden/ckpt_N500_h64/` :
  N=500, hidden=64, **100 epochs seulement** (run court, pas comparable en
  qualité aux runs 200 epochs de la section précédente), `--save_checkpoint`.
  Terminé.

**Résultat** (15 instances profondeur 0, N=N_z=500 — le sous-ensemble le plus
dur, root du B&B) :

| | L1 moyen | bat cold T=50 |
|---|---|---|
| GNN brut (0 itération) | 3300.22 | 0/15 |
| Cold-start T=50 (référence habituelle) | 3199.37 | — |
| Cold-start T=1000 (quasi-convergé) | 3128.22 | — |
| Warm-start (depuis λ_GNN) T=5 | 3300.22 | 0/15 |
| Warm-start T=10 | 3294.50 | 0/15 |
| Warm-start T=20 | 3251.92 | 0/15 |
| Warm-start T=50 | 3198.95 | 6/15 |

**Le λ brut du GNN est pire que 50 itérations à froid sur ce sous-ensemble, et
démarrer le sous-gradient depuis ce point n'apporte AUCUNE amélioration avant
~10 itérations** (warm_T5 = raw_gnn exactement, sur les 15/15 instances — la
valeur "best" trackée par `verify_sld` reste bloquée à l'itération 0 pendant
les 5 premières itérations). Il faut revenir à ~50 itérations pour rejoindre
grosso modo le niveau de la référence froide — aucun gain de temps du
warm-start ici, y compris à titre diagnostique.

**Interprétation** : ce n'est pas "le λ du GNN est un peu imprécis, quelques
pas de correction suffiraient" — le point de départ est structurellement
mauvais sur les instances les plus dures. Renforce (plutôt qu'infirme) la
conclusion de la section Synthèse : le problème n'est pas de polir un one-shot
déjà bon, c'est que le one-shot lui-même ne capture pas la bonne structure à
grand N. Va dans le sens de prioriser la piste de la tête factorisée plutôt
que d'explorer davantage le raffinement post-hoc (de toute façon écarté pour
la production).

**Limites** : échantillon de 15 instances, checkpoint sous-entraîné (100 vs
200 epochs dans les runs précédents), sous-ensemble le plus dur uniquement
(profondeur 0) — pas nécessairement représentatif de la moyenne sur les 5
profondeurs. Script du test copié dans le dépôt :
`gnn/diag_posthoc_refinement.py` (relit `gnn/runs/ablation_hidden/ckpt_N500_h64/lambdas_final.pt`).

## ⚠️ Travail parallèle découvert, non couvert par cette note

En préparant cette note, j'ai trouvé dans le scratchpad de session et dans le
dépôt des éléments qui semblent correspondre à une **investigation parallèle**,
probablement menée directement par l'utilisateur ou un autre agent, que je
n'ai **pas** analysée ni intégrée ici (je n'ai pas leurs résultats) :

- **`gnn_bipartite/`** (dans le dépôt, fichiers modifiés/trackés par git,
  changements non commités au moment de la rédaction) : `train_gnn_bipartite.py`,
  `train_full_bipartite.py`, `diag_bipartite.py`, + figures
  `convergence_bipartite_vs_homogene.png` — semble être une **architecture
  alternative** (nœuds bipartites scénario/item séparés ?) comparée à
  l'architecture actuelle ("homogène"). Ça recoupe potentiellement la piste
  factorisée proposée ci-dessus — à vérifier.
- Dans le scratchpad de session (`/tmp/.../scratchpad/`, **éphémère**, à
  rapatrier si utile) :
  - `diag_lambda_sensitivity.py` : teste si un même bruit **relatif** sur λ
    dégrade plus L1 à N=500 qu'à N=50 (sensibilité de la borne à l'erreur
    par variable quand il y en a plus).
  - `diag_ystar_sensitivity.py` : teste la sensibilité combinatoire de la
    sélection gloutonne y* — hypothèse que le classement des Δ_i(λ) est
    statistiquement plus fragile à une petite erreur sur λ quand N_z est
    grand (statistique d'ordre : écart entre valeurs consécutives au seuil de
    coupure qui rétrécit avec plus de candidats). **C'est exactement
    l'hypothèse "sensibilité combinatoire de y*" évoquée dans la conversation
    sans être testée** — semble déjà attaquée en parallèle.
  - `test_bipartite_*.py` : tests de correction (ordering, sanity check bout-
    en-bout, smoke test) pour l'architecture bipartite.

Je n'ai pas les résultats de ces scripts — à consolider avec cette note quand
tu reprendras, plutôt que de dupliquer le travail.

## Jobs et fichiers de référence

| Run | Job SLURM | Chemin résultats |
|---|---|---|
| single_N50 (baseline) | 15080032 | `gnn/runs/ablation_N/1b_single_N50/` |
| single_N500 (baseline) | 15081369 | `gnn/runs/ablation_N/1a_single_N500/` |
| hidden ablation N50 h64/h128 | 15112612 / 15112613 | `gnn/runs/ablation_hidden/N50_h64/`, `N50_h128/` |
| hidden ablation N500 h64/h128 | 15112614 / 15112615 | `gnn/runs/ablation_hidden/N500_h64/`, `N500_h128/` |
| checkpoint N500 h64 (test post-hoc) | 15120757 (en cours) | `gnn/runs/ablation_hidden/ckpt_N500_h64/` |

Rapport complet de la première étude (ablation multi-N mixing, hypothèses a-e
du protocole original) : artifact publié séparément dans la conversation

## Suite (session 2026-07-04 → 2026-07-05) : outillage bipartite, budget d'epochs, pooling enrichi, warm-start approfondi

Cette session a démarré **sans relire cette note** — erreur repérée seulement
en fin de session en préparant ce résumé (voir avertissement au point 3
ci-dessous). Le travail a porté sur `gnn_bipartite/diag_overfit_single.py`
(nouveau, miroir bipartite de `gnn/diag_overfit_single.py`) et sur
`gnn/diag_overfit_single.py` existant — diagnostics d'overfit sur **une seule
instance** (contrairement à `diag_overfit_100.py`, qui en utilise 8-100).

### 1. Confirmation indépendante de la conclusion "one-shot vs itératif" (hypothèse 3 / Synthèse ci-dessus)

Sur une instance unique N=500 (root, n=10, m=5), le GNN pur (bipartite ET
homogène) plafonne loin du sous-gradient convergé même en augmentant
fortement le budget d'epochs, avec des rendements très décroissants :
- Bipartite : 500ep → gap +9.6%, ..., 10000ep (×20) → gap +3.4%, jamais négatif.
- Homogène : 500ep → gap +4.75%, ..., 5000ep (×10) → gap +1.89%, jamais négatif seul.

Ce n'est pas une nouvelle découverte mais une **confirmation indépendante**,
sur un protocole différent (une seule instance fixe, budget d'epochs variable,
plutôt que 8-15 instances échantillonnées) : le paradigme one-shot ne passe
pas à l'échelle avec N, cohérent avec la Synthèse ci-dessus.

Détail additionnel : à budget d'epochs égal, l'homogène est systématiquement
meilleur que le bipartite sur cette instance (+4.75% vs +9.6% à 500 epochs)
mais **~10-20× plus lent par epoch** (9.5min vs <1min pour 500 epochs sur
N=500) — tradeoff qualité/coût à garder en tête si le bipartite est retenu
pour d'autres raisons (mémoire, scalabilité).

### 2. Test du pooling enrichi (mean+max+attn concaténés) — cohérent avec l'hypothèse 1 (pooling déjà écarté)

Nouvelle option `use_rich_pool` ajoutée aux deux architectures (`GNN` dans
`train_gnn_sld.py`, `GNNBipartite` dans `train_gnn_bipartite.py`) : `g_pool`
passe de 2h (une seule stratégie, mean OU attention) à 6h (mean+max+attn
concaténés). Testé à budget fixe (500 epochs), N=50 et N=500, les deux
architectures :
- Bipartite N=500 : +9.60% → +9.40% (quasi rien)
- Homogène N=500 : +4.75% → +5.03% (légèrement **pire**)

**Conclusion : enrichir le pooling global n'aide pas.** Ça converge avec
l'hypothèse 1 (le pooling global lui-même n'est pas le goulot) et renforce
donc la piste de la **tête factorisée** proposée dans "Piste retenue"
ci-dessus — c'est le bon prochain test, pas d'autres variantes de pooling.

### 3. Warm-start approfondi (T jusqu'à 600) — nuance la section "Test post-hoc" ci-dessus, ne la contredit pas

Le test post-hoc plus haut (checkpoint 100 epochs, sous-entraîné, 15
instances profondeur 0) concluait que le warm-start n'aide quasi pas avant
T~50. Cette session a répété un test similaire mais avec un modèle **beaucoup
mieux entraîné** (2500-5000 epochs) sur **une seule instance root** (la plus
facile, pas les 15 instances profondeur-0 précédentes) :
- T≤100 : gain faible, plafonne vers +1% de gap (cohérent avec le post-hoc
  précédent — pas de miracle avant ~50-100 itérations)
- T=300-400 : franchit enfin sous l1_conv (gap≤0 vers T≈400)
- 2500 epochs et 5000 epochs de GNN donnent des courbes de warm-start **quasi
  identiques** — tout le budget d'entraînement supplémentaire (2500→5000ep,
  +40min) est inutile une fois le warm-start appliqué.

**⚠️ Important, à ne pas oublier en reprenant** : ce résultat (gap≤0 via
GNN+warm-start T≈400) a été présenté à l'utilisateur comme un "objectif
atteint" avant que l'agent ne retrouve cette note et la contrainte one-shot de
l'intro. **Ce n'est PAS une solution de production** — ~400 itérations de
sous-gradient par nœud de B&B est très largement au-dessus de ce que la
contrainte one-shot autorise (l'intro de cette note dit qu'**une dizaine**
d'itérations est déjà trop lent). Le résultat reste valable comme donnée
diagnostique (même bien entraîné sur l'instance la plus facile, le λ du GNN
seul ne suffit structurellement pas à N=500) mais ne change rien à la
conclusion ni à la piste retenue (tête factorisée).

### 4. Question non résolue soulevée en fin de session : généralisation à ~100k graphes

Tout ce qui précède (cette session ET les hypothèses 1-5 plus haut) porte sur
du **surapprentissage d'une poignée d'instances spécifiques** (1 à 15 selon
le test), jamais sur un modèle entraîné une fois puis évalué sur des
instances inédites. Le passage à l'échelle réelle (~100k graphes) pose une
question différente et **non testée** : un modèle partagé, entraîné avec un
budget de gradient-steps réparti sur tout le dataset, généralise-t-il
correctement à de nouvelles instances N=500 ? Rien dans cette conversation
(ni dans les hypothèses précédentes) n'y répond. `train_full_bipartite.py` /
`train_gnn_sld.py` (entraînement multi-instances avec split train/eval) sont
les outils existants pour l'attaquer, mais pas encore utilisés dans ce sens
pour la question N=500 spécifiquement.

### Nouveaux fichiers / options créés cette session

- `gnn_bipartite/diag_overfit_single.py` (nouveau) : miroir bipartite de
  `gnn/diag_overfit_single.py`.
- `run_overfit_single_rorqual.sh` / `run_overfit_single_bipartite_rorqual.sh`
  (nouveau) : ajout du support `RICH_POOL=1`.
- `GNN.use_rich_pool` (`train_gnn_sld.py`) et `GNNBipartite.use_rich_pool`
  (`train_gnn_bipartite.py`) : option pooling enrichi, désactivée par défaut,
  rétrocompatible.
- Les deux `diag_overfit_single.py` : `lam_diversity` (décomposition
  gauge/résidu vs sous-gradient convergé), `l1_warm_sweep` (T=10/20/50/100),
  sauvegarde `lam_gnn_final.npy` — permet de rejouer un warm-start à
  n'importe quelle profondeur T sans réentraîner.

### Jobs de référence (session 2026-07-04/05)

| Run | Job SLURM | Résultat clé |
|---|---|---|
| Bipartite N50/N500 overfit_single (baseline) | 15319674 / 15319675 | premiers runs, gap +3.0% / +9.6% |
| Bipartite N500 long budget (×5/×10/×20) | 15322985 / 15324304 / 15324305 | gap +5.1% / +4.1% / +3.4%, jamais négatif |
| Bipartite N500 noclip (max_norm=0) | 15322986 | clip de gradient réfuté comme cause |
| Bipartite/Homogène rich_pool (500ep) | 15324837-15324842 | pooling enrichi sans effet |
| Homogène N500 baseline 500/1000/2500/5000ep | 15325192 / 15325193 / 15324838 / 15325918 | +4.75%→+1.89%, rendements décroissants |
| Homogène N500 2500ep v2 (warm-start sweep) | 15325917 | gap≤0 atteint à T≈400 (diagnostic, pas production) |

### Prochaine étape recommandée en reprenant

1. Implémenter et tester la **tête factorisée** (λ_ik = g(x̄_k,
   features_locales_du_scénario_i), x̄_k prédit une fois par instance,
   taille n_g indépendante de N) — seule piste correcte identifiée à ce jour
   pour rester one-shot tout en visant une meilleure scalabilité en N.
2. Si on veut vraiment évaluer la viabilité à ~100k graphes : monter un test
   de généralisation réel (entraînement multi-instances + eval sur instances
   N=500 inédites), pas un nouvel overfit single-instance — question encore
   non attaquée (point 4 ci-dessus).
(tableau + figures), non dupliqué ici.

## Extension (2026-07-05) : la dureté dépend aussi de n (nb d'items), pas que de N

Toute l'investigation ci-dessus fixe n=10 et fait varier N (nb de scénarios).
Question posée : à N **fixe** (=50, le cas le plus facile), doubler n
(nb d'items, donc n_g) rend-il la borne plus dure à apprendre, à budget
d'epochs identique ?

**Protocole** : un jeu de données a été généré ad hoc (10 instances × 3
epsilon, `collect/collect_large_base.py --n 20 --m 5 --N 50`, ~10s de calcul)
et copié dans `data/train_N50/instances_n20_m5.txt` /
`nodes_n20_m5.txt` (nouveaux fichiers, ne remplacent rien — n=10 reste
disponible dans les mêmes dossiers). `gnn/diag_overfit_single.py` sur
l'instance racine (`inst_id=0`, depth=0), config et hyperparamètres
identiques au baseline n=10 déjà publié (`hidden=64`, 500 epochs, lr=3e-4,
seed=42) — seul `--n_val` change (10 → 20).

**Résultat** (job SLURM 15354778, comparé à `N50_diag_baseline_500ep`
existant) :

| | n=10 (N_z=50) | n=20 (N_z=50) |
|---|---|---|
| gap GNN vs sous-gradient convergé | **+0.17%** | **+2.44%** |
| gap GNN vs cold-start T=50 (réf. habituelle) | -3.85% (bat la réf.) | -0.94% (bat à peine) |
| % de l'écart L1(0)→convergé comblé par le GNN | 98.7% | 77.2% |

**Conclusion** : à N identique, doubler n_g dégrade nettement la convergence
à budget d'epochs fixe — le gap résiduel est ~14× plus grand. Cohérent avec
la lecture structurelle de la Synthèse ci-dessus : la dimension du problème
one-shot est **N_z·n_g** (nombre de λ_ik à prédire), donc n_g agit comme un
second axe de mise à l'échelle, au même titre que N_z. Renforce (sans la
tester directement) la piste de la **tête factorisée** : si le goulot est
la tête par-scénario qui doit produire n_g sorties par scénario à partir
d'un vecteur de taille fixe, doubler n_g double mécaniquement la charge de
cette tête, indépendamment de N_z.

**Limites** : une seule instance (comme tous les tests `diag_overfit_single`
de cette note), un seul budget d'epochs (500) — pas encore vérifié si un
budget proportionnellement plus grand (ex. 1000-2000 epochs) referme le gap
n=20 au même niveau que n=10, ni si l'effet se cumule ou sature quand N et n
grandissent simultanément.

**Fichiers** : `gnn/runs/overfit_single/N50_n20_diag_baseline_500ep/`
(figure + `metrics.json`). Dataset n=20 généré en local (login node, module
`python/3.11`+`scipy-stack`), pas via `run_collect_base_rorqual.sh` —
si on veut l'étendre (plus d'instances, autres N), utiliser
`sbatch --export=ALL,N_VAL=20,N_SCEN=<N> collect/run_collect_base_rorqual.sh`.

### Suite immédiate : balayage epochs sur N=50/n=20 — l'axe n est un problème de budget, PAS un plafond structurel (contrairement à N)

Même instance et même config que ci-dessus, epochs ∈ {500, 1000, 2500, 5000}
(jobs 15354961/62/63), comparé directement au balayage déjà publié
N=500/n=10 (section précédente, jobs 15325192/93/15324838/15325918) :

| epochs | N=50, n=20 : gap vs conv | N=500, n=10 : gap vs conv |
|---|---|---|
| 500  | +2.44% | +4.75% |
| 1000 | +1.66% | +4.02% |
| 2500 | +0.48% | +2.98% |
| 5000 | **-0.31%** (dépasse la convergence, 102.9% de l'écart comblé) | +1.89% (jamais négatif, 89.3% comblé) |

**Conclusion, contraste net entre les deux axes** :
- **Axe n (nb d'items, n_g)** : le gap se referme **régulièrement et
  totalement** avec plus d'epochs — à 5000 epochs le GNN dépasse même la
  référence sous-gradient convergée. C'est un problème de **budget
  d'entraînement**, pas de plafond structurel : le one-shot reste apprenable,
  juste plus lentement.
- **Axe N (nb de scénarios, N_z)** : rendements clairement décroissants,
  **jamais négatif** même à 5000 epochs (plafonne à +1.89%, loin de zéro).
  Un plafond structurel, pas (seulement) un problème de budget.

Ce contraste **affine** la Synthèse plus haut ("le paradigme one-shot ne
passe pas à l'échelle avec N") : ce n'est pas la taille brute de la sortie
N_z·n_g qui pose problème en soi (sinon n=20/N=50, qui a la même taille de
sortie que n=10/N=100, se comporterait pareil que du N=100 — pas testé mais
la piste n'est plus tenable telle quelle), mais **spécifiquement le nombre
de scénarios N_z** que la tête doit différencier/pooler. Renforce encore la
piste de la tête factorisée, mais suggère que le goulot est bien dans la
façon dont l'information est agrégée *entre scénarios*, pas dans la charge
de sortie par scénario.

**Limite** : une seule instance testée par point, comme pour tout ce
protocole `diag_overfit_single` — pas de test croisé (ex. N=200, n=20)
pour vérifier si les deux axes se cumulent linéairement ou interagissent.

**Fichiers** : `gnn/runs/overfit_single/N50_n20_diag_baseline_{1000,2500,5000}ep/`.
