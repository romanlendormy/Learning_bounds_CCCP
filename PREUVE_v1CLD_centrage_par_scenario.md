# Preuve de validité de v1CLD — double relâchement à une seule couche de multiplicateurs

**Date** : 2026-08-13
**Contexte** : investigation de l'invalidité observée de la borne v1CLD sur SSP (et,
par formule partagée, sur MKP), déclenchée par une instance de debug (E=4,T=8,N=20)
en juillet 2026 (historique détaillé supprimé depuis lors du dépôt). Ce document reprend la construction depuis
zéro, avec une seule famille de multiplicateurs $\lambda_{i,j}$ indexée par
(scénario $i$, contrainte $j$) — pas de couche $\mu$ séparée — pour rester
compatible avec l'architecture GNN existante (un groupe de nœuds par
(scénario, contrainte), cf. `gnn_ssp/train_v1CLD_ssp.py`).

## 0. Mise en garde méthodologique (leçon de cette investigation)

Deux "contre-exemples" numériques ont d'abord semblé réfuter cette preuve
(`lb_salary` dépassant `sal_ub`, le plafond salarial universel). En creusant,
les deux nœuds capturés se sont révélés **globalement infaisables** : la
vraie valeur jointe $\zeta_i(0)$ (les $m$ contraintes ensemble, calculée
exactement via le solveur v1LD/CP-SAT, $\lambda=0$) vaut $-\infty$ pour un
nombre de scénarios dépassant la marge $\varepsilon N$ autorisée — c'est-à-dire
qu'aucune complétion du nœud ne peut satisfaire la chance-contrainte, quel que
soit $\lambda$. Dans ce cas, une borne "invalide au sens de `sal_ub`" est en
réalité le comportement correct (signal d'élagage) : `sal_ub` ne borne que les
solutions *faisables*, et il n'y en a aucune à ce nœud.

**Avant de traiter un dépassement de `sal_ub` comme un contre-exemple, il faut
vérifier que le nœud est faisable** (assez de scénarios avec $\zeta_i(0)$ fini
pour atteindre $1-\varepsilon$). Cette vérification a été omise pendant plusieurs
tours d'investigation ; elle est indispensable pour ne pas rejeter à tort une
preuve correcte.

## 1. Le problème original

$$
\max_{\mathbf{x}\in D(\mathbf{x})} \Big\{ f(\mathbf{x}) \;\Big|\; \frac1N\sum_{i=1}^N z_i(\mathbf{x}) \ge 1-\varepsilon \Big\},
\qquad z_i(\mathbf{x}) := \mathbb 1\Big\{\bigwedge_{j=1}^m C_j(\mathbf{x},\xi^i)\Big\}
$$

- $\mathbf x$ : la décision partagée (unique), $D(\mathbf x)$ son domaine de base.
- $f$ : objectif linéaire.
- $C_j(\cdot,\xi^i)$, $j=1,\dots,m$ : les $m$ contraintes du scénario $i$ (pour
  SSP : $m=T$, une par période ; pour MKP : $m$ dimensions de capacité).
- $z_i$ : indicateur que le scénario $i$ est *entièrement* satisfait (les $m$
  contraintes ensemble — conjonction stricte, pas de seuil interne).

## 2. La construction (une seule couche de $\lambda_{i,j}$)

Pour chaque $(i,j)$, $\lambda_{i,j}\in\mathbb R^d$, avec la seule condition :

$$
\boxed{\sum_{j=1}^m \lambda_{i,j} = 0 \quad \text{pour chaque } i \quad \text{(centrage \emph{par scénario}, pas global)}}
$$

Sous-problèmes :

$$
\theta_{i,j}(\lambda) := \max_{X\in D(X)} \big[f(X) + \Lambda_{i,j}\cdot X\big],
\qquad
\zeta_{i,j}(\lambda) := \max_{X\in D(X):\,C_j(X,\xi^i)} \big[f(X) + \Lambda_{i,j}\cdot X\big]
$$

où $\Lambda_{i,j}$ est le coût effectif utilisé dans le code
(`cost_ij = c + lam[i][j]/p_ij - Lambda_sum_i[i]/p[i]`, cf.
`solver_bab/ssp/v1cld_solver_scentered.cpp`), construit pour que
$\sum_j \Lambda_{i,j} = 0$ automatiquement, quel que soit le $\lambda$ brut fourni.
Convention : $\zeta_{i,j}=-\infty$ si le sous-problème est infaisable (aucune
séquence automate-valide ne satisfait $C_j$ compte tenu du masque courant).

Agrégats par scénario (moyenne, pas somme — cohérent avec l'échelle du code,
cf. §5) :

$$
\hat\theta_i := \frac1m\sum_{j=1}^m \theta_{i,j}(\lambda), \qquad
\hat\zeta_i := \frac1m\sum_{j=1}^m \zeta_{i,j}(\lambda) \quad (=-\infty \text{ si un terme est} -\infty)
$$

Sélection gloutonne des scénarios "à satisfaire" (couverture $Z$), puis :

$$
L_1(\lambda) := \sum_{i:\,y_i^*=0} p_i\hat\theta_i \;+\; \sum_{i:\,y_i^*=1} p_i\hat\zeta_i
$$

C'est exactement ce que calcule le code (`solve_v1cld` / `solve_v1cld_scentered`),
une fois le centrage corrigé pour être **par scénario**.

## 3. Théorème

> Pour **toute** famille $\lambda=(\lambda_{i,j})$ satisfaisant $\sum_j\lambda_{i,j}=0$
> pour chaque $i$ — sans aucune restriction de magnitude — $L_1(\lambda) \ge v^*$,
> où $v^*$ est la valeur optimale du problème original (ou $v^*=-\infty$ si le
> nœud courant est infaisable, auquel cas l'énoncé est trivial).

## 4. Démonstration

### Étape 1 — Dualité faible par scénario

Fixons $i$. Notons $\theta_i(0) := \max_{X\in D(X)} f(X)$ (ne dépend pas de $i$
ni de $\lambda$) et $\zeta_i(0) := \max_{X\in D(X):\,\bigwedge_jC_j(X,\xi^i)} f(X)$
(la vraie valeur jointe, $\lambda=0$).

**Reformulation exacte** (poids $1$ par copie, pas $1/m$) : en introduisant $m$
copies $X_1,\dots,X_m\in D(X)$ forcées égales,

$$
m\cdot\theta_i(0) = \max_{X_1=\cdots=X_m\in D(X)} \sum_{j=1}^m f(X_j)
$$

(vrai : si toutes les copies valent $X$, $\sum_j f(X) = m f(X)$).

**Relâchement du consensus** $X_j=\bar X$ avec multiplicateurs $\gamma_j:=\Lambda_{i,j}$,
sous la seule condition $\sum_j\gamma_j=0$ :

$$
m\cdot\theta_i(0) \;\le\; \max_{X_1,\dots,X_m\in D(X)} \sum_j\big[f(X_j)+\gamma_j\cdot X_j\big]
= \sum_j \theta_{i,j}(\lambda)
$$

C'est de la **dualité faible pure** : on maximise la même fonction sur un
ensemble plus grand (sans la contrainte $X_j=\bar X$). Le terme en $\bar X$
libre s'annule exactement grâce à $\sum_j\gamma_j=0$ — sans cette condition,
$\max_{\bar X}$ diverge ($+\infty$), d'où la nécessité du centrage. **Aucune
hypothèse sur la magnitude de $\gamma$** n'intervient à aucun moment.

Divisant par $m$ : $\theta_i(0) \le \hat\theta_i$.

Le même argument, en imposant en plus $C_j(X_j,\xi^i)$ dans chaque copie,
donne $\zeta_i(0) \le \hat\zeta_i$ (avec la convention $-\infty\le$ tout, donc
trivial si $\zeta_i(0)=-\infty$).

### Étape 2 — Couche scénario (Song et al., MKP chance-constrained programs — cas $\mu=0$)

$$
B(0) := \sum_i \theta_i(0) + \max_{z\in Z}\sum_i\big(\zeta_i(0)-\theta_i(0)\big)z_i \;\ge\; v^*,
\qquad Z=\Big\{z\in\{0,1\}^N : \frac1N\sum_iz_i\ge1-\varepsilon\Big\}
$$

Valide pour tout multiplicateur de couche 1 (ici pris $\equiv 0$, cas
particulier légitime du théorème).

### Étape 3 — Monotonie

$B(\theta,\zeta) := \max_{z\in Z}\sum_i\big[\theta_i(1-z_i)+\zeta_iz_i\big]$ est
croissante en chaque $\theta_i$ et chaque $\zeta_i$ : à $z$ fixé, coefficients
$1-z_i\ge0$ et $z_i\ge0$ ; un maximum de fonctions croissantes est croissant.

Comme $\hat\theta_i\ge\theta_i(0)$ et $\hat\zeta_i\ge\zeta_i(0)$ pour **chaque**
$i$ (Étape 1) :

$$
B(\hat\theta,\hat\zeta) \;\ge\; B(\theta(0),\zeta(0)) \;\ge\; v^*
$$

### Étape 4 — Le glouton calcule exactement $B(\hat\theta,\hat\zeta)$

$\max_{z\in Z}\sum_i(\hat\zeta_i-\hat\theta_i)z_i$ est un problème de sélection
à seuil de cardinalité (choisir au moins $\lceil(1-\varepsilon)N\rceil$ scénarios
pour maximiser une somme de coefficients $\le0$) — résolu exactement en triant
$\hat\Delta_i:=\hat\theta_i-\hat\zeta_i$ croissant et en prenant les plus petits
(argument d'échange standard, indépendant des valeurs numériques). Si des
$\hat\Delta_i=+\infty$ sont *forcés* dans la sélection (pas assez de scénarios à
$\hat\Delta$ fini pour atteindre le seuil), $B=+\infty$ — cohérent avec
"nœud infaisable, aucune borne finie n'a de sens" (cf. §0).

**Conclusion** : $L_1(\lambda) = B(\hat\theta,\hat\zeta) \ge v^*$, pour tout
$\lambda$ centré par scénario. $\blacksquare$

## 5. Remarque sur l'échelle (moyenne vs somme)

Le code utilise $\hat\theta_i=\frac1m\sum_j\theta_{i,j}$ (moyenne), pas une
somme. C'est cohérent avec l'Étape 1 : la reformulation exacte utilise un
poids **1** (pas $1/m$) par copie parce que le coût de base $c$ dans le code
n'est *pas* pré-divisé par $m$ (`c[k] = -salary[t]`, sans facteur). La preuve
ci-dessus est menée directement à l'échelle du code (poids 1, division par
$m$ à la fin) — pas besoin de renormaliser $\lambda$.

## 6. Ce que cette preuve couvre — et ce qu'elle ne couvre pas

- **Couvre** : la validité de $L_1(\lambda)$ comme majorant de $v^*$, pour
  **tout** $\lambda$ satisfaisant le centrage par scénario, à **tout** nœud
  (feasible ou non), sans hypothèse de magnitude. Vérifié :
  - algébriquement (le centrage par scénario annule exactement le terme libre) ;
  - numériquement, en vérifiant *directement* $\theta_i(0)\le\hat\theta_i$ et
    $\zeta_i(0)\le\hat\zeta_i$ scénario par scénario sur les deux nœuds capturés
    (`solver_bab/ssp/debug_replay_violation.cpp`, section "Verification directe") —
    toutes les inégalités tiennent, y compris aux nœuds infaisables.
- **Ne couvre pas** : pourquoi le B&B v1CLD *original* (centrage **global**,
  pas par scénario — le code actuellement en production,
  `solver_generic/v1cld_solver.cpp`) donnait une réponse finale fausse
  (20.889 au lieu de 17.867 sur l'instance de debug). Cette preuve ne
  s'applique qu'à la version **corrigée** (centrage par scénario). La version
  globale reste à ré-investiguer séparément — la question de départ (pourquoi
  le centrage global casse la borne) reste ouverte, et mérite sa propre
  vérification (probablement : le centrage global ne satisfait *pas*
  $\sum_j\Lambda_{i,j}=0$ par scénario en général, donc l'Étape 1 ne s'applique
  pas à cette version — mais ceci reste à démontrer proprement, pas juste
  assumé).

## 7. Fichiers de référence

- Preuve/dérivation initiale (couche scénario, Song et al.) : discussion du 2026-08-13.
- Implémentation testée au moment de l'investigation (diagnostic local, jamais
  commité, ne touchait pas au code de production) : `v1cld_solver_scentered.cpp`
  + `gecode_bab_ssp_scentered.hpp/.cpp` (centrage par scénario, une seule couche
  $\lambda_{i,j}$) — le correctif a été intégré directement dans
  `solver_generic/v1cld_solver.cpp` le jour même (cf. commentaire `FIX (2026-08-13)`),
  rendant ces fichiers diagnostiques obsolètes.
- `solver_bab/ssp/debug_replay_violation.cpp` (toujours en production) : outil de
  vérification directe des inégalités de dualité faible, par nœud capturé.
- Historique de l'investigation de juillet (instance de debug E=4,T=8,N=20) :
  supprimé lors du nettoyage du dépôt (2026-09).
