# Note : désactivation du passage de λ dans la tangente

## Ce qui existe dans le code

Dans `gnn/train_gnn_v1ld.py` (ligne 735), la sortie du GNN est :

```python
lam = self.lam_max * torch.tanh(lam_unscaled / self.lam_max) if self.lam_max > 0 else lam_unscaled
```

Quand `lam_max > 0`, λ est borné dans `[-lam_max, +lam_max]` via une tanh.  
Quand `lam_max == 0` (défaut argparse), λ est linéaire — **la tanh est bypassée**.

## Ce qui a été changé dans run_train_scaled.sh

`LAM_MAX` est maintenant une variable shell avec valeur par défaut `0` :

```bash
LAM_MAX="${LAM_MAX:-0}"
```

Et `--lam_max "$LAM_MAX"` est passé explicitement au script Python.

**Effet :** la tanh est garantie désactivée sauf si on surcharge explicitement `LAM_MAX`.

## Pourquoi désactiver

- Avec `lam_max > 0`, le gradient de tanh sature quand `|lam| → lam_max`
  → le modèle ne peut plus augmenter λ librement, ce qui peut bloquer la descente.
- En pratique sur les runs précédents, `lam_max` n'était jamais passé (défaut 0),
  donc aucun checkpoint existant n'est affecté.

## Pour réactiver

```bash
sbatch --export=ALL,LAM_MAX=5.0 run_train_scaled.sh
```
