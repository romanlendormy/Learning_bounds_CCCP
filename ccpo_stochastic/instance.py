"""
instance.py — Tirage de l'instance mère CCPO.

Modèle à un facteur, paramétré par fraction de variance expliquée :

    beta_i = sigma_i * sqrt(h_i) / sigma_f
    psi_i  = (1 - h_i) * sigma_i**2
    Sigma  = sigma_f**2 * outer(beta, beta) + diag(psi)

`psi_i > 0` est garanti par construction dès que `h_i < 1` — aucun rejet-retirage requis
sur ce point (contrairement à une paramétrisation directe par chargements bruts).

`sigma_f` est fixe par défaut (non identifiable indépendamment de `beta`, cf. README.md
pour la dérivation de 0.0335) : exposé en paramètre pour traçabilité, pas pour être
varié sans revalider les statistiques cibles (tests/test_instance.py).

Ce module ne couvre que l'instance mère ; la perturbation en scénarios est dans scenarios.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SIGMA_F_DEFAULT = 0.0335


def _truncated_lognormal(
    rng: np.random.Generator,
    size: int,
    median: float,
    s: float,
    lo: float,
    hi: float,
    max_tries: int = 100,
) -> np.ndarray:
    """
    Log-normale de médiane `median` et forme `s`, tronquée à [lo, hi] par rejet-retirage.

    mu = ln(median) : la médiane d'une LogNormale(mu, s) vaut exp(mu) par définition.
    """
    if not (0 < lo < hi):
        raise ValueError(f"bornes invalides: lo={lo}, hi={hi}")
    mu = math.log(median)
    out = np.empty(size, dtype=np.float64)
    filled = 0
    tries = 0
    while filled < size:
        if tries >= max_tries:
            raise RuntimeError(
                f"_truncated_lognormal: taux de rejet trop élevé après {max_tries} tirages "
                f"(median={median}, s={s}, bounds=[{lo}, {hi}]) — bornes probablement "
                "incohérentes avec la médiane/forme demandées."
            )
        tries += 1
        batch = rng.lognormal(mean=mu, sigma=s, size=size - filled)
        mask = (batch >= lo) & (batch <= hi)
        n_ok = int(mask.sum())
        out[filled : filled + n_ok] = batch[mask]
        filled += n_ok
    return out


def _induce_rank_correlation(
    rng: np.random.Generator, x: np.ndarray, y: np.ndarray, rho: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Réordonne (x, y) — sans toucher à leurs marginales — pour induire une corrélation de
    rang (Spearman) approximativement égale à `rho`. Méthode d'Iman-Conover : on tire des
    scores gaussiens corrélés (rho), et on réordonne x/y selon le rang de ces scores.

    rho == 0.0 : no-op, x et y sont retournés inchangés (ordre de tirage original).
    """
    if rho == 0.0:
        return x, y
    n = x.size
    w1 = rng.standard_normal(n)
    z = rng.standard_normal(n)
    w2 = rho * w1 + math.sqrt(1.0 - rho**2) * z

    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    x_out = np.empty(n, dtype=np.float64)
    y_out = np.empty(n, dtype=np.float64)
    x_out[np.argsort(w1)] = x_sorted
    y_out[np.argsort(w2)] = y_sorted
    return x_out, y_out


@dataclass
class MotherInstance:
    """Instance mère CCPO : rendements espérés et covariance à facteur unique."""

    n: int
    sigma_f: float
    sigma: np.ndarray  # (n,) volatilités
    h: np.ndarray  # (n,) fraction de variance expliquée par le facteur commun
    c: np.ndarray  # (n,) rendements espérés
    beta: np.ndarray  # (n,) chargements sur le facteur
    psi: np.ndarray  # (n,) variances idiosyncratiques
    Sigma: np.ndarray  # (n, n) matrice de covariance

    @property
    def rho_bar(self) -> float:
        """Corrélation moyenne hors diagonale (theoriquement rho_ij = sqrt(h_i h_j))."""
        d = np.sqrt(np.diag(self.Sigma))
        corr = self.Sigma / np.outer(d, d)
        mask = ~np.eye(self.n, dtype=bool)
        return float(corr[mask].mean())

    @property
    def sigma_bar(self) -> float:
        return float(self.sigma.mean())

    @property
    def c_bar(self) -> float:
        return float(self.c.mean())

    @property
    def Sigma_bar(self) -> float:
        """Moyenne de toutes les entrées de Sigma, diagonale incluse (sert au budget de risque B, cf. discretize.py)."""
        return float(self.Sigma.mean())


def draw_mother_instance(
    rng: np.random.Generator,
    n: int,
    sigma_f: float = SIGMA_F_DEFAULT,
    capm_alignment: float = 0.0,
    sigma_median: float = 0.045,
    sigma_s: float = 0.15,
    sigma_bounds: tuple[float, float] = (0.036, 0.069),
    h_alpha: float = 5.0,
    h_beta: float = 4.4,
    c_median: float = 2.5e-3,
    c_s: float = 0.85,
    c_bounds: tuple[float, float] = (1.0e-4, 1.2e-2),
) -> MotherInstance:
    """
    Tire une instance mère CCPO à `n` actifs.

    `capm_alignment` (rho, 0 dans toutes les expériences) induit une
    corrélation de rang entre `c_i` et `h_i` via `_induce_rank_correlation`.
    """
    if not (-1.0 < capm_alignment < 1.0):
        raise ValueError(f"capm_alignment doit être dans (-1, 1), reçu {capm_alignment}")

    sigma = _truncated_lognormal(rng, n, sigma_median, sigma_s, *sigma_bounds)
    h = rng.beta(h_alpha, h_beta, size=n)
    c = _truncated_lognormal(rng, n, c_median, c_s, *c_bounds)

    c, h = _induce_rank_correlation(rng, c, h, capm_alignment)

    beta = sigma * np.sqrt(h) / sigma_f
    psi = (1.0 - h) * sigma**2

    if np.any(psi <= 0):
        raise RuntimeError(
            "draw_mother_instance: psi_i <= 0 détecté — h_i doit rester < 1 strictement "
            "(check 2 bloquant)."
        )

    Sigma = sigma_f**2 * np.outer(beta, beta) + np.diag(psi)

    return MotherInstance(
        n=n, sigma_f=sigma_f, sigma=sigma, h=h, c=c, beta=beta, psi=psi, Sigma=Sigma
    )
