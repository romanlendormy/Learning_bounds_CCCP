"""
txt_format.py — Constantes et utilitaires partagés par writers.py / readers.py.

`epsilon` est stocké comme fraction entière exacte (`epsilon_num`/`epsilon_den`), jamais
comme flottant, pour éviter toute ambiguïté d'arrondi entre l'écriture et la lecture
(le solveur déduit `ceil((eps_den - eps_num) * N / eps_den)`).
"""

from __future__ import annotations

import math
import subprocess
from fractions import Fraction

GENERATOR_VERSION = "0.1.0"


def epsilon_to_fraction(epsilon: float, max_denominator: int = 1000) -> tuple[int, int]:
    """Convertit un epsilon flottant (ex. 0.1) en fraction entière réduite (1, 10)."""
    if not (0.0 < epsilon < 1.0):
        raise ValueError(f"epsilon doit être dans (0, 1), reçu {epsilon}")
    frac = Fraction(epsilon).limit_denominator(max_denominator)
    return frac.numerator, frac.denominator


def saa_threshold(epsilon_num: int, epsilon_den: int, N: int) -> int:
    """`ceil((1 - epsilon) * N)` calculé exactement en entiers."""
    return math.ceil((epsilon_den - epsilon_num) * N / epsilon_den)


def get_git_commit() -> str:
    """Hash court du commit courant, pour `meta.txt`. "unknown" si indisponible."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"
