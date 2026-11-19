"""
ccpo_libs.py — Emplacement des bibliothèques C++ CCPO appelées depuis Python (ctypes).

Elles sont compilées par CMake avec le reste du dépôt (cf. README, « Building ») :
  ccpo_lagrangian.so   L(λ) et sous-gradient (solver_generic/ccpo/ccpo_lagrangian_bind.cpp)
  presolve_exact.so    presolve exact certifié (solver_generic/ccpo/presolve_bind.cpp)
  ccpo_graph.so        graphe du GNN en C++ (solver_bab/ccpo/ccpo_graph_bind.cpp)
Dossier de build : $BUILD_DIR, par défaut <dépôt>/build.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUILD_DIR = Path(os.environ.get("BUILD_DIR", REPO / "build"))
_SUBDIR = {"ccpo_lagrangian": "solver_generic", "presolve_exact": "solver_generic", "ccpo_graph": "solver_bab"}


def lib_path(name: str) -> Path:
    """Chemin de la bibliothèque `name` (sans extension) dans le dossier de build."""
    return BUILD_DIR / _SUBDIR[name] / f"{name}.so"
