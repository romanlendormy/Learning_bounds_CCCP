"""
calibrate_adaptive.py — Calibration de l'échelle de la demande à E, T, N, ε
FIXÉS (au lieu de dériver E d'un tightness_target fixe, cf. calibrate.py).

Contexte
--------
Le balayage empirique (bench_ssp_bab_file + ssp_probe) a montré que le ratio
Q_peak/E ("tightness") n'a PAS le même effet sur la difficulté selon T — à
tightness=2.0, T=8 donne ~25% d'instances "dures mais résolues" alors que
T=16 donne 0% quel que soit le tightness testé (structurellement trop dur).
Un tightness_target universel ne peut donc pas être maintenu pour des
(E, T, N) arbitraires.

Principe
--------
Au lieu de faire confiance à un ratio théorique, on laisse le VRAI solveur
(Gecode base, sonde rapide via ssp_probe) juger la difficulté, et on
ajuste l'échelle de la demande par bissection jusqu'à tomber dans une
fenêtre de nœuds cible :

    échelle trop faible → résolu trivialement (peu de nœuds)   → augmenter
    échelle trop forte  → INFEASIBLE ou trop de nœuds (>hi)    → diminuer
    dans la fenêtre [target_lo, target_hi]                      → accepter

Un premier point de départ (sans appel solveur) est estimé avec la même
formule que calibrate.calibrate_E, inversée : au lieu de E = Q_peak*ratio,
on choisit l'échelle de la demande pour viser Q_peak = E/ratio_guess.
Ça réduit le nombre d'itérations de bissection nécessaires quand (T, N, ε)
sont proches du régime déjà validé empiriquement (T=8, ratio≈1.9).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_SSP_CPP_DIR = Path(__file__).resolve().parent.parent / "solver_generic" / "ssp"
if str(_SSP_CPP_DIR) not in sys.path:
    sys.path.insert(0, str(_SSP_CPP_DIR))


def initial_scale_guess(demand_raw: np.ndarray, epsilon: float, E: int,
                        ratio_guess: float = 1.9) -> float:
    """
    Point de départ sans appel solveur : même calcul que calibrate_E mais
    inversé — on cherche l'échelle à appliquer à demand_raw pour que son
    quantile (1-ε) de pic (max sur T) vaille E / ratio_guess.
    """
    d_quantile = np.quantile(demand_raw, 1.0 - epsilon, axis=1)  # (T,)
    Q_peak_natural = float(d_quantile.max())
    if Q_peak_natural <= 1e-9:
        return 1.0
    target_Q_peak = E / ratio_guess
    return target_Q_peak / Q_peak_natural


def classify_probe(status: str, nodes: int, target_lo: float, target_hi: float) -> str:
    """Retourne 'too_easy' / 'target' / 'too_hard' pour piloter la bissection."""
    if status == "INFEASIBLE":
        return "too_hard"   # demande trop élevée → diminuer l'échelle
    if nodes > target_hi:
        return "too_hard"
    if status == "OPTIMAL" and nodes < target_lo:
        return "too_easy"
    # OPTIMAL dans la fenêtre, ou TIMEOUT/FEASIBLE avec nodes dans la fenêtre
    return "target"


def _write_instance_scaled(out_dir: Path, dfa, salary: np.ndarray,
                           demand_TN_scaled: np.ndarray, E: int, epsilon: float) -> None:
    """Écrit les 4 fichiers texte (cf. export_to_txt.write_instance) pour un
    essai de bissection — évite le détour par HDF5."""
    from export_to_txt import write_instance

    T, N = demand_TN_scaled.shape
    ep = dict(dfa=dfa, E=E, T=T, N=N, epsilon=epsilon,
             demand=demand_TN_scaled, salary=salary)
    write_instance(ep, out_dir)


def _probe(inst_dir: Path, probe_bin: Path, time_lim: float) -> tuple[str, int]:
    proc = subprocess.run(
        [str(probe_bin), str(inst_dir), str(time_lim)],
        capture_output=True, text=True, timeout=time_lim + 15,
    )
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if not line:
        raise RuntimeError(f"ssp_probe n'a rien retourné pour {inst_dir} "
                          f"(stderr: {proc.stderr[:500]})")
    parts = line.split(",")
    # dir,E,T,N,eps,Q,n_trans,status,nodes,fails,time_s,objective
    status, nodes = parts[7], int(parts[8])
    return status, nodes


def calibrate_demand_scale(
    dfa, salary: np.ndarray, demand_raw: np.ndarray,  # demand_raw : (T, N) à échelle de référence
    E: int, epsilon: float,
    probe_bin: Path,
    ratio_guess: float = 1.9,
    target_lo: float = 1e4,
    target_hi: float = 1e6,
    probe_time_lim: float = 5.0,
    max_iters: int = 8,
) -> dict:
    """
    Cherche une échelle de demande telle que l'instance (dfa, salary,
    demand_raw*échelle, E, epsilon) tombe dans [target_lo, target_hi] nœuds
    sous Gecode base (probe_time_lim secondes), sans jamais toucher à E.

    Retourne un dict :
      scale, demand (T,N) int, status ('target'|'no_target_found'),
      n_iters, history (liste de (scale, status, nodes, category))
    """
    scale = initial_scale_guess(demand_raw, epsilon, E, ratio_guess)
    lo_scale, hi_scale = None, None  # bornes connues too_easy / too_hard
    history = []

    with tempfile.TemporaryDirectory(prefix="ssp_calib_") as tmp:
        inst_dir = Path(tmp) / "trial"
        for _ in range(max_iters):
            demand_scaled = np.maximum(0, np.round(demand_raw * scale)).astype(int)
            _write_instance_scaled(inst_dir, dfa, salary, demand_scaled, E, epsilon)
            status, nodes = _probe(inst_dir, probe_bin, probe_time_lim)
            cat = classify_probe(status, nodes, target_lo, target_hi)
            history.append((scale, status, nodes, cat))

            if cat == "target":
                return dict(scale=scale, demand=demand_scaled, status="target",
                           n_iters=len(history), history=history)

            if cat == "too_easy":
                lo_scale = scale
                scale = scale * 2.0 if hi_scale is None else (scale + hi_scale) / 2.0
            else:  # too_hard
                hi_scale = scale
                scale = scale / 2.0 if lo_scale is None else (scale + lo_scale) / 2.0

    # Pas de fenêtre cible atteinte dans le budget d'itérations — l'appelant
    # décide (redessiner un nouvel automate, ou accepter le dernier essai).
    return dict(scale=scale, demand=demand_scaled, status="no_target_found",
               n_iters=len(history), history=history)
