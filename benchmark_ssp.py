"""
benchmark_ssp.py — Sweep de faisabilité SSP stochastique.

Génère une grille d'instances SSP chance-constrained, les résout via CP-SAT,
et caractérise leur difficulté (trivial / non-trivial / infaisable).

Calibration de la tightness
----------------------------
La demande bimodale sur un horizon T nécessite deux groupes d'employés couvrant
les pics AM et PM, qui deviennent de plus en plus disjoints quand T croît.
La règle T/4 est empiriquement validée :
    T=8  → 2.0 ×Q_peak   T=12 → 3.0   T=16 → 4.0   T=20 → 5.0

Correction de l'automate (fix_automaton)
-----------------------------------------
- L_min ≥ n_P + meal_len + 2  : garantit ≥ 2 symboles W par service minimal
- L_max ≥ L_min + 2           : fenêtre de longueur raisonnable
- b_max = L_max − meal_len    : autorise le repas n'importe où dans le service
  (sans ça, un b_max petit force le repas tôt → couverture nulle sur le pic AM)

Usage
-----
    python benchmark_ssp.py                      # paramètres par défaut
    python benchmark_ssp.py --out results.json   # sortie personnalisée
    python benchmark_ssp.py --workers 4          # moins de workers CP-SAT
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "data"))

from ssp_generate.automaton import (
    AutomatonParams,
    sample_automaton_params, build_shift_dfa, is_nonempty_at_length,
)
from ssp_generate.calibrate import calibrate_E, adjust_E_for_feasibility
from ssp_generate.cost import generate_cost
from ssp_generate.demand import generate_demand_scenarios, make_bimodal_nominal
from ssp_generate.groundtruth import solve_instance

# ── Grille de paramètres ──────────────────────────────────────────────────────
T_VALUES   = [8, 12, 16, 20]
N_VALUES   = [20, 50, 100]
EPS_VALUES = [0.05, 0.10, 0.20]
N_REPS     = 3
MASTER_SEED = 42

# ── Solveur ───────────────────────────────────────────────────────────────────
TIME_TRIVIAL_CHECK  = 2.0    # si OPTIMAL en ≤ 2 s → instance triviale
TIME_FULL           = 60.0   # budget CP-SAT pour la résolution complète
N_WORKERS           = 6      # workers parallèles (CP-SAT est multi-threadé)

# ── Génération d'instances ────────────────────────────────────────────────────
TIGHTNESS_PER_PERIOD = 0.25  # tightness = T × TIGHTNESS_PER_PERIOD  (= T/4)
MAX_E                = 200   # plafond E pour calibrate_E (large pour grands T)
MAX_DFA_ATTEMPTS     = 3     # re-tirages de DFA si faisabilité non vérifiée
TIME_FEASIBILITY_CHECK = 5.0 # vérification rapide de faisabilité (secondes)


# ── Correction de l'automate ──────────────────────────────────────────────────

def fix_automaton(aut: AutomatonParams) -> AutomatonParams:
    """
    Corrige les paramètres de l'automate DFA tiré aléatoirement pour éviter
    l'infaisabilité structurelle due à la couverture bimodale.

    - L_min ≥ n_P + meal_len + 2  : ≥ 2 W dans le service minimal
    - L_max ≥ L_min + 2           : fenêtre de longueur > 0
    - b_max = L_max − meal_len    : repas libre dans tout le service
    - win_end ≤ L_max − 1         : fenêtre de pause dans [win_start, L_max-1]
    """
    L_min = max(aut.L_min, aut.n_P + aut.meal_len + 2)
    L_max = max(aut.L_max, L_min + 2)
    b_max = max(L_max - aut.meal_len, aut.a_min + 1)
    win_end = max(min(aut.win_end, L_max - 1), aut.win_start)
    return AutomatonParams(
        L_min=L_min, L_max=L_max, m=aut.m, n_P=aut.n_P,
        meal_len=aut.meal_len, a_min=aut.a_min, b_max=b_max,
        win_start=aut.win_start, win_end=win_end,
    )


# ── Génération d'une instance ─────────────────────────────────────────────────

def make_instance(
    T: int, N: int, epsilon: float, seed: int,
) -> dict:
    """
    Génère une instance SSP stochastique complète.

    Retourne un dict avec : dfa, salary, demand, E, Q_peak, aut, tightness.
    Si aucun DFA faisable n'est trouvé après MAX_DFA_ATTEMPTS, lève RuntimeError.
    """
    seq = np.random.SeedSequence(seed)
    rng_aut, rng_dem, _, rng_cost = (np.random.default_rng(s) for s in seq.spawn(4))

    d_bar  = make_bimodal_nominal(T, rng=rng_dem)
    demand = generate_demand_scenarios(d_bar, rho=0.55, sigma_dem=0.20, N=N, rng=rng_dem)
    salary, _ = generate_cost(
        T, rng_cost,
        b0_range=(1.5, 1.5), b_soir_range=(0.5, 0.5), b_nuit_range=(1.0, 1.0),
    )

    tightness = T * TIGHTNESS_PER_PERIOD   # e.g. T=16 → 4.0

    dfa = aut = meta = E = None
    for _ in range(MAX_DFA_ATTEMPTS):
        # Tirer un DFA non-vide pour cet horizon
        for _inner in range(30):
            raw = sample_automaton_params(rng_aut)
            aut = fix_automaton(raw)
            dfa = build_shift_dfa(aut)
            if is_nonempty_at_length(dfa, T):
                break
        else:
            continue  # 30 tirages intérieurs échoués → nouvel essai DFA

        E, meta = calibrate_E(demand, epsilon, tightness, dfa, T, max_E=MAX_E)
        E, _    = adjust_E_for_feasibility(E, demand, epsilon, dfa, T, max_E=MAX_E)

        # Vérification rapide de faisabilité (ignorée pour T=8, trop coûteux/inutile)
        if T >= 12:
            res = solve_instance(dfa, salary, demand, E, epsilon,
                                 time_limit=TIME_FEASIBILITY_CHECK, verbosity=0)
            if res.status in ("OPTIMAL", "FEASIBLE"):
                break
        else:
            break  # T=8 : on accepte sans vérification

    if meta is None:
        raise RuntimeError(
            f"Aucun DFA faisable trouvé pour T={T} après {MAX_DFA_ATTEMPTS} tentatives"
        )

    return dict(
        dfa=dfa, salary=salary, demand=demand,
        E=E, Q_peak=meta["Q_peak"], aut=aut, tightness=tightness,
    )


# ── Worker (exécuté dans un processus séparé) ─────────────────────────────────

def solve_task(task: tuple) -> dict:
    """
    Génère une instance SSP et la résout en deux étapes :
    1. Résolution rapide (TIME_TRIVIAL_CHECK) → trivial si OPTIMAL
    2. Résolution complète (TIME_FULL)        → statut final, gap, temps

    Retourne un dict de résultats sérialisable en JSON.
    """
    T, N, eps, rep, seed = task
    try:
        inst = make_instance(T, N, eps, seed)
        dfa, salary, demand, E, Q_peak, aut, tightness = (
            inst["dfa"], inst["salary"], inst["demand"],
            inst["E"], inst["Q_peak"], inst["aut"], inst["tightness"],
        )

        res_quick = solve_instance(dfa, salary, demand, E, eps,
                                   time_limit=TIME_TRIVIAL_CHECK, verbosity=0)
        trivial = (res_quick.status == "OPTIMAL")

        res_full = res_quick if trivial else solve_instance(
            dfa, salary, demand, E, eps, time_limit=TIME_FULL, verbosity=0
        )

        gap = None
        if res_full.objective is not None and res_full.lower_bound is not None:
            gap = abs(res_full.objective - res_full.lower_bound) / (abs(res_full.objective) + 1e-9)

        return {
            "T": T, "N": N, "eps": eps, "rep": rep, "seed": seed,
            # instance
            "E": E, "Q_peak": round(Q_peak, 3),
            "tightness": round(tightness, 4),
            "tightness_ratio": round(E / (Q_peak + 1e-9), 3),
            "L_min": aut.L_min, "L_max": aut.L_max,
            "n_P": aut.n_P, "meal_len": aut.meal_len,
            "b_max": aut.b_max, "n_dfa_states": dfa.num_states,
            # difficulté
            "trivial": trivial,
            "quick_time": round(res_quick.solve_time, 4),
            # résultat final
            "status": res_full.status,
            "solve_time": round(res_full.solve_time, 4),
            "objective": res_full.objective,
            "lower_bound": res_full.lower_bound,
            "gap": round(gap, 6) if gap is not None else None,
            "error": None,
        }
    except Exception as exc:
        return {
            "T": T, "N": N, "eps": eps, "rep": rep, "seed": seed,
            "E": None, "Q_peak": None, "tightness": round(T * TIGHTNESS_PER_PERIOD, 4),
            "tightness_ratio": None,
            "L_min": None, "L_max": None, "n_P": None, "meal_len": None,
            "b_max": None, "n_dfa_states": None, "trivial": None,
            "quick_time": None, "status": "ERROR",
            "solve_time": None, "objective": None, "lower_bound": None,
            "gap": None, "error": str(exc),
        }


# ── Construction de la grille de tâches ───────────────────────────────────────

def build_tasks(
    T_vals=T_VALUES, N_vals=N_VALUES, eps_vals=EPS_VALUES,
    n_reps=N_REPS, master_seed=MASTER_SEED,
) -> list[tuple]:
    rng = np.random.default_rng(master_seed)
    tasks = []
    for T in T_vals:
        for N in N_vals:
            for eps in eps_vals:
                for rep in range(n_reps):
                    seed = int(rng.integers(0, 2**31))
                    tasks.append((T, N, eps, rep, seed))
    return tasks


# ── Affichage en ligne ─────────────────────────────────────────────────────────

def _fmt_row(r: dict, done: int, total: int) -> str:
    trivial_s = "trivial" if r.get("trivial") else ("hard" if r.get("trivial") is False else "?")
    t_s  = f"{r['solve_time']:.1f}s" if r["solve_time"] is not None else "N/A"
    gap_s = f"gap={r['gap']:.3f}" if r.get("gap") is not None else ""
    return (
        f"  [{done:3d}/{total}] T={r['T']:2d} N={r['N']:3d} eps={r['eps']:.2f} rep={r['rep']} | "
        f"E={str(r.get('E','?')):>3} tgt={r['tightness']:.1f}× "
        f"{r['status']:<10} {t_s:>6} {trivial_s:<8} {gap_s}"
    )


# ── Point d'entrée ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out", default="bench_results.json",
                        help="Fichier de sortie JSON (défaut: bench_results.json)")
    parser.add_argument("--workers", type=int, default=N_WORKERS,
                        help=f"Workers parallèles (défaut: {N_WORKERS})")
    parser.add_argument("--time-full", type=float, default=TIME_FULL,
                        help=f"Budget CP-SAT par instance en secondes (défaut: {TIME_FULL})")
    args = parser.parse_args()

    tasks = build_tasks()
    n = len(tasks)
    print(
        f"Sweep SSP : {n} instances  "
        f"({len(T_VALUES)}T × {len(N_VALUES)}N × {len(EPS_VALUES)}ε × {N_REPS} reps)\n"
        f"  tightness = T×{TIGHTNESS_PER_PERIOD} (T/4)  ·  "
        f"fix_aut(b_max=L_max−meal_len)  ·  retry={MAX_DFA_ATTEMPTS}\n"
        f"  quick={TIME_TRIVIAL_CHECK}s / full={args.time_full}s  ·  workers={args.workers}",
        flush=True,
    )

    results, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(solve_task, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            done += 1
            print(_fmt_row(r, done, n), flush=True)

    elapsed = time.time() - t0
    n_ok  = sum(1 for r in results if r["status"] in ("OPTIMAL", "FEASIBLE"))
    n_inf = sum(1 for r in results if r["status"] == "INFEASIBLE")
    n_triv = sum(1 for r in results if r.get("trivial"))
    print(
        f"\nTerminé en {elapsed:.0f}s  ·  "
        f"faisable={n_ok}/{n}  infaisable={n_inf}  trivial={n_triv}",
        flush=True,
    )

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"→ {args.out}", flush=True)


if __name__ == "__main__":
    main()
