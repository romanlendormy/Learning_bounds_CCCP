"""
generator_adaptive.py — Génération d'un épisode SSP à (E, T, N, ε) FIXÉS,
sans passer par un tightness_target universel (cf. calibrate_adaptive.py
pour la justification : le ratio Q_peak/E n'a pas un effet stable sur la
difficulté selon T — validé empiriquement, pas juste supposé).

Usage :
    from ssp_stochastic.generator_adaptive import generate_episode_fixed_E

    ep = generate_episode_fixed_E(ep_id=0, E=15, T=12, N=50, epsilon=0.1,
                                  probe_bin=Path("solver_bab/build/ssp_probe"))
    # ep : dict compatible avec writers.write_episode / export_to_txt.write_instance
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from .automaton import sample_automaton_params, build_shift_dfa, is_nonempty_at_length
from .cost import generate_cost
from .demand import generate_demand_scenarios, make_bimodal_nominal
from .calibrate_adaptive import calibrate_demand_scale
from .writers import open_dataset, write_episode

log = logging.getLogger(__name__)

DEFAULT_AUTOMATON_RANGES = dict(
    L_min_range=(4, 6), L_max_range=(6, 10),
    m_choices=(2, 3, 4), n_P_choices=(0, 1),
    meal_len_choices=(1, 2), a_min_range=(1, 3),
)


def generate_episode_fixed_E(
    ep_id: int, E: int, T: int, N: int, epsilon: float,
    probe_bin: Path,
    master_seed: int = 42,
    ratio_guess: float = 1.9,
    target_lo: float = 1e4,
    target_hi: float = 1e6,
    probe_time_lim: float = 5.0,
    max_bisect_iters: int = 8,
    max_automaton_retries: int = 15,
    rho_range: tuple[float, float] = (0.3, 0.8),
    sigma_dem_range: tuple[float, float] = (0.1, 0.3),
    automaton_ranges: dict | None = None,
    cost_ranges: dict | None = None,
) -> dict:
    """
    Génère un épisode à E fixe : automate + coûts tirés comme d'habitude,
    demande calibrée par bissection (calibrate_demand_scale) pour tomber
    dans [target_lo, target_hi] nœuds sous Gecode base — pas de dérivation
    de E depuis un tightness_target.

    Redessine l'automate (et la demande) jusqu'à max_automaton_retries fois
    si la bissection ne trouve pas de zone cible (ex: automate structurellement
    trop dur/trop simple, cf. l'effet "états=24" découvert empiriquement).

    Lève RuntimeError si aucun essai n'aboutit.
    """
    aut_ranges = {**DEFAULT_AUTOMATON_RANGES, **(automaton_ranges or {})}
    cost_ranges = cost_ranges or {}

    seq = np.random.SeedSequence([master_seed, ep_id])
    rng_aut, rng_cost, rng_dem = (np.random.default_rng(s) for s in seq.spawn(3))

    last_diag = None
    for attempt in range(max_automaton_retries):
        rho = float(rng_aut.uniform(*rho_range))
        sigma_dem = float(rng_aut.uniform(*sigma_dem_range))

        dfa = None
        for _inner in range(30):
            params = sample_automaton_params(rng_aut, **aut_ranges)
            candidate = build_shift_dfa(params)
            if is_nonempty_at_length(candidate, T):
                dfa = candidate
                break
        if dfa is None:
            log.debug("ep %05d essai %d : DFA vide pour T=%d, re-tirage", ep_id, attempt, T)
            continue

        salary, cost_meta = generate_cost(T, rng_cost, **cost_ranges)

        d_bar = make_bimodal_nominal(T, rng=rng_dem)
        demand_raw = generate_demand_scenarios(d_bar, rho, sigma_dem, N, rng_dem)

        calib = calibrate_demand_scale(
            dfa, salary, demand_raw, E, epsilon, probe_bin,
            ratio_guess=ratio_guess, target_lo=target_lo, target_hi=target_hi,
            probe_time_lim=probe_time_lim, max_iters=max_bisect_iters,
        )
        last_diag = calib

        if calib["status"] == "target":
            return dict(
                dfa=dfa, salary=salary, demand=calib["demand"], d_bar=d_bar,
                E=E, T=T, N=N, epsilon=epsilon,
                meta=dict(
                    ep_id=ep_id, rho=rho, sigma_dem=sigma_dem,
                    automaton_attempt=attempt, scale=calib["scale"],
                    calib_n_iters=calib["n_iters"], calib_history=calib["history"],
                    **cost_meta,
                ),
            )
        log.debug("ep %05d essai %d : bissection sans zone cible (dernier: %s), re-tirage automate",
                  ep_id, attempt, calib["history"][-1] if calib["history"] else None)

    raise RuntimeError(
        f"ep {ep_id}: aucune combinaison automate/échelle trouvée dans la fenêtre cible "
        f"après {max_automaton_retries} essais (E={E}, T={T}, N={N}, eps={epsilon}). "
        f"Dernier diagnostic : {last_diag['history'] if last_diag else None}"
    )


# ── générateur de dataset (plusieurs (E,T,N,ε), mélangeables dans un fichier) ─

def generate_dataset_fixed_E(specs: list[dict], output_dir: Path, probe_bin: Path,
                             master_seed: int = 42, **calib_kwargs) -> int:
    """
    specs : liste de dicts {E, T, N, epsilon, n_reps=1} — un dataset peut
    mélanger plusieurs combinaisons (E,T,N,ε) dans le même fichier HDF5,
    contrairement à generate_dataset (generator.py) qui fixe T/N/ε pour tout
    le dataset et ne fait varier que l'automate/la demande via tightness_target.

    Retourne le nombre d'épisodes effectivement écrits (échecs de calibration
    exclus, cf. RuntimeError de generate_episode_fixed_E).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = output_dir / "dataset.h5"
    manifest_lines = []
    ep_id = 0

    with open_dataset(h5_path, mode="w") as h5f:
        for spec in specs:
            n_reps = spec.get("n_reps", 1)
            for _ in range(n_reps):
                try:
                    ep = generate_episode_fixed_E(
                        ep_id=ep_id, E=spec["E"], T=spec["T"], N=spec["N"],
                        epsilon=spec["epsilon"], probe_bin=probe_bin,
                        master_seed=master_seed, **calib_kwargs,
                    )
                except RuntimeError as e:
                    log.error("ep %05d : %s", ep_id, e)
                    ep_id += 1
                    continue

                write_episode(
                    h5f, ep_id=ep_id, dfa=ep["dfa"], salary=ep["salary"],
                    demand=ep["demand"], d_bar=ep["d_bar"],
                    E=ep["E"], T=ep["T"], N=ep["N"], epsilon=ep["epsilon"],
                    rho=ep["meta"]["rho"], sigma_dem=ep["meta"]["sigma_dem"],
                    meta=ep["meta"],
                )
                manifest_lines.append(
                    f"ep_{ep_id:05d} E={ep['E']} T={ep['T']} N={ep['N']} "
                    f"eps={ep['epsilon']} scale={ep['meta']['scale']:.4f} "
                    f"states={ep['dfa'].num_states} "
                    f"aut_attempts={ep['meta']['automaton_attempt']+1}"
                )
                log.info("ep %05d OK  E=%d T=%d N=%d states=%d scale=%.3f",
                         ep_id, ep["E"], ep["T"], ep["N"],
                         ep["dfa"].num_states, ep["meta"]["scale"])
                ep_id += 1

    (output_dir / "manifest.txt").write_text("\n".join(manifest_lines) + "\n")
    log.info("Dataset généré dans %s : %d/%d épisodes", output_dir, len(manifest_lines), ep_id)
    return len(manifest_lines)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--E", type=int, required=True, help="Nombre d'employés (fixe)")
    p.add_argument("--T", type=int, required=True, help="Horizon (périodes)")
    p.add_argument("--N", type=int, required=True, help="Scénarios in-sample")
    p.add_argument("--eps", type=float, default=0.10, help="Niveau de risque ε")
    p.add_argument("--n-reps", type=int, default=10, help="Nombre d'épisodes à générer")
    p.add_argument("--output", default="dataset_fixed_E", help="Répertoire de sortie")
    p.add_argument("--probe-bin", default="solver_bab/build/ssp_probe")
    p.add_argument("--target-lo", type=float, default=1e4)
    p.add_argument("--target-hi", type=float, default=1e6)
    p.add_argument("--probe-time-lim", type=float, default=5.0)
    p.add_argument("--master-seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")
    spec = dict(E=args.E, T=args.T, N=args.N, epsilon=args.eps, n_reps=args.n_reps)
    n_ok = generate_dataset_fixed_E(
        [spec], Path(args.output), Path(args.probe_bin),
        master_seed=args.master_seed,
        target_lo=args.target_lo, target_hi=args.target_hi,
        probe_time_lim=args.probe_time_lim,
    )
    print(f"-> {args.output}/dataset.h5  ({n_ok}/{args.n_reps} épisodes)")


if __name__ == "__main__":
    main()
