"""
generator.py — Génération en lot du dataset SSP chance-constrained.

Point d'entrée :
    python -m ssp_stochastic.generator --config config_default.yaml --output dataset/

Structure de sortie :
    dataset/
        config.yaml
        seeds.txt
        manifest.txt
        episodes/
            ep_00000/ ...
            ep_00001/ ...
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import yaml

from .automaton import AutomatonParams, build_shift_dfa, is_nonempty_at_length, sample_automaton_params
from .calibrate import calibrate_E, adjust_E_for_feasibility
from .cost import generate_cost
from .demand import generate_demand_scenarios, make_bimodal_nominal
from .writers import open_dataset, write_episode

log = logging.getLogger(__name__)


# ── config par défaut ─────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "n_episodes": 10,
    "T": 16,
    "N": 50,
    "K_oos": 2000,
    "epsilon": 0.1,
    "tightness_target": 1.1,
    "rho_range": [0.3, 0.8],
    "sigma_dem_range": [0.1, 0.3],
    "L_min_range": [4, 6],
    "L_max_range": [6, 10],
    "m_choices": [2, 3, 4],
    "n_P_choices": [0, 1],
    "meal_len_choices": [1, 2],
    "a_min_range": [1, 3],
    "b0_range": [1.0, 2.0],
    "b_soir_range": [0.2, 0.8],
    "b_nuit_range": [0.5, 1.5],
    "master_seed": 42,
    "max_automaton_retries": 20,
}


# ── génération d'un épisode ───────────────────────────────────────────────────

def generate_episode(
    ep_id: int,
    T: int,
    N: int,
    epsilon: float,
    tightness_target: float,
    rho: float,
    sigma_dem: float,
    aut_params: AutomatonParams,
    cost_params_extra: dict,
    seeds: dict,
    cfg: dict,
) -> dict:
    """
    Génère un seul épisode.  Retourne un dict avec les données et métadonnées.

    Le calcul est entièrement déterministe donné les seeds.
    """
    # Sous-RNGs par module (dérivés de la graine de l'épisode)
    # spawn(3) → [rng_dem, rng_oos, rng_cost] — ordre stable pour regenerate_oos
    seq = np.random.SeedSequence(seeds["episode"])
    rng_dem, _rng_oos, rng_cost = (np.random.default_rng(s)
                                   for s in seq.spawn(3))

    # ── Demande ───────────────────────────────────────────────────────────────
    d_bar = make_bimodal_nominal(T, rng=rng_dem)
    demand = generate_demand_scenarios(d_bar, rho, sigma_dem, N, rng_dem)
    # demand_oos n'est pas généré ici — utiliser regenerate_oos(inst, K)

    # ── Coûts ─────────────────────────────────────────────────────────────────
    salary, cost_meta = generate_cost(
        T, rng_cost,
        b0_range=tuple(cfg.get("b0_range", [1.0, 2.0])),
        b_soir_range=tuple(cfg.get("b_soir_range", [0.2, 0.8])),
        b_nuit_range=tuple(cfg.get("b_nuit_range", [0.5, 1.5])),
    )

    # ── Automate ──────────────────────────────────────────────────────────────
    dfa = build_shift_dfa(aut_params)
    if not is_nonempty_at_length(dfa, T):
        raise RuntimeError(f"ep {ep_id}: DFA vide pour T={T}, params={aut_params}")

    # ── Calibration E ─────────────────────────────────────────────────────────
    E, calib_meta = calibrate_E(demand, epsilon, tightness_target, dfa, T)
    E, adj_status = adjust_E_for_feasibility(E, demand, epsilon, dfa, T)
    if adj_status != "ok":
        log.warning("ep %05d : %s", ep_id, adj_status)

    # ── Métadonnées ───────────────────────────────────────────────────────────
    meta: dict = {
        "ep_id": ep_id,
        "T": T,
        "N": N,
        "K_oos": cfg["K_oos"],
        "epsilon": epsilon,
        "rho": rho,
        "sigma_dem": sigma_dem,
        # automate
        "L_min": aut_params.L_min,
        "L_max": aut_params.L_max,
        "m": aut_params.m,
        "n_P": aut_params.n_P,
        "meal_len": aut_params.meal_len,
        "a_min": aut_params.a_min,
        "b_max": aut_params.b_max,
        "win_start": aut_params.win_start,
        "win_end": aut_params.win_end,
        "dfa_num_states": dfa.num_states,
        "dfa_num_transitions": len(dfa.delta),
        "dfa_density": len(dfa.delta) / max(1, dfa.num_states * 4),
        # coûts
        **cost_meta,
        # calibration
        **calib_meta,
        # graines
        "seed_episode": seeds["episode"],
        # axes de difficulté (manifest)
        "axis_E": E,
        "axis_epsilon": epsilon,
        "axis_ratio_tightness": calib_meta["ratio_tightness"],
        "axis_rho": rho,
        "axis_sigma_dem": sigma_dem,
        "axis_N": N,
        "axis_T": T,
        "axis_dfa_states": dfa.num_states,
    }

    return {
        "dfa": dfa,
        "salary": salary,
        "demand": demand,
        "d_bar": d_bar,
        "E": E,
        "meta": meta,
    }


# ── générateur principal ──────────────────────────────────────────────────────

def generate_dataset(cfg: dict, output_dir: Path) -> None:
    """
    Génère n_episodes épisodes et écrit tout le dataset.

    Paramètres
    ----------
    cfg        : configuration (voir DEFAULT_CONFIG pour les clés)
    output_dir : répertoire de sortie (créé si absent)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    n_ep      = cfg["n_episodes"]
    T         = cfg["T"]
    N         = cfg["N"]
    K_oos     = cfg["K_oos"]
    epsilon   = cfg["epsilon"]
    tightness = cfg["tightness_target"]
    master    = cfg["master_seed"]

    # Dérivation des graines : une sous-séquence par épisode
    master_seq = np.random.SeedSequence(master)
    ep_seqs    = master_seq.spawn(n_ep)

    h5_path = output_dir / "dataset.h5"
    manifest_lines = []
    seeds_lines    = [f"master {master}"]

    with open_dataset(h5_path, mode="w") as h5f:
        for ep_id, ep_seq in enumerate(ep_seqs):
            # Graine entière unique par épisode (pour log)
            rng_master_ep = np.random.default_rng(ep_seq)
            ep_int_seed   = int(rng_master_ep.integers(0, 2**62))

            sub_seqs = ep_seq.spawn(4)
            seeds = {
                "episode":   ep_int_seed,
                "automaton": sub_seqs[0],
                "cost":      sub_seqs[1],
                "demand":    sub_seqs[2],
                "oos":       sub_seqs[3],
            }
            seeds_lines.append(f"ep_{ep_id:05d} ep_seed={ep_int_seed}")

            # Paramètres aléatoires de l'épisode
            rng_ep    = np.random.default_rng(seeds["automaton"])
            rho       = float(rng_ep.uniform(*cfg["rho_range"]))
            sigma_dem = float(rng_ep.uniform(*cfg["sigma_dem_range"]))

            # Automate : re-tirer si langage vide pour ce T
            aut_params = None
            for attempt in range(cfg.get("max_automaton_retries", 20)):
                p = sample_automaton_params(
                    rng_ep,
                    L_min_range=tuple(cfg["L_min_range"]),
                    L_max_range=tuple(cfg["L_max_range"]),
                    m_choices=tuple(cfg["m_choices"]),
                    n_P_choices=tuple(cfg["n_P_choices"]),
                    meal_len_choices=tuple(cfg["meal_len_choices"]),
                    a_min_range=tuple(cfg["a_min_range"]),
                )
                dfa_test = build_shift_dfa(p)
                if is_nonempty_at_length(dfa_test, T):
                    aut_params = p
                    break
                log.debug("ep %05d attempt %d: DFA vide, re-tirage", ep_id, attempt)
            if aut_params is None:
                log.error("ep %05d : impossible de trouver un automate non-vide après %d essais",
                          ep_id, cfg.get("max_automaton_retries", 20))
                continue

            try:
                ep_data = generate_episode(
                    ep_id=ep_id, T=T, N=N,
                    epsilon=epsilon, tightness_target=tightness,
                    rho=rho, sigma_dem=sigma_dem,
                    aut_params=aut_params,
                    cost_params_extra={},
                    seeds=seeds, cfg=cfg,
                )
            except Exception as e:
                log.error("ep %05d : erreur de génération : %s", ep_id, e)
                continue

            # Écriture dans le fichier HDF5 (demand_oos non stocké, régénérable)
            write_episode(
                h5f,
                ep_id=ep_id,
                dfa=ep_data["dfa"],
                salary=ep_data["salary"],
                demand=ep_data["demand"],
                d_bar=ep_data["d_bar"],
                E=ep_data["E"], T=T, N=N, epsilon=epsilon,
                rho=rho, sigma_dem=sigma_dem,
                meta=ep_data["meta"],
            )

            m = ep_data["meta"]
            manifest_lines.append(
                f"ep_{ep_id:05d} "
                f"E={m['axis_E']} T={T} N={N} epsilon={epsilon} "
                f"rho={rho:.3f} sigma_dem={sigma_dem:.3f} "
                f"tightness={m['axis_ratio_tightness']:.3f} "
                f"dfa_states={m['axis_dfa_states']}"
            )
            log.info("ep %05d OK  E=%d  DFA=%d states  tightness=%.2f",
                     ep_id, ep_data["E"], ep_data["dfa"].num_states,
                     m["axis_ratio_tightness"])

    # Fichiers compagnons (indépendants du HDF5)
    (output_dir / "manifest.txt").write_text(
        "# ep_id E T N epsilon rho sigma_dem tightness dfa_states\n"
        + "\n".join(manifest_lines) + "\n"
    )
    (output_dir / "seeds.txt").write_text("\n".join(seeds_lines) + "\n")
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    log.info("Dataset généré dans %s : %d épisodes  (%s)",
             output_dir, len(manifest_lines), h5_path.name)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description="Générateur SSP chance-constrained")
    parser.add_argument("--config", type=str, default=None,
                        help="Fichier config YAML (défaut : config embarquée)")
    parser.add_argument("--output", type=str, default="dataset",
                        help="Répertoire de sortie")
    parser.add_argument("--n", type=int, default=None,
                        help="Nombre d'épisodes (override config)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Graine maître (override config)")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    cfg = dict(DEFAULT_CONFIG)
    if args.config:
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f))
    if args.n is not None:
        cfg["n_episodes"] = args.n
    if args.seed is not None:
        cfg["master_seed"] = args.seed

    generate_dataset(cfg, Path(args.output))


if __name__ == "__main__":
    main()
