"""
collect_ccpo_samples.py — Collecte des nœuds B&B (REF, CP pur) servant à entraîner le GNN CCPO.

Grille : n ∈ {6, 8, 10} × N ∈ {20, 50, 100}, m = Q = round(2,5 n), k = max(3, n // 2), L = 2, U = 8,
γ = 1,25, ε = 0,1. Pour chaque cellule, on tire des instances (ccpo_stochastic, graine déterministe dérivée
de --seed_base) jusqu'à en avoir `target_optimal` résolues à l'optimum par REF en 20 s
(10 000 pour n = 6 et 8 ; 3 000 / 3 000 / 1 500 pour n = 10, N = 20 / 50 / 100, cellule la plus coûteuse).

Pour chaque instance : bab_ccpo_collect écrit ses nœuds dans un fichier temporaire, dont on garde la racine
et au plus `--cap` nœuds non-racine, répartis à peu près également entre les valeurs de n_free (les nœuds
profonds, très nombreux, sont aussi les plus souvent infaisables). Seules ces lignes sont ajoutées à
<out_dir>/nodes_n{n}_N{N}.txt ; l'instance (ligne d'en-tête, optimum, drapeaux de scénarios) va dans
<out_dir>/instances_n{n}_N{N}.txt.

Reprise : un fichier de progression par cellule (<out_dir>/_progress_n{n}_N{N}.json) garde la prochaine
tentative ; un job relancé reprend là où le précédent s'est arrêté, sans jamais réutiliser un instance_id.

Étapes suivantes : curate_feasible_examples.py, puis recompute_presolve_flags.py (cf. gnn_ccpo/README.md).

Usage (depuis la racine du dépôt) :
    python3 collect/collect_ccpo_samples.py --out_dir <dir> --cap 300
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ccpo_stochastic.instance import draw_mother_instance  # noqa: E402
from ccpo_stochastic.scenarios import (KAPPA_DEFAULT, NOISE_BETA_DEFAULT, NOISE_MARKET_DEFAULT,  # noqa: E402
                                       NOISE_PSI_DEFAULT, P_SHOCK_DEFAULT, generate_scenarios)
from ccpo_stochastic.discretize import discretize_instance  # noqa: E402
from ccpo_stochastic.writers import write_episode  # noqa: E402

BUILD_DIR = Path(__import__("os").environ.get("BUILD_DIR", REPO_ROOT / "build"))
COLLECTOR_BIN = BUILD_DIR / "solver_bab" / "bab_ccpo_collect"

# Coefficients de perturbation effectivement utilisés (défauts de ccpo_stochastic), écrits dans meta.txt.
NOISE_PARAMS = {"noise_beta": NOISE_BETA_DEFAULT, "noise_psi": NOISE_PSI_DEFAULT,
                "noise_market": NOISE_MARKET_DEFAULT, "p_shock": P_SHOCK_DEFAULT, "kappa": KAPPA_DEFAULT}

GRID = [
    (6, 20), (6, 50), (6, 100),
    (8, 20), (8, 50), (8, 100),
    (10, 20), (10, 50), (10, 100),
]

CELL_CONFIG = {
    (6, 20):   dict(target_optimal=10000, time_lim=20.0, max_attempts=25000),
    (6, 50):   dict(target_optimal=10000, time_lim=20.0, max_attempts=20000),
    (6, 100):  dict(target_optimal=10000, time_lim=20.0, max_attempts=30000),
    (8, 20):   dict(target_optimal=10000, time_lim=20.0, max_attempts=17000),
    (8, 50):   dict(target_optimal=10000, time_lim=20.0, max_attempts=14000),
    (8, 100):  dict(target_optimal=10000, time_lim=20.0, max_attempts=14000),
    (10, 20):  dict(target_optimal=3000,  time_lim=20.0, max_attempts=3500),
    (10, 50):  dict(target_optimal=3000,  time_lim=20.0, max_attempts=4200),
    (10, 100): dict(target_optimal=1500,  time_lim=20.0, max_attempts=2500),
}


CAP_PER_INSTANCE = 300  # nœuds non-racine gardés par instance
GAMMA = 1.25
EPSILON = 0.1
L, U = 2, 8


def q_k_for(n: int) -> tuple[int, int]:
    # Ratios de l'instance de référence n=8, Q=20, k=4 (Q/n = 2,5, k/n = 0,5) ; avec Q = 2n,
    # la part d'instances infaisables était trop élevée.
    return round(2.5 * n), max(3, n // 2)


def _n_free(line: str) -> int:
    """Nombre de -1 dans q_mask (champs 4+) -- cout negligeable, pas de
    reconstruction complete necessaire juste pour stratifier."""
    return line.rstrip("\n").split(",")[3:].count("-1")


def _stratified_sample_by_nfree(lines: list[str], cap: int, rng: random.Random) -> list[str]:
    """Repartit le quota `cap` A PEU PRES EGALEMENT entre les valeurs de
    n_free presentes (round-robin sur les buckets, pas un tirage uniforme
    brut) : un tirage uniforme sur tous les noeuds
    surrepresente massivement les noeuds profonds (n_free petit), qui sont
    aussi les plus souvent PROUVES INFAISABLES (mesure empirique : 98% a
    n_free=2, 74% a n_free=6).
    Stratifier par n_free evite que ces noeuds peu utiles noient les noeuds
    peu profonds (plus rares dans l'arbre, mais plus souvent faisables et
    plus informatifs pour le GNN)."""
    if not lines:
        return []
    buckets: dict[int, list[str]] = {}
    for l in lines:
        buckets.setdefault(_n_free(l), []).append(l)
    for b in buckets.values():
        rng.shuffle(b)

    kept: list[str] = []
    remaining = cap
    active = list(buckets.keys())
    while remaining > 0 and active:
        rng.shuffle(active)
        for nf in list(active):
            if remaining <= 0:
                break
            if buckets[nf]:
                kept.append(buckets[nf].pop())
                remaining -= 1
            if not buckets[nf]:
                active.remove(nf)
    return kept


def _sample_instance_lines(lines: list[str], cap: int, rng: random.Random) -> list[str]:
    """Racine (depth==0, 2e champ) toujours gardee + au plus `cap` non-racine,
    stratifies par n_free (cf. _stratified_sample_by_nfree) -- appliquee EN
    UNE PASSE juste apres la collecte d'UNE instance (jamais ecrite sur
    disque non-echantillonnee)."""
    if not lines:
        return []
    roots = [l for l in lines if l.split(",", 2)[1] == "0"]
    non_roots = [l for l in lines if l.split(",", 2)[1] != "0"]
    kept_non_root = _stratified_sample_by_nfree(non_roots, cap, rng)
    return roots + kept_non_root


def collect_one(n: int, N: int, instance_id: int, seed: int, tmp_dir: Path,
                 time_lim: float, max_depth: int, cap: int, rng: random.Random,
                 instances_out: Path, nodes_out: Path) -> str:
    Q, k = q_k_for(n)
    rng_master = np.random.default_rng(seed)
    mother = draw_mother_instance(rng_master, n=n)
    sc = generate_scenarios(np.random.default_rng(seed + 1), mother, N=N)
    try:
        di = discretize_instance(mother, sc, Q=Q, k=k, L=L, U=U, gamma=GAMMA)
    except (ValueError, RuntimeError):
        return "skip_structural"

    write_episode(
        tmp_dir, instance_id, mother=mother, discretized=di, epsilon=EPSILON,
        seed_master=seed, n_episodes=1,
        noise_params=NOISE_PARAMS,
    )
    ep_dir = tmp_dir / f"ep_{instance_id:06d}"

    # Nœuds de CETTE instance dans un fichier temporaire dedie -- jamais
    # accumules en brut dans nodes_out (cf. docstring du module).
    nodes_tmp = tmp_dir / f"nodes_tmp_{instance_id:06d}.txt"
    if nodes_tmp.exists():
        nodes_tmp.unlink()

    r = subprocess.run(
        [str(COLLECTOR_BIN), str(ep_dir), str(instance_id), str(time_lim),
         str(max_depth), str(instances_out), str(nodes_tmp)],
        capture_output=True, text=True,
    )
    shutil.rmtree(ep_dir, ignore_errors=True)

    status = f"error: {r.stderr.strip()[:200]}" if r.returncode != 0 else (
        r.stdout.strip().split(",")[2] if r.stdout.strip() else "no_output")

    if nodes_tmp.exists():
        with open(nodes_tmp) as f:
            raw_lines = f.readlines()
        nodes_tmp.unlink()
        if raw_lines:
            kept = _sample_instance_lines(raw_lines, cap, rng)
            with open(nodes_out, "a") as f:
                f.writelines(kept)

    return status


def _progress_path(out_dir: Path, n: int, N: int) -> Path:
    return out_dir / f"_progress_n{n}_N{N}.json"


def _load_progress(out_dir: Path, n: int, N: int, instances_out: Path) -> dict:
    """next_attempt vient du JSON (source de verite pour l'unicite des
    instance_id) ; n_optimal est RECOMPTE depuis instances_out (source de
    verite pour la progression reelle -- robuste a une coupure en cours
    d'ecriture du JSON)."""
    path = _progress_path(out_dir, n, N)
    next_attempt = 0
    if path.exists():
        try:
            next_attempt = json.loads(path.read_text())["next_attempt"]
        except (json.JSONDecodeError, KeyError):
            next_attempt = 0
    n_optimal = 0
    if instances_out.exists():
        with open(instances_out) as f:
            n_optimal = sum(1 for _ in f)
    return {"next_attempt": next_attempt, "n_optimal": n_optimal}


def _save_progress(out_dir: Path, n: int, N: int, next_attempt: int) -> None:
    path = _progress_path(out_dir, n, N)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"next_attempt": next_attempt}))
    tmp.replace(path)  # ecriture atomique


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_depth", type=int, default=-1)
    ap.add_argument("--seed_base", type=int, default=2_000_000)
    ap.add_argument("--cap", type=int, default=CAP_PER_INSTANCE)
    ap.add_argument("--rng_seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_tmp_episodes"
    tmp_dir.mkdir(exist_ok=True)
    rng = random.Random(args.rng_seed)

    for n, N in GRID:
        cfg = CELL_CONFIG[(n, N)]
        instances_out = out_dir / f"instances_n{n}_N{N}.txt"
        nodes_out = out_dir / f"nodes_n{n}_N{N}.txt"

        progress = _load_progress(out_dir, n, N, instances_out)
        attempt = progress["next_attempt"]
        n_optimal = progress["n_optimal"]
        if n_optimal >= cfg["target_optimal"]:
            print(f"=== n={n} N={N} DEJA COMPLET -- {n_optimal}/{cfg['target_optimal']} "
                  f"(reprise, {attempt} tentatives deja faites) ===", flush=True)
            continue
        print(f"=== n={n} N={N} reprise a l'attempt {attempt}, "
              f"{n_optimal}/{cfg['target_optimal']} OPTIMAL deja acquis ===", flush=True)

        status_counts: dict[str, int] = {}
        attempt_at_start = attempt
        while n_optimal < cfg["target_optimal"] and attempt < cfg["max_attempts"]:
            seed = args.seed_base + hash((n, N, attempt)) % 1_000_000_000
            status = collect_one(n, N, attempt, seed, tmp_dir, cfg["time_lim"], args.max_depth,
                                  args.cap, rng, instances_out, nodes_out)
            status_counts[status] = status_counts.get(status, 0) + 1
            if status == "OPTIMAL":
                n_optimal += 1
            attempt += 1
            if (attempt - attempt_at_start) % 20 == 0:
                _save_progress(out_dir, n, N, attempt)
            if (attempt - attempt_at_start) % 200 == 0:
                print(f"[n={n} N={N}] tentative {attempt}/{cfg['max_attempts']}, "
                      f"OPTIMAL {n_optimal}/{cfg['target_optimal']} -- {status_counts}", flush=True)
        _save_progress(out_dir, n, N, attempt)
        print(f"=== n={n} N={N} TERMINE (ou max_attempts atteint) -- {n_optimal} OPTIMAL en "
              f"{attempt} tentatives -- {status_counts} ===", flush=True)

    if all(_load_progress(out_dir, n, N, out_dir / f"instances_n{n}_N{N}.txt")["n_optimal"]
           >= CELL_CONFIG[(n, N)]["target_optimal"] for n, N in GRID):
        print("=== Collecte v2 termine (toutes cellules completes) ===", flush=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
