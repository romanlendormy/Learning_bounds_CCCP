"""
writers.py — Écriture texte d'un épisode CCPO.

Un dossier par épisode (`ep_XXXXXX/instance.txt`, `scenarios.txt`, `meta.txt`), plus un
`manifest.txt` global à la racine du dataset. Format OR-Library : espaces comme
séparateurs, indices en base 1, tout entier sauf `meta.txt` (clé = valeur) et le
`gamma`/`epsilon` du manifest (informatifs, la valeur qui compte pour le solveur est
`epsilon_num`/`epsilon_den` dans `instance.txt`).

`--gzip` (`gzip_scenarios=True`) : le module `gzip` standard écrit le mtime dans
l'en-tête par défaut, ce qui casserait le déterminisme octet-identique.
On force donc `mtime=0` explicitement.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

from generators.ccpo.discretize import DiscretizedInstance
from generators.ccpo.instance import MotherInstance
from generators.ccpo.txt_format import GENERATOR_VERSION, epsilon_to_fraction, get_git_commit


def _write_text(path: Path, text: str, gzip_output: bool) -> None:
    data = text.encode("utf-8")
    if gzip_output:
        with open(path, "wb") as f, gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as gz:
            gz.write(data)
    else:
        path.write_bytes(data)


def instance_txt_content(
    n: int,
    N: int,
    k: int,
    Q: int,
    L: int,
    U: int,
    eps_num: int,
    eps_den: int,
    B: int,
    c_int: np.ndarray,
) -> str:
    lines = [
        f"{n} {N} {k} {Q} {L} {U} {eps_num} {eps_den}",
        f"{B}",
        " ".join(str(int(v)) for v in c_int),
    ]
    return "\n".join(lines) + "\n"


def scenarios_txt_content(Sigma_int: np.ndarray, psi_int: np.ndarray) -> str:
    N, n, _ = Sigma_int.shape
    rows, cols = np.triu_indices(n)  # triangle supérieur, diagonale incluse
    rows_l, cols_l = rows.tolist(), cols.tolist()

    out_lines: list[str] = []
    for j in range(N):
        out_lines.append(str(j + 1))  # indice de scénario, base 1
        out_lines.append(" ".join(str(int(v)) for v in psi_int[j]))
        vals = Sigma_int[j][rows, cols].tolist()
        out_lines.extend(
            f"{r + 1} {c + 1} {v}" for r, c, v in zip(rows_l, cols_l, vals)
        )
    return "\n".join(out_lines) + "\n"


def meta_txt_content(meta: dict) -> str:
    return "".join(f"{key} = {value}\n" for key, value in meta.items())


def _append_manifest_line(
    manifest_path: Path,
    ep_id: int,
    n: int,
    N: int,
    k: int,
    Q: int,
    gamma: float,
    epsilon: float,
    seed_master: int,
) -> None:
    line = f"{ep_id} {n} {N} {k} {Q} {gamma:.6g} {epsilon:.6g} {seed_master}\n"
    with open(manifest_path, "a") as f:
        f.write(line)


def write_episode(
    root_dir: Path,
    ep_id: int,
    *,
    mother: MotherInstance,
    discretized: DiscretizedInstance,
    epsilon: float,
    seed_master: int,
    n_episodes: int,
    noise_params: dict,
    capm_alignment: float = 0.0,
    gzip_scenarios: bool = False,
    include_git_commit: bool = True,
) -> Path:
    """
    Écrit un épisode complet : `ep_XXXXXX/{instance,scenarios,meta}.txt` et une ligne
    dans `manifest.txt` à la racine de `root_dir`.

    `noise_params` : dict des coefficients de perturbation effectivement utilisés
    (noise_beta, noise_psi, noise_market, p_shock, kappa) — écrits tels quels dans
    meta.txt pour traçabilité.
    """
    root_dir = Path(root_dir)
    ep_dir = root_dir / f"ep_{ep_id:06d}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    eps_num, eps_den = epsilon_to_fraction(epsilon)

    instance_text = instance_txt_content(
        n=discretized.n,
        N=discretized.N,
        k=discretized.k,
        Q=discretized.Q,
        L=discretized.L,
        U=discretized.U,
        eps_num=eps_num,
        eps_den=eps_den,
        B=discretized.B,
        c_int=discretized.c_int,
    )
    (ep_dir / "instance.txt").write_text(instance_text)

    scen_text = scenarios_txt_content(discretized.Sigma_int, discretized.psi_int)
    scen_path = ep_dir / ("scenarios.txt.gz" if gzip_scenarios else "scenarios.txt")
    _write_text(scen_path, scen_text, gzip_output=gzip_scenarios)

    meta = {
        "generator_version": GENERATOR_VERSION,
        "git_commit": get_git_commit() if include_git_commit else "not_recorded",
        "ep_id": ep_id,
        "n_episodes": n_episodes,
        "seed_master": seed_master,
        "sigma_f": mother.sigma_f,
        "S": discretized.S,
        "S_c": discretized.S_c,
        "gamma": discretized.gamma,
        "capm_alignment": capm_alignment,
        **noise_params,
        "sigma_bar": discretized.sigma_bar,
        "rho_bar": mother.rho_bar,
    }
    (ep_dir / "meta.txt").write_text(meta_txt_content(meta))

    _append_manifest_line(
        root_dir / "manifest.txt",
        ep_id=ep_id,
        n=discretized.n,
        N=discretized.N,
        k=discretized.k,
        Q=discretized.Q,
        gamma=discretized.gamma,
        epsilon=epsilon,
        seed_master=seed_master,
    )

    return ep_dir
