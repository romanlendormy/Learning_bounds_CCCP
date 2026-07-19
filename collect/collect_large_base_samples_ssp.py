"""
collect_large_base_samples_ssp.py — Collecte CC-SSP (shift scheduling
chance-constrained stochastique) avec pruning CP pur + sampling 10 %.

Analogue SSP de collect/collect_large_base_samples.py (MKP) : génère une
instance à (E, T, N, epsilon) fixés, résout un B&B Gecode CP pur (pas de
borne Lagrangienne — cf. solver_bab/ssp/collect_ssp_base.{hpp,cpp}), puis ne
conserve que 10 % des nœuds collectés (+ la racine, toujours gardée), tirés
uniformément au hasard.

Génération : ssp_stochastic.generator_adaptive.generate_episode_fixed_E
(E, T, N choisis librement — la difficulté est calibrée par bissection sur
l'échelle de la demande, via une sonde ssp_probe, pour tomber dans une
fenêtre de nœuds cible [target_lo, target_hi] sous Gecode base). C'est
l'approche qui a produit les seules données de collecte SSP existant à ce
jour (cf. ssp_stochastic/sweep_bound_efficacy.py).

Le format de sortie est scrupuleusement analogue au format C du MKP :
  inst_id, depth, curr_cost, x_mask[E*T], z_mask[N]  (3 + E*T + N champs)
  x_mask : 0=O,1=P,2=R,3=W fixé ; 4=libre.  z_mask : 0/1 fixé, 2=libre.

Les résultats s'accumulent dans :
  <outdir>_base_s10/instances_E{E}_T{T}_N{N}.txt
  <outdir>_base_s10/nodes_E{E}_T{T}_N{N}.txt
  <outdir>_base_s10/seeds_done_E{E}_T{T}_N{N}.txt

Usage :
  python collect_large_base_samples_ssp.py \\
      --E 15 --T 12 --N 50 --n_instances 200 \\
      --epsilon 0.1 --max_depth 10 \\
      --binary solver_bab/build/bab_ssp_gecode_base_collect \\
      --probe_bin solver_bab/build/ssp_probe \\
      --outdir data/train_ssp

Reprendre une collecte interrompue :
  python collect_large_base_samples_ssp.py ... --seed_start 200
"""

import argparse
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "solver_generic" / "ssp"))

from ssp_stochastic.generator_adaptive import generate_episode_fixed_E  # noqa: E402
from export_to_txt import write_instance  # noqa: E402


SAMPLE_RATE = 0.10   # fraction de nœuds conservés par B&B


# ─────────────────────────────────────────────────────────────────────────────
# Helpers fichiers — identiques à collect_large_base_samples.py (MKP) : le
# format nœuds SSP a aussi "depth" en champ d'index 1 (cf. write_node_line_ssp
# côté C++), donc _sample_lines s'applique tel quel, sans modification.
# ─────────────────────────────────────────────────────────────────────────────

def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def _append_new_lines(src: Path, dst: Path, lines_before: int) -> int:
    """Copie dans dst les lignes ajoutées à src depuis lines_before."""
    if not src.exists():
        return 0
    with open(src) as f:
        all_lines = f.readlines()
    new_lines = all_lines[lines_before:]
    if not new_lines:
        return 0
    with open(dst, "a") as f:
        f.writelines(new_lines)
    return len(new_lines)


def _sample_lines(lines: list[str], rate: float, rng: random.Random) -> list[str]:
    """
    Depuis une liste de lignes (un B&B complet) :
      - Conserve toujours les nœuds racines (depth == 0, champ index 1).
      - Tire uniformément `rate` des nœuds non-racines (min 0).
    Retourne la liste des lignes conservées.
    """
    roots     = [l for l in lines if int(l.split(",", 3)[1]) == 0]
    non_roots = [l for l in lines if int(l.split(",", 3)[1]) != 0]
    k = max(0, round(len(non_roots) * rate))
    sampled_non_roots = rng.sample(non_roots, min(k, len(non_roots))) if k > 0 else []
    return roots + sampled_non_roots


# ─────────────────────────────────────────────────────────────────────────────
# Génération + export d'une instance → répertoire temporaire (4 fichiers texte)
# ─────────────────────────────────────────────────────────────────────────────

def generate_ssp_instance_dir(seed: int, args) -> tuple[str | None, dict | None]:
    """
    Génère un épisode SSP (E,T,N,epsilon fixés) calibré par bissection, et
    l'exporte dans un répertoire temporaire au format 4-fichiers attendu par
    ssp_loader.hpp. Retourne (répertoire, ep) ou (None, None) si la
    calibration échoue (RuntimeError propagée par l'appelant sinon).
    """
    ep = generate_episode_fixed_E(
        ep_id=seed, E=args.E, T=args.T, N=args.N, epsilon=args.epsilon,
        probe_bin=Path(args.probe_bin), master_seed=args.master_seed,
        ratio_guess=args.ratio_guess,
        target_lo=args.target_lo, target_hi=args.target_hi,
        probe_time_lim=args.probe_time_lim,
        max_bisect_iters=args.max_bisect_iters,
        max_automaton_retries=args.max_automaton_retries,
    )
    tmp_dir = tempfile.mkdtemp(prefix="ssp_ccp_tmp_")
    write_instance(ep, Path(tmp_dir))
    return tmp_dir, ep


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collecte GNN CC-SSP (CP pur) avec sampling 10 % par B&B.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--E",           type=int,   required=True,
                        help="Nombre d'employés (fixe, calibration porte sur la demande)")
    parser.add_argument("--T",           type=int,   required=True, help="Horizon (périodes)")
    parser.add_argument("--N",           type=int,   required=True, help="Scénarios in-sample")
    parser.add_argument("--epsilon",     type=float, default=0.10)
    parser.add_argument("--n_instances", type=int,   default=100)
    parser.add_argument("--max_depth",   type=int,   default=10)
    parser.add_argument("--seed_start",  type=int,   default=0)
    parser.add_argument("--sample_rate", type=float, default=SAMPLE_RATE,
                        help=f"Fraction de nœuds à conserver par B&B (défaut {SAMPLE_RATE})")
    parser.add_argument("--rng_seed",    type=int,   default=0,
                        help="Graine pour le sampling (défaut 0)")
    parser.add_argument("--master_seed", type=int,   default=42,
                        help="Graine maître pour generate_episode_fixed_E (défaut 42)")
    parser.add_argument("--time_limit",  type=float, default=-1.0,
                        help="Limite en secondes par B&B de collecte (-1 = illimité)")
    parser.add_argument("--ratio_guess",          type=float, default=1.9)
    parser.add_argument("--target_lo",            type=float, default=1e4,
                        help="Borne basse de la fenêtre de difficulté (nœuds, Gecode base)")
    parser.add_argument("--target_hi",            type=float, default=1e6,
                        help="Borne haute de la fenêtre de difficulté (nœuds, Gecode base)")
    parser.add_argument("--probe_time_lim",       type=float, default=5.0)
    parser.add_argument("--max_bisect_iters",     type=int,   default=8)
    parser.add_argument("--max_automaton_retries", type=int,  default=15)
    parser.add_argument("--binary",      type=str,
                        default="solver_bab/build/bab_ssp_gecode_base_collect")
    parser.add_argument("--probe_bin",   type=str,
                        default="solver_bab/build/ssp_probe")
    parser.add_argument("--outdir",      type=str,   default="data/train_ssp")
    parser.add_argument("--verbose",     action="store_true")

    args = parser.parse_args()

    if not os.path.isfile(args.binary):
        print(f"Erreur : binaire introuvable : {args.binary}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.probe_bin):
        print(f"Erreur : binaire introuvable : {args.probe_bin}", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.rng_seed)
    tag = f"E{args.E}_T{args.T}_N{args.N}"

    # Dossier de staging (fichiers génériques écrits par le binaire C++, non
    # échantillonnés) — le binaire appende _base lui-même, comme pour MKP.
    staging_dir = Path(f"{args.outdir}_base")
    staging_dir.mkdir(parents=True, exist_ok=True)
    outdir_staging = args.outdir  # binaire écrit dans {outdir}_base/

    # Dossier de sortie final (avec sampling)
    outdir = Path(f"{args.outdir}_base_s10")
    outdir.mkdir(parents=True, exist_ok=True)

    inst_generic  = staging_dir / f"instances_{tag}.txt"
    nodes_generic = staging_dir / f"nodes_{tag}.txt"

    inst_m    = outdir / f"instances_{tag}.txt"
    nodes_m   = outdir / f"nodes_{tag}.txt"
    prog_file = outdir / f"seeds_done_{tag}.txt"

    done_seeds: set[int] = set()
    if prog_file.exists():
        done_seeds = {int(l.strip()) for l in prog_file.read_text().splitlines() if l.strip()}
        if done_seeds:
            print(f"[reprise] {len(done_seeds)} seeds déjà traitées")

    # --n_instances est un TOTAL cible (pas "combien en ajouter") : sur
    # reprise, on ne complète que ce qu'il manque pour l'atteindre.
    remaining = max(0, args.n_instances - len(done_seeds))
    todo: list[int] = []
    seed = args.seed_start
    while len(todo) < remaining:
        if seed not in done_seeds:
            todo.append(seed)
        seed += 1

    print("=" * 60)
    print(f"Collecte CC-SSP (CP pur, sampling {args.sample_rate*100:.0f}%) :"
          f" E={args.E}  T={args.T}  N={args.N}  epsilon={args.epsilon}")
    print(f"  cible     : {args.n_instances} instances au total"
          + (f"  (seeds {todo[0]}..{todo[-1]})" if todo else ""))
    print(f"  à traiter : {len(todo)}  ({len(done_seeds)} déjà faites)")
    print(f"  max_depth : {args.max_depth}")
    print(f"  fenêtre difficulté (calibration) : [{args.target_lo:.0f}, {args.target_hi:.0f}] nœuds")
    print(f"  sortie    : {nodes_m.name}  (dans {outdir})")
    print("=" * 60)

    if not todo:
        print("Rien à faire : cible déjà atteinte.")
        return

    solved_total       = 0
    skipped_total       = 0
    calib_failed_total  = 0
    nodes_seen          = 0
    nodes_kept          = 0

    for i, seed in enumerate(todo):
        print(f"\n[{i+1}/{len(todo)}] seed={seed}", flush=True)

        try:
            tmp_dir, ep = generate_ssp_instance_dir(seed, args)
        except RuntimeError as e:
            print(f"    calibration échouée : {e}")
            calib_failed_total += 1
            with open(prog_file, "a") as pf:
                pf.write(f"{seed}\n")
            continue

        try:
            print(f"    ep généré : E={ep['E']} T={ep['T']} N={ep['N']} "
                  f"scale={ep['meta']['scale']:.3f} states={ep['dfa'].num_states}")

            inst_before  = _count_lines(inst_generic)
            nodes_before = _count_lines(nodes_generic)

            cmd = [args.binary, tmp_dir, str(args.time_limit),
                   outdir_staging, str(args.max_depth)]
            result = subprocess.run(cmd, capture_output=not args.verbose, text=True)

            if result.returncode != 0:
                print(f"    ERREUR (code {result.returncode})")
                if not args.verbose and result.stderr:
                    print(f"    stderr: {result.stderr.strip()}")
                skipped_total += 1
                with open(prog_file, "a") as pf:
                    pf.write(f"{seed}\n")
                continue

            solved_total += 1
            if args.verbose:
                for line in (result.stdout or "").splitlines():
                    if "OPTIMAL" in line or "TIMEOUT" in line or "INFEASIBLE" in line:
                        print(f"    {line.strip()}")

            # Instances : copier tel quel (toutes les lignes nouvelles)
            n_inst = _append_new_lines(inst_generic, inst_m, inst_before)

            # Nœuds : sampling (racine toujours gardée, 10 % du reste) — un
            # seul B&B par seed (pas de boucle epsilon comme pour MKP, cf.
            # docstring), donc pas de découpage en tranches nécessaire.
            new_node_lines = []
            if nodes_generic.exists():
                with open(nodes_generic) as f:
                    all_node_lines = f.readlines()
                new_node_lines = all_node_lines[nodes_before:]

            n_seen = len(new_node_lines)
            n_kept_this = 0
            if new_node_lines:
                kept_lines = _sample_lines(new_node_lines, args.sample_rate, rng)
                with open(nodes_m, "a") as f:
                    f.writelines(kept_lines)
                n_kept_this = len(kept_lines)

            nodes_seen += n_seen
            nodes_kept += n_kept_this

            status = "OPTIMAL (collectée)" if n_inst > 0 else "non optimale (ignorée)"
            print(f"     -> {status}  |  +{n_inst} instance(s)  |  "
                  f"nœuds: {n_seen} vus -> {n_kept_this} conservés")

            with open(prog_file, "a") as pf:
                pf.write(f"{seed}\n")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print(f"Terminé : {solved_total} B&B exécutés, {skipped_total} en erreur, "
          f"{calib_failed_total} échecs de calibration (sur {len(todo)} seeds)")
    print(f"Nœuds   : {nodes_seen} vus -> {nodes_kept} conservés"
          f" ({nodes_kept/max(nodes_seen,1)*100:.1f} %)")


if __name__ == "__main__":
    main()
