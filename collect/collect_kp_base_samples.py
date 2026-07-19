"""
collect_kp_base_samples.py — Collecte CC-KP (knapsack stochastique
chance-contraint, cas particulier m=1 du CC-MKP) avec pruning CP pur +
sampling 10 %.

Analogue de collect_large_base_samples.py (CC-MKP à m contraintes) : même
binaire C++ (bab_mkp_gecode_base_collect, générique en m — lit m depuis le
fichier Song, cf. gecode_base/main_collect_gecode_base.cpp), même format de
sortie, même logique de sampling/reprise. La seule différence de fond est
m=1, fixé en dur ci-dessous (pas de --m en CLI, même choix que collect_kp.py)
pour éviter toute confusion avec le CC-MKP.

⚠️ Ne PAS confondre avec collect_kp.py (lignée collect_large.py — mode
v1ld, sans sampling, time_limit illimité). Ce script-ci reproduit la lignée
qui a RÉELLEMENT servi à collecter train_base_s10 côté MKP : pruning CP pur
(REF, aucune borne calculée pendant la collecte), les labels θ/ζ sont
recalculés a posteriori côté Python au chargement (cf.
gnn/train_scaled_v1ld.py). C'est ce qui permet de scaler à 10 000
instances/combo sans buter sur le coût par nœud de V1LD (cf. sweep
sweep_kp_gecode_grid.py : V1LD timeout déjà à n=20/N≥250, REF non).

À la fin de chaque B&B (chaque appel binaire / chaque epsilon), seuls
`--sample_rate` (défaut 10 %) des nœuds collectés pour cette instance sont
conservés (+ la racine, toujours gardée), tirés uniformément au hasard.

Le format de sortie est scrupuleusement identique au format C :
  inst_id, depth, curr_cost, x_mask[n], z_mask[N]  (3 + n + N champs)

Les résultats s'accumulent dans :
  <outdir>_base_s10/instances_n{n}_m1_N{N}.txt
  <outdir>_base_s10/nodes_n{n}_m1_N{N}.txt
  <outdir>_base_s10/seeds_done_n{n}_m1_N{N}.txt

Usage :
  python collect_kp_base_samples.py \\
      --n 10 --N 100 --n_instances 200 \\
      --epsilon 0.1 --max_depth 10 \\
      --binary solver_bab/build/bab_mkp_gecode_base_collect \\
      --outdir data/train_kp

Reprendre une collecte interrompue :
  python collect_kp_base_samples.py ... --seed_start 400
"""

import argparse
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_GEN_DIR = Path(__file__).parent.parent / "data" / "knapsack"
sys.path.insert(0, str(_GEN_DIR))
from create_database_knapsack_pisinger import (
    generate_pisinger_instance,
    draw_mu,
    generate_scenario,
)

# Knapsack simple : une seule contrainte de ressource. Fixé en dur (pas de
# --m en CLI) pour éviter toute confusion avec le CC-MKP — cf. docstring.
M = 1

SAMPLE_RATE = 0.10   # fraction de nœuds conservés par B&B


# ─────────────────────────────────────────────────────────────────────────────
# Génération d'une instance → fichier temporaire Song
# ─────────────────────────────────────────────────────────────────────────────

def generate_song_tempfile(n, N, seed,
                            alpha=0.5, R=1000,
                            correlation="weakly_correlated"):
    inst = generate_pisinger_instance(n, M, alpha, R, correlation, seed)

    rng_mu   = np.random.default_rng(seed)
    mu       = draw_mu(n, rng_mu)
    rng_scen = np.random.default_rng(seed + 1_000_000)

    b_flat = [round(float(v), 4) for v in inst["b"]]
    c_flat = [round(float(v), 4) for v in inst["c"]]

    tf = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="cckp_tmp_"
    )
    for scen_id in range(N):
        A_k    = generate_scenario(inst["A"], mu, rng_scen)
        A_flat = [round(float(v), 4) for v in A_k.flatten()]
        header = [0, scen_id, M, n, 0]
        tf.write(",".join(str(v) for v in header + A_flat + b_flat + c_flat) + "\n")
    tf.close()
    return tf.name


# ─────────────────────────────────────────────────────────────────────────────
# Helpers fichiers
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
# Collecte d'une instance pour tous les epsilons
# ─────────────────────────────────────────────────────────────────────────────

def collect_instance(tmpfile, epsilons, args):
    solved  = 0
    skipped = 0
    for eps in epsilons:
        cmd = [
            args.binary,
            tmpfile,
            str(eps),
            str(args.time_limit),
            args.outdir_staging,
            str(args.max_depth),
        ]
        result = subprocess.run(cmd, capture_output=not args.verbose, text=True)
        if result.returncode != 0:
            print(f"    ERREUR eps={eps} (code {result.returncode})")
            if not args.verbose and result.stderr:
                print(f"    stderr: {result.stderr.strip()}")
            skipped += 1
        else:
            solved += 1
            if args.verbose:
                for line in (result.stdout or "").splitlines():
                    if "OPTIMAL" in line or "INFEASIBLE" in line or "FEASIBLE" in line:
                        print(f"    eps={eps:.1f}  {line.strip()}")
    return solved, skipped


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collecte GNN CC-KP (CP pur, m=1) avec sampling 10 % par B&B.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--n",           type=int,   default=10)
    parser.add_argument("--N",           type=int,   default=100)
    parser.add_argument("--alpha",       type=float, default=0.5)
    parser.add_argument("--R",           type=int,   default=1000)
    parser.add_argument("--correlation", type=str,   default="weakly_correlated",
                        choices=["uncorrelated", "weakly_correlated",
                                 "strongly_correlated"])
    parser.add_argument("--epsilon",     nargs="+",  type=float,
                        default=[0.1])
    parser.add_argument("--n_instances", type=int,   default=100)
    parser.add_argument("--max_depth",   type=int,   default=10)
    parser.add_argument("--seed_start",  type=int,   default=0)
    parser.add_argument("--sample_rate", type=float, default=SAMPLE_RATE,
                        help=f"Fraction de nœuds à conserver par B&B (défaut {SAMPLE_RATE})")
    parser.add_argument("--rng_seed",    type=int,   default=0,
                        help="Graine pour le sampling (défaut 0)")
    parser.add_argument("--time_limit",   type=float, default=-1.0,
                        help="Limite en secondes par B&B (-1 = illimité)")
    parser.add_argument("--binary",      type=str,
                        default="solver_bab/build/bab_mkp_gecode_base_collect")
    parser.add_argument("--outdir",      type=str,   default="data/train_kp")
    parser.add_argument("--verbose",     action="store_true")

    args = parser.parse_args()

    if not os.path.isfile(args.binary):
        print(f"Erreur : binaire introuvable : {args.binary}", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.rng_seed)

    # Dossier de staging (fichiers génériques écrits par le binaire C++)
    # Le binaire C++ appende _base lui-même → on passe args.outdir, pas staging_dir.
    staging_dir = Path(f"{args.outdir}_base")
    staging_dir.mkdir(parents=True, exist_ok=True)
    args.outdir_staging = args.outdir  # binaire écrit dans {outdir}_base/

    # Dossier de sortie final (avec sampling)
    outdir = Path(f"{args.outdir}_base_s10")
    outdir.mkdir(parents=True, exist_ok=True)

    # Fichiers génériques (staging) — noms imposés par le binaire C++, indexés
    # par n SEUL (pas par N) : cf. gecode_base/main_collect_gecode_base.cpp.
    inst_generic  = staging_dir / f"instances_n{args.n}.txt"
    nodes_generic = staging_dir / f"nodes_n{args.n}.txt"

    # Fichiers cibles _m1 dans le dossier samplé (m=1 explicite dans le nom,
    # même si toujours 1 ici, pour rester comparable/greppable vs MKP et
    # éviter le bug de collision inter-N déjà rencontré côté MKP quand N est
    # absent du nom de fichier de progression).
    inst_m    = outdir / f"instances_n{args.n}_m{M}_N{args.N}.txt"
    nodes_m   = outdir / f"nodes_n{args.n}_m{M}_N{args.N}.txt"
    prog_file = outdir / f"seeds_done_n{args.n}_m{M}_N{args.N}.txt"

    done_seeds: set[int] = set()
    if prog_file.exists():
        done_seeds = {int(l.strip()) for l in prog_file.read_text().splitlines() if l.strip()}
        if done_seeds:
            print(f"[reprise] {len(done_seeds)} seeds déjà traitées")

    # --n_instances est un TOTAL cible (pas "combien en ajouter") : sur reprise,
    # on ne complète que ce qu'il manque pour l'atteindre.
    remaining = max(0, args.n_instances - len(done_seeds))
    todo: list[int] = []
    seed = args.seed_start
    while len(todo) < remaining:
        if seed not in done_seeds:
            todo.append(seed)
        seed += 1

    print("=" * 60)
    print(f"Collecte CC-KP (CP pur, sampling {args.sample_rate*100:.0f}%) :"
          f" n={args.n}  m={M}  N={args.N}")
    print(f"  cible     : {args.n_instances} instances au total"
          + (f"  (seeds {todo[0]}..{todo[-1]})" if todo else ""))
    print(f"  à traiter : {len(todo)}  ({len(done_seeds)} déjà faites)")
    print(f"  epsilon   : {args.epsilon}")
    print(f"  max_depth : {args.max_depth}")
    print(f"  sortie    : {nodes_m.name}  (dans {outdir})")
    print("=" * 60)

    if not todo:
        print("Rien à faire : cible déjà atteinte.")
        return

    solved_total  = 0
    skipped_total = 0
    nodes_seen    = 0
    nodes_kept    = 0

    for i, seed in enumerate(todo):
        print(f"\n[{i+1}/{len(todo)}] seed={seed}", flush=True)

        inst_before  = _count_lines(inst_generic)
        nodes_before = _count_lines(nodes_generic)

        tmpfile = generate_song_tempfile(
            args.n, args.N, seed,
            alpha=args.alpha, R=args.R, correlation=args.correlation,
        )
        try:
            s, k = collect_instance(tmpfile, args.epsilon, args)
            solved_total  += s
            skipped_total += k

            status = f"OK ({s}/{len(args.epsilon)} eps)" if k == 0 else f"{s} OK, {k} ignorés"
            print(f"  -> {status}")

            # Instances : copier tel quel (toutes les lignes nouvelles)
            n_inst = _append_new_lines(inst_generic, inst_m, inst_before)

            # Nœuds : sampling par epsilon (chaque tranche = un B&B).
            # La racine (depth=0) est toujours conservée ; sample_rate du reste.
            new_node_lines = []
            if nodes_generic.exists():
                with open(nodes_generic) as f:
                    all_node_lines = f.readlines()
                new_node_lines = all_node_lines[nodes_before:]

            n_eps_solved = s
            if new_node_lines and n_eps_solved > 0:
                # Découper en tranches par epsilon (le binaire écrit en séquence)
                chunk_size, rem = divmod(len(new_node_lines), n_eps_solved)
                kept_lines = []
                start = 0
                for eps_idx in range(n_eps_solved):
                    end = start + chunk_size + (1 if eps_idx < rem else 0)
                    chunk = new_node_lines[start:end]
                    kept_lines.extend(_sample_lines(chunk, args.sample_rate, rng))
                    start = end

                with open(nodes_m, "a") as f:
                    f.writelines(kept_lines)

                n_seen = len(new_node_lines)
                n_kept_this = len(kept_lines)
            else:
                n_seen = 0
                n_kept_this = 0

            nodes_seen += n_seen
            nodes_kept += n_kept_this

            print(f"     +{n_inst} instances  |  nœuds: {n_seen} vus → {n_kept_this} conservés")

            with open(prog_file, "a") as pf:
                pf.write(f"{seed}\n")
        finally:
            os.unlink(tmpfile)

    print("\n" + "=" * 60)
    print(f"Terminé : {solved_total} collectes réussies, {skipped_total} ignorées")
    print(f"Nœuds   : {nodes_seen} vus → {nodes_kept} conservés"
          f" ({nodes_kept/max(nodes_seen,1)*100:.1f} %)")


if __name__ == "__main__":
    main()
