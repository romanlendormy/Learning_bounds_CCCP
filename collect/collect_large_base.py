"""
collect_large_base.py — Collecte de données GNN avec pruning CP pur (gecode_base).

Identique à collect_large.py mais utilise bab_mkp_gecode_base_collect au lieu de
bab_mkp_gecode_collect. Le pruning repose uniquement sur la propagation CP de Gecode
(sans borne Lagrangienne), ce qui produit davantage de nœuds par instance.

Format de sortie des nœuds (Format C — sans ub_node ni incumbent) :
  inst_id, depth, curr_cost, x_mask[n], z_mask[N]  (3 + n + N champs)

Les résultats s'accumulent dans des fichiers suffixés par la valeur de m :
  <outdir>_base/instances_n{n}_m{m}.txt
  <outdir>_base/nodes_n{n}_m{m}.txt
  <outdir>_base/seeds_done_n{n}_m{m}.txt

Le binaire C++ écrit d'abord dans les fichiers génériques (instances_n{n}.txt,
nodes_n{n}.txt) ; le script Python redirige aussitôt les nouvelles lignes vers
les fichiers _m{m} correspondants.

Usage :
  python collect_large_base.py \\
      --n 20 --m 5 --N 100 --n_instances 100 \\
      --epsilon 0.1 0.2 0.3 --max_depth 10 \\
      --binary solver_bab/build/bab_mkp_gecode_base_collect \\
      --outdir data/train

Reprendre une collecte interrompue :
  python collect_large_base.py ... --seed_start 42
"""

import argparse
import os
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


# ─────────────────────────────────────────────────────────────────────────────
# Génération d'une instance en mémoire → fichier temporaire Song
# ─────────────────────────────────────────────────────────────────────────────

def generate_song_tempfile(n, m, N, seed,
                            alpha=0.5, R=1000,
                            correlation="weakly_correlated"):
    inst = generate_pisinger_instance(n, m, alpha, R, correlation, seed)

    rng_mu   = np.random.default_rng(seed)
    mu       = draw_mu(n, rng_mu)
    rng_scen = np.random.default_rng(seed + 1_000_000)

    b_flat = [round(float(v), 4) for v in inst["b"]]
    c_flat = [round(float(v), 4) for v in inst["c"]]

    tf = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="ccmk_tmp_"
    )
    for scen_id in range(N):
        A_k    = generate_scenario(inst["A"], mu, rng_scen)
        A_flat = [round(float(v), 4) for v in A_k.flatten()]
        header = [0, scen_id, m, n, 0]
        tf.write(",".join(str(v) for v in header + A_flat + b_flat + c_flat) + "\n")
    tf.close()
    return tf.name


# ─────────────────────────────────────────────────────────────────────────────
# Collecte d'une instance pour tous les epsilons
# ─────────────────────────────────────────────────────────────────────────────

def collect_instance(tmpfile, epsilons, args):
    """
    Appelle le binaire CP-pur pour chaque epsilon.
    Retourne (n_solved, n_skipped).
    """
    solved  = 0
    skipped = 0
    for eps in epsilons:
        cmd = [
            args.binary,
            tmpfile,
            str(eps),
            "-1",               # time_limit illimité
            args.outdir,
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
# Helpers pour rediriger les nouvelles lignes vers les fichiers _m{m}
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


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collecte GNN CC-MKP avec pruning CP pur (sans borne Lagrangienne).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--n",           type=int,   default=20)
    parser.add_argument("--m",           type=int,   default=5)
    parser.add_argument("--N",           type=int,   default=100,
                        help="Nombre de scénarios")
    parser.add_argument("--alpha",       type=float, default=0.5)
    parser.add_argument("--R",           type=int,   default=1000)
    parser.add_argument("--correlation", type=str,   default="weakly_correlated",
                        choices=["uncorrelated", "weakly_correlated",
                                 "strongly_correlated"])
    parser.add_argument("--epsilon",     nargs="+",  type=float,
                        default=[0.1, 0.2, 0.3])
    parser.add_argument("--n_instances", type=int,   default=100)
    parser.add_argument("--max_depth",   type=int,   default=10)
    parser.add_argument("--seed_start",  type=int,   default=0)
    parser.add_argument("--binary",  type=str,
                        default="solver_bab/build/bab_mkp_gecode_base_collect")
    parser.add_argument("--outdir",  type=str, default="data/train")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if not os.path.isfile(args.binary):
        print(f"Erreur : binaire introuvable : {args.binary}", file=sys.stderr)
        sys.exit(1)

    outdir = Path(f"{args.outdir}_base")
    outdir.mkdir(parents=True, exist_ok=True)

    # Fichiers génériques écrits par le binaire C++ (staging)
    inst_generic  = outdir / f"instances_n{args.n}.txt"
    nodes_generic = outdir / f"nodes_n{args.n}.txt"

    # Fichiers cibles suffixés par m
    inst_m  = outdir / f"instances_n{args.n}_m{args.m}.txt"
    nodes_m = outdir / f"nodes_n{args.n}_m{args.m}.txt"

    # Fichier de progression spécifique à m (évite les conflits entre runs parallèles)
    progress_file = outdir / f"seeds_done_n{args.n}_m{args.m}.txt"

    done_seeds: set[int] = set()
    if progress_file.exists():
        done_seeds = {int(l.strip()) for l in progress_file.read_text().splitlines() if l.strip()}
        if done_seeds:
            print(f"[reprise] {len(done_seeds)} seeds déjà traitées (d'après {progress_file.name})")

    todo: list[int] = []
    seed = args.seed_start
    while len(todo) < args.n_instances:
        if seed not in done_seeds:
            todo.append(seed)
        seed += 1

    print("=" * 60)
    print(f"Collecte CC-MKP (CP pur) : n={args.n}  m={args.m}  N={args.N}")
    print(f"  demandées    : {args.n_instances}  (seeds {todo[0]}..{todo[-1]})")
    print(f"  à traiter    : {len(todo)}  ({len(done_seeds)} déjà faites)")
    print(f"  epsilon      : {args.epsilon}")
    print(f"  max_depth    : {args.max_depth}")
    print(f"  sortie       : {inst_m.name}, {nodes_m.name}")
    print("=" * 60)

    solved_total  = 0
    skipped_total = 0

    for i, seed in enumerate(todo):
        print(f"\n[{i+1}/{len(todo)}] seed={seed}", flush=True)

        # Lignes déjà présentes dans les fichiers génériques avant cet appel
        inst_before  = _count_lines(inst_generic)
        nodes_before = _count_lines(nodes_generic)

        tmpfile = generate_song_tempfile(
            args.n, args.m, args.N, seed,
            alpha=args.alpha, R=args.R, correlation=args.correlation,
        )
        try:
            s, k = collect_instance(tmpfile, args.epsilon, args)
            solved_total  += s
            skipped_total += k
            status = f"OK ({s}/{len(args.epsilon)} eps)" if k == 0 else f"{s} OK, {k} ignorés"
            print(f"  -> {status}")

            # Rediriger les nouvelles lignes vers les fichiers _m{m}
            n_inst  = _append_new_lines(inst_generic,  inst_m,  inst_before)
            n_nodes = _append_new_lines(nodes_generic, nodes_m, nodes_before)
            if n_inst or n_nodes:
                print(f"     +{n_inst} instances, +{n_nodes} nœuds → _m{args.m}")

            with open(progress_file, "a") as pf:
                pf.write(f"{seed}\n")
        finally:
            os.unlink(tmpfile)

    print("\n" + "=" * 60)
    print(f"Terminé : {solved_total} collectes réussies"
          f", {skipped_total} ignorées"
          f"  (sur {len(todo)} x {len(args.epsilon)} = "
          f"{len(todo) * len(args.epsilon)} au total)")


if __name__ == "__main__":
    main()
