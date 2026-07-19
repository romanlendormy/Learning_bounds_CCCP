"""
collect_kp.py — Collecte de données GNN pour le knapsack stochastique
chance-contraint (CC-KP), cas particulier m=1 du CC-MKP.

Copie quasi conforme de collect_large.py (CC-MKP à m contraintes) : même
générateur Pisinger+Song, même binaire C++ (bab_mkp_gecode_collect), même
format de sortie. La seule différence de fond est m=1, fixé en dur ci-dessous
(pas de --m en CLI) — le reste (θ, ζ, branchement, format des fichiers
instances_*/nodes_*) est strictement générique en m côté C++
(cf. solver_bab/mkp/gecode_bab.hpp, boucles `for j in range(m)`), donc
fonctionne sans aucune modification :
  - mode v1ld  : MKPProblemBnB::solve_zeta = B&B multi-contraintes (Dantzig) ;
                 à m=1, c'est exactement un B&B knapsack 1D exact.
  - mode v1cld : MKPCLDProblem::solve_zeta = knapsack 1D exact par contrainte ;
                 à m=1, une seule "copie" par scénario, donc v1cld ≡ v1ld
                 (la décomposition CLD ne fait plus rien puisqu'il n'y a
                 rien à décomposer).

Génère n_instances instances Pisinger + perturbation Song en mémoire,
les résout avec bab_mkp_gecode_collect pour chaque epsilon, jusqu'à max_depth.
Les instances ne sont PAS sauvegardées sur disque : un fichier temporaire est
créé juste le temps du B&B, puis supprimé.

Les résultats s'accumulent dans :
  <outdir>_<mode>/instances_n{n}.txt
  <outdir>_<mode>/nodes_n{n}.txt
(répertoire et start_id gérés automatiquement par le binaire C++)

Corrélation par défaut — "weakly_correlated" : calibration empirique
(n=10..30, epsilon=0.1, N=100, T_root=200/T_node=30, cap 45s) montrant une
frontière dur-mais-résoluble nette entre n=15 (résolu en 5-7s) et n=20
(timeout) pour ce régime. La classe fortement corrélée (profit ≈ poids +
R/10, cf. Pisinger) donne un comportement moins fiable à petits n déjà
(instances INFEASIBLE dès n=10 dans la même calibration) et n'a pas été
retenue comme défaut.

Usage :
  python collect_kp.py \\
      --n 30 --N 100 --n_instances 100 \\
      --epsilon 0.1 0.2 0.3 --max_depth 10 \\
      --binary solver_bab/build/bab_mkp_gecode_collect \\
      --outdir data/train_kp --mode v1ld

Reprendre une collecte interrompue (avec --seed_start) :
  python collect_kp.py ... --seed_start 42
  Le binaire détecte automatiquement le start_id depuis les fichiers existants.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# Import des fonctions de génération depuis le générateur Pisinger existant
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


# ─────────────────────────────────────────────────────────────────────────────
# Génération d'une instance en mémoire → fichier temporaire Song
# ─────────────────────────────────────────────────────────────────────────────

def generate_song_tempfile(n, m, N, seed,
                            alpha=0.5, R=1000,
                            correlation="strongly_correlated"):
    """
    Génère une instance CC-KP (base Pisinger + perturbation Song) et
    l'écrit dans un fichier temporaire au format Song CSV.

    Format de chaque ligne :
      0, scen_id, m, n, 0,  A[j=0..m-1, k=0..n-1]_flat,  b[m],  c[n]

    Retourne le chemin du fichier temporaire (à supprimer après usage).
    """
    inst = generate_pisinger_instance(n, m, alpha, R, correlation, seed)

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
        header = [0, scen_id, m, n, 0]   # rep=0 pour toutes les lignes
        tf.write(",".join(str(v) for v in header + A_flat + b_flat + c_flat) + "\n")
    tf.close()
    return tf.name


# ─────────────────────────────────────────────────────────────────────────────
# Collecte d'une instance pour tous les epsilons
# ─────────────────────────────────────────────────────────────────────────────

def collect_instance(tmpfile, epsilons, args):
    """
    Appelle le binaire C++ pour chaque epsilon sur le fichier temporaire.
    Retourne (n_solved, n_skipped).
    """
    solved  = 0
    skipped = 0
    for eps in epsilons:
        cmd = [
            args.binary,
            tmpfile,
            str(eps),
            str(args.T_node),
            "-1",               # time_limit illimité
            str(args.T_root),
            args.mode,
            args.outdir,
            str(args.max_depth),
            "1" if args.patience_mode else "0",
            str(args.patience_tol),
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
                # Extraire la ligne de résumé du binaire
                for line in (result.stdout or "").splitlines():
                    if "OPTIMAL" in line or "INFEASIBLE" in line or "FEASIBLE" in line:
                        print(f"    eps={eps:.1f}  {line.strip()}")
    return solved, skipped


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collecte de données GNN pour le knapsack stochastique chance-contraint (CC-KP, m=1).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Paramètres du problème
    parser.add_argument("--n",           type=int,   default=30)
    parser.add_argument("--N",           type=int,   default=100,
                        help="Nombre de scénarios")
    parser.add_argument("--alpha",       type=float, default=0.5,
                        help="Tightness α (défaut 0.5)")
    parser.add_argument("--R",           type=int,   default=1000,
                        help="Range [1,R] pour poids/profits (défaut 1000)")
    parser.add_argument("--correlation", type=str,   default="weakly_correlated",
                        choices=["uncorrelated", "weakly_correlated",
                                 "strongly_correlated"],
                        help="Défaut weakly_correlated : frontière "
                             "dur-mais-résoluble validée par calibration "
                             "(cf. docstring du module)")
    # Collecte
    parser.add_argument("--epsilon",     nargs="+",  type=float,
                        default=[0.1, 0.2, 0.3])
    parser.add_argument("--n_instances", type=int,   default=100)
    parser.add_argument("--max_depth",   type=int,   default=10)
    parser.add_argument("--seed_start",  type=int,   default=0,
                        help="Graine de départ (pour reprendre une collecte)")
    # Solveur
    parser.add_argument("--binary",  type=str,
                        default="solver_bab/build/bab_mkp_gecode_collect")
    parser.add_argument("--outdir",  type=str, default="data/train_kp")
    parser.add_argument("--mode",    type=str, default="v1ld",
                        choices=["v1ld", "v1cld"],
                        help="v1ld ≡ v1cld à m=1 (aucune contrainte à "
                             "décomposer) ; v1ld par défaut, plus direct")
    parser.add_argument("--T_root",       type=int,   default=200)
    parser.add_argument("--T_node",       type=int,   default=30)
    parser.add_argument("--patience_mode", action="store_true",
                        help="Convergence par tolérance au lieu d'itérations fixes")
    parser.add_argument("--patience_tol", type=float, default=1.0,
                        help="Tolérance de convergence (défaut 1.0)")
    parser.add_argument("--verbose", action="store_true",
                        help="Affiche la sortie du binaire C++")

    args = parser.parse_args()

    # Vérifier que le binaire existe
    if not os.path.isfile(args.binary):
        print(f"Erreur : binaire introuvable : {args.binary}", file=sys.stderr)
        sys.exit(1)

    # ── Fichier de progression (reprise automatique) ──────────────────────────
    outdir = Path(f"{args.outdir}_{args.mode}")
    outdir.mkdir(parents=True, exist_ok=True)
    progress_file = outdir / f"seeds_done_n{args.n}.txt"

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
    print(f"Collecte CC-KP : n={args.n}  m={M}  N={args.N}  corr={args.correlation}")
    print(f"  demandées    : {args.n_instances}  (seeds {todo[0]}..{todo[-1]})")
    print(f"  à traiter    : {len(todo)}  ({len(done_seeds)} déjà faites)")
    print(f"  epsilon      : {args.epsilon}")
    print(f"  max_depth    : {args.max_depth}")
    node_cfg = (f"patience(tol={args.patience_tol})" if args.patience_mode
                else f"T_node={args.T_node}")
    print(f"  mode         : {args.mode}  T_root={args.T_root}  {node_cfg}")
    print(f"  sortie       : {outdir}/")
    print("=" * 60)

    solved_total  = 0
    skipped_total = 0

    for i, seed in enumerate(todo):
        print(f"\n[{i+1}/{len(todo)}] seed={seed}", flush=True)

        tmpfile = generate_song_tempfile(
            args.n, M, args.N, seed,
            alpha=args.alpha, R=args.R, correlation=args.correlation,
        )
        try:
            s, k = collect_instance(tmpfile, args.epsilon, args)
            solved_total  += s
            skipped_total += k
            status = f"OK ({s}/{len(args.epsilon)} eps)" if k == 0 else f"{s} OK, {k} ignorés"
            print(f"  -> {status}")
            # Marquer la seed comme traitée
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
