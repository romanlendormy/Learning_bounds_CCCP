"""
Générateur d'instances stochastiques CCSSP (Chance-Constrained Stochastic
Sequence Problem) selon la procédure de Pesant (2004).

Problème
--------
Trouver une séquence  x = (x_1, ..., x_n)  avec  x_t ∈ W = {0, ..., m-1}
qui maximise   Σ_{t,a}  profit[t][a] · 1[x_t = a]
sous la contrainte :

    P( pour tout j : l'automate j accepte x ) ≥ 1 − ε

Deux modes de génération disponibles :

Mode Pesant (défaut)
  Chaque scénario est constitué de k automates DFA tirés indépendamment :
  • transitions δ : Q × m → {0,...,Q-1} ∪ {-1}
      - (q, a) actif avec proba p_trans → état d'arrivée ~ Unif({0,...,Q-1})
      - sinon δ[q][a] = -1  (transition morte → rejet immédiat)
  • états acceptants : Bernoulli(p_acc) par état

Mode base corrélé (--base_mode)
  Pour chaque épisode, on génère k automates de BASE complètement définis :
  • toutes les transitions actives (p_trans = 1), destination ~ Unif({0,...,Q-1})
  • états acceptants : Bernoulli(p_acc) par état — FIXÉS pour tout l'épisode
  Puis pour chaque scénario, on PERTURBE chaque automate de base :
  • chaque transition (q, a) est conservée avec proba p_keep, sinon mise à -1
  • les états acceptants restent identiques à ceux de l'automate de base
  Ce mode induit une corrélation entre scénarios (structure partagée) tout en
  gardant une variabilité contrôlée par p_keep.

La matrice de profits (n × m, entiers dans [0, 100]) est DÉTERMINISTE
(identique pour tous les scénarios d'un même épisode).

Format de sortie — une ligne CSV par scénario
---------------------------------------------
  episode, scenario, n, m, k, Q,
  profit[0,0], ..., profit[n-1,m-1],         ← n·m valeurs (row = période, col = activité)
  delta_0[0,0], ..., delta_0[Q-1,m-1],       ⎫
  acc_0[0], ..., acc_0[Q-1],                 ⎬ k blocs (j = 0 .. k-1)
  ...                                         ⎭
  delta_{k-1}[...], acc_{k-1}[...]

  avec delta_j[q*m + a] ∈ {-1, 0, ..., Q-1}  et  acc_j[q] ∈ {0, 1}

Taille d'une ligne : 6 + n*m + k*(Q*m + Q) = 6 + n*m + k*Q*(m+1)

Regroupement : plusieurs réplications (= épisodes) × plusieurs tailles N
→ un fichier par taille : ccsp-pesant-n{n}-m{m}-k{k}-Q{Q}-S{N}.txt

Utilisation
-----------
  python create_database_ssp_pesant.py
  python create_database_ssp_pesant.py --n 10 --m 5 --k 3 --Q 4 --N_sizes 100 500 1000
  python create_database_ssp_pesant.py --p_trans 0.8 --p_acc 0.4 --n_reps 5 --seed 0
  python create_database_ssp_pesant.py --output_dir data/ssp/datasets
"""

import argparse
import os
import numpy as np


# ---------------------------------------------------------------------------
# Génération d'un automate (Pesant 2004)
# ---------------------------------------------------------------------------

def generate_automaton(Q: int, m: int, p_trans: float, p_acc: float,
                        rng: np.random.Generator) -> dict:
    """Génère un DFA aléatoire à Q états sur un alphabet de taille m.

    Transitions δ : Q × m → {-1, 0, ..., Q-1}
      • actif avec proba p_trans → état d'arrivée ~ Uniforme({0,...,Q-1})
      • inactif                  → -1 (transition morte)

    États acceptants : chaque état est terminal avec proba p_acc.
    État initial     : toujours 0.

    Retourne un dict {delta: (Q,m) int array, accepting: (Q,) bool array}.
    """
    # Activation des transitions : Bernoulli(p_trans)
    active = rng.random((Q, m)) < p_trans           # (Q, m) bool

    # États d'arrivée : Uniforme({0,...,Q-1}) quand actif, sinon -1
    arrivals = rng.integers(0, Q, size=(Q, m))      # (Q, m) int
    delta = np.where(active, arrivals, -1)           # (Q, m) int, -1 = inactif

    # États acceptants : Bernoulli(p_acc)
    accepting = rng.random(Q) < p_acc               # (Q,) bool

    return {"delta": delta, "accepting": accepting}


# ---------------------------------------------------------------------------
# Vérification qu'un automate est non trivial
# ---------------------------------------------------------------------------

def automaton_has_valid_path(Q: int, m: int, n: int, delta: np.ndarray,
                              accepting: np.ndarray) -> bool:
    """Retourne True si l'automate accepte au moins une séquence de longueur n."""
    # BFS sur l'ensemble des états atteignables après t étapes
    reachable = {0}
    for _ in range(n):
        next_states = set()
        for q in reachable:
            for a in range(m):
                q_next = int(delta[q, a])
                if q_next >= 0:
                    next_states.add(q_next)
        reachable = next_states
        if not reachable:
            return False
    # Vérifie qu'au moins un état atteignable est acceptant
    return any(accepting[q] for q in reachable)


# ---------------------------------------------------------------------------
# Génération d'un scénario
# ---------------------------------------------------------------------------

def generate_scenario(k: int, Q: int, m: int, n: int,
                       p_trans: float, p_acc: float,
                       rng: np.random.Generator,
                       ensure_feasible: bool = True,
                       max_retries: int = 100) -> list[dict]:
    """Génère k automates indépendants pour un scénario.

    Si ensure_feasible=True, chaque automate est re-tiré jusqu'à ce qu'il
    accepte au moins une séquence (évite les scénarios trivialement infaisables).
    """
    automata = []
    for _ in range(k):
        for attempt in range(max_retries):
            aut = generate_automaton(Q, m, p_trans, p_acc, rng)
            if not ensure_feasible or automaton_has_valid_path(
                    Q, m, n, aut["delta"], aut["accepting"]):
                automata.append(aut)
                break
        else:
            # Fallback : automate trivial qui accepte tout (état 0 acceptant,
            # toutes les transitions bouclent sur 0)
            delta_trivial = np.zeros((Q, m), dtype=int)
            accepting_trivial = np.zeros(Q, dtype=bool)
            accepting_trivial[0] = True
            automata.append({"delta": delta_trivial, "accepting": accepting_trivial})
    return automata


# ---------------------------------------------------------------------------
# Mode base corrélé : automate de base + perturbation par scénario
# ---------------------------------------------------------------------------

def generate_base_automaton(Q: int, m: int, p_acc: float,
                             rng: np.random.Generator) -> dict:
    """Génère un automate de base : toutes les transitions définies (p_trans=1),
    états acceptants tirés Bernoulli(p_acc).

    Retourne {delta: (Q,m) int array, accepting: (Q,) bool array}.
    """
    delta = rng.integers(0, Q, size=(Q, m))   # destination ~ Unif, toutes actives
    accepting = rng.random(Q) < p_acc
    return {"delta": delta, "accepting": accepting}


def perturb_automaton(base_aut: dict, p_keep: float,
                      rng: np.random.Generator) -> dict:
    """Retourne une copie de l'automate où chaque transition est conservée
    avec proba p_keep et mise à -1 sinon.  Les états acceptants sont inchangés.
    """
    delta = base_aut["delta"].copy()
    dead_mask = rng.random(delta.shape) >= p_keep   # True = à supprimer
    delta[dead_mask] = -1
    return {"delta": delta, "accepting": base_aut["accepting"].copy()}


# ---------------------------------------------------------------------------
# Construction d'une ligne CSV
# ---------------------------------------------------------------------------

def scenario_to_csv_values(profit: np.ndarray, automata: list[dict]) -> list:
    """Sérialise les données d'un scénario en liste de valeurs CSV.

    profit   : (n, m) int array
    automata : liste de k dicts {delta:(Q,m), accepting:(Q,)}

    Sortie :
      profit_flat(n*m), [delta_flat(Q*m), accepting_flat(Q)] × k
    """
    vals = list(map(int, profit.flatten()))   # n*m entiers
    for aut in automata:
        vals += list(map(int, aut["delta"].flatten()))      # Q*m (entiers, -1 possible)
        vals += list(map(int, aut["accepting"].flatten()))  # Q (0/1)
    return vals


# ---------------------------------------------------------------------------
# Génération complète d'un fichier
# ---------------------------------------------------------------------------

def generate_file(n: int, m: int, k: int, Q: int,
                  N: int, n_reps: int,
                  p_trans: float, p_acc: float,
                  seed_base: int,
                  output_dir: str,
                  ensure_feasible: bool = True,
                  base_mode: bool = False,
                  p_keep: float = 0.7) -> str:
    """Génère un fichier CSV pour une taille N donnée (n_reps épisodes × N scénarios).

    base_mode=False : mode Pesant — chaque automate tiré indépendamment (p_trans, p_acc).
    base_mode=True  : mode base corrélé — k automates de base par épisode (p_acc),
                      puis chaque scénario perturbe chaque base (p_keep).

    Retourne le chemin du fichier créé.
    """
    os.makedirs(output_dir, exist_ok=True)
    if base_mode:
        fname = f"ccsp-pesant-n{n}-m{m}-k{k}-Q{Q}-S{N}-base-pk{p_keep:.2f}.txt"
    else:
        fname = f"ccsp-pesant-n{n}-m{m}-k{k}-Q{Q}-S{N}.txt"
    fpath = os.path.join(output_dir, fname)

    total_lines = 0
    with open(fpath, "w") as f:
        for ep in range(n_reps):
            rng = np.random.default_rng(seed_base + ep * 100_000 + N)

            profit = rng.integers(0, 101, size=(n, m))

            if base_mode:
                # Générer les k automates de base pour cet épisode
                base_automata = [generate_base_automaton(Q, m, p_acc, rng)
                                 for _ in range(k)]

            for scen in range(N):
                if base_mode:
                    automata = [perturb_automaton(base, p_keep, rng)
                                for base in base_automata]
                else:
                    automata = generate_scenario(k, Q, m, n, p_trans, p_acc, rng,
                                                 ensure_feasible=ensure_feasible)
                data = scenario_to_csv_values(profit, automata)
                header = [ep, scen, n, m, k, Q]
                f.write(",".join(map(str, header + data)) + "\n")
                total_lines += 1

    print(f"  -> {fpath}  ({n_reps} épisodes × {N} scénarios = {total_lines} lignes)")
    return fpath


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Génère des instances CCSSP aléatoires selon Pesant (2004).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n",       type=int, default=10,
                        help="Nombre de périodes (longueur de la séquence)")
    parser.add_argument("--m",       type=int, default=5,
                        help="Nombre d'activités (|W| = m, W = {0,...,m-1})")
    parser.add_argument("--k",       type=int, default=3,
                        help="Nombre d'automates par scénario")
    parser.add_argument("--Q",       type=int, default=4,
                        help="Nombre d'états par automate")
    parser.add_argument("--N_sizes", type=int, nargs="+", default=[100, 500, 1000],
                        help="Liste des tailles de scénarios à générer")
    parser.add_argument("--n_reps",  type=int, default=5,
                        help="Nombre de réplications (épisodes) par taille")
    parser.add_argument("--p_trans", type=float, default=0.70,
                        help="Probabilité d'activation d'une transition")
    parser.add_argument("--p_acc",   type=float, default=0.50,
                        help="Probabilité qu'un état soit acceptant")
    parser.add_argument("--no_feasibility_check", action="store_true",
                        help="Désactive le re-tirage des automates trivialement infaisables (mode Pesant)")
    parser.add_argument("--base_mode", action="store_true",
                        help="Mode base corrélé : automates de base par épisode + perturbation par scénario")
    parser.add_argument("--p_keep",  type=float, default=0.70,
                        help="(base_mode) Probabilité de conserver chaque transition de l'automate de base")
    parser.add_argument("--output_dir", default="data/ssp/datasets",
                        help="Dossier de sortie")
    parser.add_argument("--seed",    type=int, default=42,
                        help="Graine de base pour la reproductibilité")
    args = parser.parse_args()

    ensure_feasible = not args.no_feasibility_check

    print("\n=== Générateur CCSSP — Pesant (2004) ===")
    print(f"  n={args.n}, m={args.m}, k={args.k}, Q={args.Q}")
    if args.base_mode:
        print(f"  MODE : base corrélé  (p_acc={args.p_acc}, p_keep={args.p_keep})")
    else:
        print(f"  MODE : Pesant  (p_trans={args.p_trans}, p_acc={args.p_acc}, ensure_feasible={ensure_feasible})")
    print(f"  N_sizes={args.N_sizes}, n_reps={args.n_reps}, seed={args.seed}")
    line_size = 6 + args.n * args.m + args.k * args.Q * (args.m + 1)
    print(f"  Taille d'une ligne : {line_size} valeurs\n")

    for N in args.N_sizes:
        generate_file(
            n=args.n, m=args.m, k=args.k, Q=args.Q,
            N=N, n_reps=args.n_reps,
            p_trans=args.p_trans, p_acc=args.p_acc,
            seed_base=args.seed,
            output_dir=args.output_dir,
            ensure_feasible=ensure_feasible,
            base_mode=args.base_mode,
            p_keep=args.p_keep,
        )

    print("\n=== Terminé ===")
    print("\nFormat de chaque ligne :")
    print("  [episode, scenario, n, m, k, Q,")
    print("   profit[0,0], ..., profit[n-1,m-1],         (n*m entiers, row=période, col=activité)")
    print("   delta_0[0,0], ..., delta_0[Q-1,m-1],       ⎫")
    print("   acc_0[0], ..., acc_0[Q-1],                 ⎬ k blocs (j = 0..k-1)")
    print("   ...]                                        ⎭")
    print("  delta_j[q*m + a] ∈ {-1, 0,...,Q-1}  (-1 = transition inactive)")
    print("  acc_j[q] ∈ {0, 1}  (1 = état terminal/acceptant)")


if __name__ == "__main__":
    main()
