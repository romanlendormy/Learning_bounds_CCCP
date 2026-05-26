"""
Générateur d'instances stochastiques CCMKP selon la procédure de Song et al.
(Section 7 du supplément de Song et al., 2023).

Procédure
---------
Point de départ : une instance déterministe (A_base, b_base, c) de l'OR-Library.

1. Pour chaque item j : tirer μ_j ~ Exp(mean=0.1), tronqué à [0, 1].
   (μ_j = probabilité que l'item j "fail to appear" dans un scénario)
   Ces μ_j sont tirés UNE SEULE FOIS pour toute la génération de l'instance.

2. Pour chaque scénario k = 1..S :
   - Pour chaque item j :
       * Bernoulli(μ_j) → si l'item fail : A^k_{:, j} = 0  (colonne entière à 0)
       * Sinon : A^k_{i,j} ~ N(A_{i,j}, (0.1 * A_{i,j})²), projeté R+
     → les poids d'un même item sur différentes lignes sont CORRÉLÉS
       (partagent le même tirage Bernoulli).
   - b^k = b_base  (capacités déterministes)
   - c^k = c_base  (profits déterministes)

3. Tailles testées : |S| ∈ {100, 500, 1000, 3000}
4. 5 réplications indépendantes par taille |S|.
   (chaque réplication = nouveau tirage de S scénarios avec les mêmes μ_j)

Format de sortie — identique au générateur existant, une ligne par scénario :
  [episode, scenario, m, n, replication_id,
   A^k_00, ..., A^k_{m-1,n-1},   (m*n valeurs, ligne=contrainte)
   b_0, ..., b_{m-1}]            (m valeurs, FIXES)

Champ 4 (ex-sigma) = replication_id ∈ {0, 1, 2, 3, 4}
→ Le loader existant (load_mkp_instances_py) lit ce champ comme "sigma" :
  il ne sert pas dans le solver, donc la compatibilité est préservée.

Un fichier compagnon *-mu.txt enregistre les μ_j (une valeur par ligne)
pour traçabilité et reproductibilité.

Utilisation
-----------
  python create_database_knapsack_song.py mknap1-4
  python create_database_knapsack_song.py mknap1-6 --S_sizes 100 500 1000
  python create_database_knapsack_song.py mknap1-4 --n_reps 5 --seed 42
  python create_database_knapsack_song.py mknap1-4 --output_dir data/knapsack/datasets/song

Argument instance : <file>-<idx>  (ex: mknap1-4 = 4e instance de mknap1.txt)
"""

import argparse
import os
import sys
import numpy as np


# ---------------------------------------------------------------------------
# Lecture OR-Library
# ---------------------------------------------------------------------------

def _tokens(filepath):
    with open(filepath) as f:
        for line in f:
            for tok in line.split():
                yield tok


def parse_orlib_file(filepath):
    """Parse un fichier OR-Library (mknap1 / mknapcb*).

    Retourne une liste de dicts :
        {name, n, m, opt, c, A, b}
    """
    gen      = _tokens(filepath)
    basename = os.path.splitext(os.path.basename(filepath))[0]
    K        = int(next(gen))
    out      = []
    for k in range(K):
        n   = int(next(gen))
        m   = int(next(gen))
        opt = float(next(gen))
        c   = np.array([float(next(gen)) for _ in range(n)])
        A   = np.array([[float(next(gen)) for _ in range(n)] for _ in range(m)])
        b   = np.array([float(next(gen)) for _ in range(m)])
        out.append(dict(
            name=f"{basename}_{k + 1}",
            n=n, m=m, opt=opt,
            c=c, A=A, b=b,
        ))
    return out


def get_instance(instance_spec, orlib_dir):
    """Résoud 'mknap1-4' → (instance_dict, filepath).

    instance_spec : '<file>-<idx>'  (idx 1-based, ex: mknap1-4)
    orlib_dir     : dossier contenant les fichiers .txt
    """
    parts = instance_spec.rsplit("-", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Format attendu : <fichier>-<indice>  ex: mknap1-4  (reçu: {instance_spec!r})"
        )
    file_base, idx_str = parts
    idx = int(idx_str)  # 1-based

    # Cherche le fichier dans orlib_dir
    candidates = [
        os.path.join(orlib_dir, file_base + ".txt"),
        os.path.join(orlib_dir, file_base),
    ]
    filepath = next((p for p in candidates if os.path.exists(p)), None)
    if filepath is None:
        raise FileNotFoundError(
            f"Fichier OR-Library introuvable pour {file_base!r} dans {orlib_dir!r}"
        )

    instances = parse_orlib_file(filepath)
    if idx < 1 or idx > len(instances):
        raise IndexError(
            f"Indice {idx} hors bornes : {file_base}.txt contient {len(instances)} instances"
        )
    return instances[idx - 1], filepath


# ---------------------------------------------------------------------------
# Génération Song
# ---------------------------------------------------------------------------

def draw_mu(n, rng):
    """μ_j ~ Exp(mean=0.1) tronqué à [0, 1], j = 0..n-1."""
    mu = rng.exponential(scale=0.1, size=n)
    return np.minimum(mu, 1.0)


def generate_scenario_song(A_base, mu, rng):
    """Génère une réalisation A^k selon la procédure Song.

    Pour chaque item j :
      - Bernoulli(μ_j) → 0 si fail (colonne entière à 0)
      - Sinon A^k_{i,j} ~ N(A_{i,j}, (0.1 * A_{i,j})²), projeté R+

    Retourne A^k ∈ R^{m×n} (non-négatif).
    """
    m, n  = A_base.shape
    A_k   = np.zeros((m, n))

    # Bernoulli : vaut 1 si l'item "fail to appear"
    fail  = rng.binomial(1, mu, size=n).astype(bool)  # (n,)

    # Items qui apparaissent (fail == False)
    appear = ~fail
    if appear.any():
        A_appear  = A_base[:, appear]                          # (m, n_appear)
        std       = 0.1 * A_appear
        noise     = rng.normal(0.0, std)
        A_k[:, appear] = np.maximum(A_appear + noise, 0.0)
    # Colonnes fail restent à 0 (déjà initialisées)

    return A_k


def generate_replication(inst, mu, S, rng):
    """Génère une réplication de S scénarios.

    Retourne une liste de S lignes CSV (chaque ligne = liste de floats).
    Format : A_flat + b_flat + c_flat
    (A varie par scénario, b et c sont déterministes et répétés pour auto-suffisance)
    """
    A_base = inst["A"]      # (m, n)
    b_base = inst["b"]      # (m,)
    c_base = inst["c"]      # (n,)
    m, n   = A_base.shape

    b_flat = [round(v, 4) for v in b_base]
    c_flat = [round(v, 4) for v in c_base]

    lines = []
    for _ in range(S):
        A_k    = generate_scenario_song(A_base, mu, rng)
        A_flat = [round(v, 4) for v in A_k.flatten()]
        lines.append(A_flat + b_flat + c_flat)
    return lines


def generate_all_splits(inst, mu, S_sizes, n_reps, seed_base, output_dir):
    """Génère tous les fichiers pour les tailles S_sizes × n_reps réplications."""
    inst_name = inst["name"]
    os.makedirs(output_dir, exist_ok=True)

    for S in S_sizes:
        outpath = os.path.join(output_dir, f"ccmk-song-{inst_name}-S{S}.txt")
        total   = 0

        with open(outpath, "w") as f:
            for rep in range(n_reps):
                rng   = np.random.default_rng(seed_base + rep * 10_000 + S)
                lines = generate_replication(inst, mu, S, rng)

                for scen_id, data_line in enumerate(lines):
                    # [episode=rep, scenario, m, n, replication_id, A_flat, b_flat, c_flat]
                    header = [rep, scen_id, inst["m"], inst["n"], rep]
                    f.write(",".join(str(v) for v in header + data_line) + "\n")
                    total += 1

        print(f"  -> {outpath}  ({n_reps} réplications × {S} scénarios = {total} lignes)")


def save_mu(inst, mu, output_dir):
    """Enregistre μ_j dans un fichier texte compagnon."""
    mu_path = os.path.join(output_dir, f"ccmk-song-{inst['name']}-mu.txt")
    with open(mu_path, "w") as f:
        f.write("# mu_j (probabilité de défaillance par item) — Song et al.\n")
        f.write(f"# instance={inst['name']}  n={inst['n']}\n")
        for j, v in enumerate(mu):
            f.write(f"{j},{v:.8f}\n")
    print(f"  -> {mu_path}  (μ_j enregistrés)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Génère des instances CCMKP selon la procédure Song et al."
    )
    parser.add_argument(
        "instance",
        help="Instance OR-Library au format <fichier>-<indice>  (ex: mknap1-4)",
    )
    parser.add_argument(
        "--orlib_dir",
        default="data/knapsack/beasley",
        help="Dossier contenant les fichiers OR-Library (défaut: data/knapsack/beasley)",
    )
    parser.add_argument(
        "--output_dir",
        default="data/knapsack/datasets-song",
        help="Dossier de sortie (défaut: data/knapsack/datasets-song)",
    )
    parser.add_argument(
        "--S_sizes",
        nargs="+",
        type=int,
        default=[100, 500, 1000, 3000],
        help="Tailles de scénarios à générer (défaut: 100 500 1000 3000)",
    )
    parser.add_argument(
        "--n_reps",
        type=int,
        default=5,
        help="Nombre de réplications indépendantes par taille (défaut: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine de base pour la reproductibilité (défaut: 42)",
    )
    args = parser.parse_args()

    print(f"\n=== Génération Song et al. — {args.instance} ===")
    print(f"    S_sizes : {args.S_sizes}")
    print(f"    n_reps  : {args.n_reps}")
    print(f"    seed    : {args.seed}")

    # Charger l'instance
    inst, filepath = get_instance(args.instance, args.orlib_dir)
    print(f"\n  Instance : {inst['name']}  (n={inst['n']}, m={inst['m']}, opt={inst['opt']})")
    print(f"  Fichier  : {filepath}\n")

    # Tirer μ_j une seule fois (caractérise l'instance, pas les réplications)
    rng_mu = np.random.default_rng(args.seed)
    mu     = draw_mu(inst["n"], rng_mu)
    print(f"  μ_j tirés (Exp(0.1) tronqué [0,1]) :")
    print(f"    min={mu.min():.4f}  moy={mu.mean():.4f}  max={mu.max():.4f}")
    print(f"    items avec μ>0.1 : {(mu > 0.1).sum()}/{inst['n']}\n")

    # Générer les fichiers
    generate_all_splits(
        inst=inst,
        mu=mu,
        S_sizes=args.S_sizes,
        n_reps=args.n_reps,
        seed_base=args.seed + 1,   # +1 pour séparer du tirage de μ
        output_dir=args.output_dir,
    )

    # Sauvegarder μ_j
    save_mu(inst, mu, args.output_dir)

    print("\n=== Terminé ===")
    print("\nFormat de chaque ligne :")
    print("  [replication, scenario, m, n, replication_id,")
    print("   A^k_00, ..., A^k_{m-1,n-1},  (m*n valeurs — seul A varie)")
    print("   b_0, ..., b_{m-1},            (m valeurs — FIXES = b_base)")
    print("   c_0, ..., c_{n-1}]            (n valeurs — FIXES = c_base)")


if __name__ == "__main__":
    main()
