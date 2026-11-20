"""
Générateur d'instances MKP stochastiques selon la procédure de Pisinger.

Étape 1 — Instance déterministe (procédure de Pisinger 2004/2005)
-----------------------------------------------------------------
Pour n items, m contraintes, tightness α, range R, corrélation c :

  Poids :  w_{i,j} ~ U[1, R]   ∀ i ∈ [m], j ∈ [n]

  Profits (selon corrélation) :
    uncorrelated      : p_j ~ U[1, R]
    weakly_correlated : p_j ~ U[max(1, w̄_j − R/10),  w̄_j + R/10]
                        où w̄_j = (1/m) Σ_i w_{i,j}
    strongly_correlated : p_j = round(w̄_j + R/10)

  Capacités : b_i = floor(α × Σ_j w_{i,j})

Étape 2 — Perturbation stochastique (procédure Song et al. 2023)
-----------------------------------------------------------------
Identique au générateur Song : μ_j ~ Exp(0.1) tronqué [0,1],
puis pour chaque scénario k :
  - fail_j ~ Bernoulli(μ_j) → A^k_{:,j} = 0  si fail
  - sinon A^k_{i,j} ~ N(A_{i,j}, (0.1 A_{i,j})²), projeté R+

Instances du B&B : generate_song_tempfile écrit une instance complète (n, m, N scénarios) au format Song lu par
les binaires MKP ; c'est elle qu'utilisent la collecte (collect/mkp/) et les instances de test (test_paper/mkp/).

Sortie de la CLI (fichiers OR-Library et jeux de scénarios, hors chaîne de l'article)
------
  data/mkp/pisinger/
      pisinger-n{n}-m{m}-a{alpha_pct}-{corr}-r{seed}.txt   ← format OR-Library
  data/mkp/datasets-pisinger/
      pisinger-n{n}-m{m}-a{alpha_pct}-{corr}-r{seed}-S{S}.txt
      pisinger-n{n}-m{m}-a{alpha_pct}-{corr}-r{seed}-mu.txt

Utilisation
-----------
  python generators/mkp/pisinger.py --n 20 --m 5 --alpha 0.5
  python generators/mkp/pisinger.py --n 20 --m 5 --alpha 0.5 \\
      --correlation weakly_correlated --S_sizes 100 500 --n_reps 5

Sweep rapide (plusieurs n, m ou alpha) :
  python generators/mkp/pisinger.py \\
      --n 10 20 30 --m 3 5 --alpha 0.25 0.5 0.75
"""

import argparse
import os
import sys
import tempfile

import numpy as np


# ---------------------------------------------------------------------------
# Génération déterministe — Pisinger
# ---------------------------------------------------------------------------

def generate_pisinger_instance(n, m, alpha=0.5, R=1000,
                                correlation="uncorrelated", seed=42):
    """Génère une instance MKP selon la procédure de Pisinger.

    Retourne un dict {name, n, m, c, A, b, alpha, R, correlation}.
    """
    rng = np.random.default_rng(seed)

    # Poids : w[i, j] = poids de l'item j dans la contrainte i
    A = rng.integers(1, R + 1, size=(m, n)).astype(float)

    # Profits selon le type de corrélation
    if correlation == "uncorrelated":
        c = rng.integers(1, R + 1, size=n).astype(float)

    elif correlation == "weakly_correlated":
        w_bar = A.mean(axis=0)                      # poids moyen par item
        lo = np.maximum(1.0, w_bar - R / 10)
        hi = w_bar + R / 10
        c = np.array([
            rng.uniform(lo[j], max(hi[j], lo[j] + 1))
            for j in range(n)
        ], dtype=float)

    elif correlation == "strongly_correlated":
        w_bar = A.mean(axis=0)
        c = np.round(w_bar + R / 10).astype(float)
        c = np.maximum(c, 1.0)

    else:
        raise ValueError(f"correlation inconnue : {correlation!r}  "
                         f"(uncorrelated | weakly_correlated | strongly_correlated)")

    # Capacités : floor(alpha × Σ_j w_{i,j})
    b = np.floor(alpha * A.sum(axis=1))
    b = np.maximum(b, 1.0)   # capacité minimale = 1

    alpha_pct = int(round(alpha * 100))
    name = f"pisinger-n{n}-m{m}-a{alpha_pct}-{correlation[:2]}-r{seed}"

    return dict(name=name, n=n, m=m, c=c, A=A, b=b,
                alpha=alpha, R=R, correlation=correlation)


def save_orlib(inst, output_dir):
    """Sauvegarde l'instance au format OR-Library (1 instance, K=1)."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, inst["name"] + ".txt")
    n, m = inst["n"], inst["m"]

    def fmt_row(values, per_line=10):
        tokens = [str(int(v)) if v == int(v) else f"{v:.4f}" for v in values]
        return "\n".join(
            " ".join(tokens[i:i + per_line])
            for i in range(0, len(tokens), per_line)
        )

    with open(path, "w") as f:
        f.write("1\n")
        f.write(f"{n} {m} 0\n")          # opt = 0 (inconnu)
        f.write(fmt_row(inst["c"]) + "\n")
        for row in inst["A"]:
            f.write(fmt_row(row) + "\n")
        f.write(fmt_row(inst["b"]) + "\n")

    return path


# ---------------------------------------------------------------------------
# Perturbation stochastique — Song et al.
# ---------------------------------------------------------------------------

def draw_mu(n, rng):
    """μ_j ~ Exp(mean=0.1) tronqué à [0, 1]."""
    return np.minimum(rng.exponential(scale=0.1, size=n), 1.0)


def generate_scenario(A_base, mu, rng):
    """Génère A^k selon la procédure Song."""
    m, n = A_base.shape
    A_k  = np.zeros((m, n))
    fail = rng.binomial(1, mu, size=n).astype(bool)
    appear = ~fail
    if appear.any():
        A_ap  = A_base[:, appear]
        noise = rng.normal(0.0, 0.1 * A_ap)
        A_k[:, appear] = np.maximum(A_ap + noise, 0.0)
    return A_k


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


# ---------------------------------------------------------------------------
# Sortie de la CLI
# ---------------------------------------------------------------------------

def generate_all_splits(inst, mu, S_sizes, n_reps, seed_base, output_dir):
    """Génère les fichiers stochastiques pour tous les S_sizes × n_reps."""
    os.makedirs(output_dir, exist_ok=True)
    name = inst["name"]
    m, n = inst["m"], inst["n"]

    for S in S_sizes:
        outpath = os.path.join(output_dir, f"ccmk-song-{name}-S{S}.txt")
        total   = 0

        with open(outpath, "w") as f:
            for rep in range(n_reps):
                rng   = np.random.default_rng(seed_base + rep * 10_000 + S)
                b_flat = [round(v, 4) for v in inst["b"]]
                c_flat = [round(v, 4) for v in inst["c"]]

                for scen_id in range(S):
                    A_k    = generate_scenario(inst["A"], mu, rng)
                    A_flat = [round(v, 4) for v in A_k.flatten()]
                    header = [rep, scen_id, m, n, rep]
                    f.write(",".join(str(v) for v in header + A_flat + b_flat + c_flat) + "\n")
                    total += 1

        print(f"  -> {outpath}  ({n_reps} réplications × {S} scénarios = {total} lignes)")


def save_mu(inst, mu, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    mu_path = os.path.join(output_dir, f"ccmk-song-{inst['name']}-mu.txt")
    with open(mu_path, "w") as f:
        f.write(f"# instance={inst['name']}  n={inst['n']}\n")
        for j, v in enumerate(mu):
            f.write(f"{j},{v:.8f}\n")
    print(f"  -> {mu_path}")


# ---------------------------------------------------------------------------
# Entrée principale
# ---------------------------------------------------------------------------

def run_one(n, m, alpha, R, correlation, S_sizes, n_reps, seed,
            orlib_dir, dataset_dir, verbose=True):
    """Génère une instance complète (déterministe + stochastique)."""
    inst = generate_pisinger_instance(n, m, alpha, R, correlation, seed)

    if verbose:
        print(f"\n=== {inst['name']} ===")
        print(f"    n={n}, m={m}, α={alpha}, R={R}, corr={correlation}")
        W_max = int(inst["b"].max())
        print(f"    W_max={W_max}  (capacité maximale)")
        print(f"    Σ_j p_j = {inst['c'].sum():.0f},  "
              f"min/max p = {inst['c'].min():.0f}/{inst['c'].max():.0f}")

    orlib_path = save_orlib(inst, orlib_dir)
    if verbose:
        print(f"  -> {orlib_path}  (format OR-Library)")

    rng_mu = np.random.default_rng(seed)
    mu     = draw_mu(n, rng_mu)
    if verbose:
        print(f"  μ_j : min={mu.min():.4f}  moy={mu.mean():.4f}  max={mu.max():.4f}")

    save_mu(inst, mu, dataset_dir)
    generate_all_splits(inst, mu, S_sizes, n_reps, seed, dataset_dir)

    return inst


def main():
    parser = argparse.ArgumentParser(
        description="Génère des instances MKP (Pisinger) + perturbation stochastique (Song).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n",    nargs="+", type=int,   required=True,
                        help="Nombre(s) d'items (ex: --n 20  ou  --n 10 20 30)")
    parser.add_argument("--m",    nargs="+", type=int,   required=True,
                        help="Nombre(s) de contraintes (ex: --m 5  ou  --m 3 5 10)")
    parser.add_argument("--alpha", nargs="+", type=float, default=[0.5],
                        help="Tightness α = b_i / Σ w_{i,j}  (défaut: 0.5)")
    parser.add_argument("--R",    type=int,   default=1000,
                        help="Range [1, R] pour profits et poids  (défaut: 1000)")
    parser.add_argument("--correlation", nargs="+",
                        default=["uncorrelated"],
                        choices=["uncorrelated", "weakly_correlated",
                                 "strongly_correlated"],
                        help="Type(s) de corrélation profit/poids  (défaut: uncorrelated)")
    parser.add_argument("--S_sizes", nargs="+", type=int, default=[100],
                        help="Tailles de scénarios (défaut: 100)")
    parser.add_argument("--n_reps", type=int, default=5,
                        help="Réplications Song par taille S  (défaut: 5)")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Graine de base  (défaut: 42)")
    parser.add_argument("--orlib_dir", default="data/mkp/pisinger",
                        help="Dossier des instances OR-Library générées")
    parser.add_argument("--dataset_dir", default="data/mkp/datasets-pisinger",
                        help="Dossier des datasets stochastiques")
    args = parser.parse_args()

    total = 0
    for n in args.n:
        for m in args.m:
            for alpha in args.alpha:
                for corr in args.correlation:
                    run_one(
                        n=n, m=m, alpha=alpha, R=args.R,
                        correlation=corr,
                        S_sizes=args.S_sizes,
                        n_reps=args.n_reps,
                        seed=args.seed,
                        orlib_dir=args.orlib_dir,
                        dataset_dir=args.dataset_dir,
                    )
                    total += 1

    print(f"\n=== Terminé : {total} instance(s) générée(s) ===")


if __name__ == "__main__":
    main()
