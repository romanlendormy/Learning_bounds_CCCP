"""
Générateur d'instances CCSP satisfaisant la condition structurelle de la Proposition 4
(Ahmed et al. 2016).

Condition SUFFISANTE (globale, pas seulement par ligne) :
  Pour tout scénario i, il existe un support global S ⊆ {0,...,n-1} avec |S| ≤ 2 tel que :
  - b^i_j = 1  →  supp(A^i_j) ⊆ S  (toutes les lignes actives partagent le même univers)
  - b^i_j = 0  →  A^i_j = 0         (lignes inactives nulles)

Pourquoi c'est suffisant :
  Soit x ∈ {0,1}^n. Posons s = Σ_{k∈S} x_k ∈ {0,1,2}.
  Pour toute ligne active j (b_j=1), A^i_j x = Σ_{k∈supp(A^i_j)} x_k ≤ s ≤ 2.
  - Si s = |supp(A^i_j)| (tous couverts) : G_j = 0.
  - Si s > |supp(A^i_j)| : impossible (supp ⊆ S avec au moins 1 col dans S).
  En fait : si x couvre tous les cols de S (s=|S|), toutes les lignes actives sont couvertes
  (G ≤ 0), donc G ∈ Z^m_-. Sinon, pour chaque ligne active j :
    A^i_j x ∈ {0,1} (car supp(A^i_j) ⊆ S et |S|≤2, et x binaire)
  donc G_j = b_j - A^i_j x ∈ {0,1}, ce qui donne G ∈ {0,1}^m.
  Les lignes inactives (b_j=0, A_j=0) donnent G_j=0 dans tous les cas.

Format de sortie identique au générateur original :
  [episode, scenario, nb_constraints, nb_items, pA, pB, A_ijk..., b_ik...]
"""

import numpy as np
import os


def generate_scenario_prop4(m, n, pB, rng):
    """
    Génère un scénario (A^i, b^i) satisfaisant la condition GLOBALE de Prop 4.

    Algorithme :
      1. Tirer b^i ~ Bernoulli(pB), garantir au moins un 1.
      2. Tirer un support global S ⊆ {0,...,n-1} de taille 1 ou 2.
      3. Pour chaque ligne active (b_j=1) : choisir 1 ou 2 colonnes dans S.
      4. Pour chaque ligne inactive (b_j=0) : A_j = 0 (zéro).
    """
    # --- Génération de b^i ---
    b = rng.binomial(1, pB, size=(m,))
    if b.sum() == 0:
        b[rng.integers(0, m)] = 1

    # --- Support global S de taille 1 ou 2 ---
    s_size = int(rng.integers(1, 3))   # 1 ou 2
    S = rng.choice(n, size=s_size, replace=False)

    # --- Génération de A^i avec supp(A_j) ⊆ S pour lignes actives ---
    A = np.zeros((m, n), dtype=int)
    for j in range(m):
        if b[j] == 1:
            # Choisir 1 à s_size colonnes dans S (au moins 1)
            k_ones = int(rng.integers(1, s_size + 1))
            cols = rng.choice(S, size=k_ones, replace=False)
            A[j, cols] = 1
        # else: A_j reste nul

    return A, b


def verify_prop4(A, b):
    """
    Vérifie la condition GLOBALE de Prop 4 :
      - lignes inactives (b_j=0) : A_j = 0
      - lignes actives  (b_j=1) : leurs supports sont tous contenus dans un S commun |S|≤2

    Retourne True si OK, False sinon.
    """
    global_support = set()
    for j in range(len(b)):
        if b[j] == 0:
            if int(A[j].sum()) > 0:
                return False
        else:
            cols = set(int(k) for k in np.where(A[j] > 0)[0])
            global_support |= cols
    return len(global_support) <= 2


def generate_episode_prop4(episode_id, nb_scenarios, nb_constraints, nb_items,
                            pB_range, rng):
    """
    Génère un épisode complet. pB tiré une seule fois par épisode.
    pA est stocké comme valeur indicative (densité effective de A).
    """
    pB = rng.uniform(*pB_range)

    lines = []
    for scenario_id in range(nb_scenarios):
        A, b = generate_scenario_prop4(nb_constraints, nb_items, pB, rng)

        assert verify_prop4(A, b), "BUG : scénario généré ne respecte pas Prop 4"

        # Densité effective de A (pour compatibilité format)
        pA_eff = round(float(A.mean()), 6)

        header = [episode_id, scenario_id, nb_constraints, nb_items,
                  pA_eff, round(pB, 6)]
        A_flat = A.flatten().tolist()
        b_flat = b.tolist()

        lines.append(header + A_flat + b_flat)

    return lines


def generate_dataset_prop4(filepath, nb_episodes, nb_scenarios,
                            nb_constraints, nb_items,
                            pB_range=(0.3, 0.7), seed=None):
    """
    Génère un dataset satisfaisant Prop 4 et l'écrit dans filepath.
    Même format CSV que le générateur original.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    total_lines = 0
    with open(filepath, 'w') as f:
        for ep in range(nb_episodes):
            lines = generate_episode_prop4(
                episode_id=ep,
                nb_scenarios=nb_scenarios,
                nb_constraints=nb_constraints,
                nb_items=nb_items,
                pB_range=pB_range,
                rng=rng,
            )
            for line in lines:
                f.write(",".join(str(v) for v in line) + "\n")
            total_lines += len(lines)

    print(f"  -> {filepath}  "
          f"({nb_episodes} épisodes × {nb_scenarios} scénarios = {total_lines} lignes)")


if __name__ == "__main__":
    print("=== Génération des datasets CCSP — Prop 4 ===\n")

    # small : exhaustive check possible (n=20 → 2^20 ≈ 1M)
    generate_dataset_prop4(
        filepath="data/covset/datasets/test/ccsp-test-prop4-small.txt",
        nb_episodes=10,
        nb_scenarios=5,
        nb_constraints=5,
        nb_items=20,
        pB_range=(0.3, 0.7),
        seed=42,
    )

    # medium : n=50, plus de scénarios
    generate_dataset_prop4(
        filepath="data/covset/datasets/test/ccsp-test-prop4-medium.txt",
        nb_episodes=5,
        nb_scenarios=20,
        nb_constraints=10,
        nb_items=50,
        pB_range=(0.3, 0.7),
        seed=43,
    )

    # large : n=100, beaucoup de scénarios
    generate_dataset_prop4(
        filepath="data/covset/datasets/test/ccsp-test-prop4-large.txt",
        nb_episodes=3,
        nb_scenarios=50,
        nb_constraints=15,
        nb_items=100,
        pB_range=(0.3, 0.7),
        seed=44,
    )

    print("\n=== Terminé ===")
    print("\nCondition Prop 4 garantie (globale) :")
    print("  Toutes les lignes actives d'un scénario partagent un support S ⊆ {0,...,n-1}, |S|≤2")
    print("  Lignes inactives (b_j=0) : A_j = 0")
