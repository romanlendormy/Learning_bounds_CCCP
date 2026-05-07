"""
Tests de validation du solveur v1_LD sur les instances covset.

Test 1 — Gap de dualité    : gap = (v1LD* − v_MIP) / |v_MIP|  doit ≈ 0
Test 2 — Vecteurs de coût  : idem avec c ~ U[1,10], confirme l'absence d'artefact
Test 3 — Taux de faisabilité : fraction de scénarios couverts par x_MIP*

Usage :
  conda run -n base python run_tests.py [--dataset small|medium|large] [--n_inst N]
"""

import argparse
import sys
import time
import numpy as np

sys.path.insert(0, ".")
from solver.instance_reader import load_instances
from solver.v1ld_solver import solve_v1LD
from solver.mip_reference import solve_mip_exact
from solver.instance_reader import CCSPInstance

# ─────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────

def _feasibility_rate(inst) -> float:
    """Fraction de scénarios i tels qu'il existe x∈{0,1}^n couvrant toutes les
    contraintes j : A^i_j x ≥ b^i_j.
    Approximé par la faisabilité du LP de couverture (relâchement linéaire).
    """
    try:
        from scipy.optimize import linprog
    except ImportError:
        return float("nan")

    n, m, N = inst.n, inst.m, inst.N
    feasible = 0
    for i in range(N):
        # LP : min 0 s.t. A[i] x ≥ b[i], 0 ≤ x ≤ 1
        # scipy.linprog attend ≤ → inverser : -A[i] x ≤ -b[i]
        res = linprog(
            np.zeros(n),
            A_ub=-inst.A[i],
            b_ub=-inst.b[i],
            bounds=[(0, 1)] * n,
            method="highs",
        )
        if res.status == 0:  # faisable
            feasible += 1
    return feasible / N


def _fmt_row(label, values, fmt=".4f"):
    vals = "  ".join(f"{v:{fmt}}" for v in values)
    return f"  {label:<20s} {vals}"


# ─────────────────────────────────────────────
# Test 1 : gap de dualité, c = ones(n)
# ─────────────────────────────────────────────

def test1_duality_gap(instances, T=200):
    print("\n=== TEST 1 — Gap de dualité (c = ones(n)) ===")
    gaps, times_mip, times_ld = [], [], []
    v_mips, v_lds = [], []
    violations = 0

    for inst in instances:
        t0 = time.perf_counter()
        v_mip, status = solve_mip_exact(inst)
        times_mip.append(time.perf_counter() - t0)

        if status != "Optimal" or np.isnan(v_mip):
            print(f"  ep {inst.episode}: MIP {status}, skip")
            continue

        t0 = time.perf_counter()
        res = solve_v1LD(inst, T=T)
        times_ld.append(time.perf_counter() - t0)

        v_ld = res["v1LD_star"]
        # Pour minimisation : borne inférieure → v_ld ≤ v_mip (normalement)
        gap = (v_mip - v_ld) / max(abs(v_mip), 1e-9)
        gaps.append(gap)
        v_mips.append(v_mip)
        v_lds.append(v_ld)

        # Vérifier que la borne est bien ≤ v_mip à chaque itération
        conv = res["convergence"]
        n_viol = sum(1 for b in conv if b > v_mip + 1e-4)
        violations += n_viol

        print(
            f"  ep {inst.episode:3d}: v_MIP={v_mip:.4f}  v1LD={v_ld:.4f}"
            f"  gap={gap:.4f}  viol={n_viol}/{T}"
        )

    if gaps:
        gaps = np.array(gaps)
        print(f"\n  ── Résumé ──")
        print(f"  Instances testées : {len(gaps)}")
        print(f"  Gap moyen         : {gaps.mean():.4f}")
        print(f"  Gap max           : {gaps.max():.4f}")
        print(f"  Gap < 1%          : {(gaps < 0.01).sum()}/{len(gaps)} ({(gaps<0.01).mean()*100:.0f}%)")
        print(f"  Gap < 5%          : {(gaps < 0.05).sum()}/{len(gaps)} ({(gaps<0.05).mean()*100:.0f}%)")
        print(f"  Violations bound  : {violations} total sur {len(gaps)*T} itérations")
        print(f"  Temps MIP moyen   : {np.mean(times_mip):.3f}s")
        print(f"  Temps v1LD moyen  : {np.mean(times_ld):.3f}s")

    return gaps


# ─────────────────────────────────────────────
# Test 2 : gap avec c ~ U[1, 10]
# ─────────────────────────────────────────────

def test2_random_costs(instances, T=200, seed=42):
    print("\n=== TEST 2 — Gap de dualité (c ~ U[1, 10]) ===")
    rng = np.random.default_rng(seed)
    gaps = []
    violations = 0

    for inst in instances:
        # Remplacer c par un vecteur aléatoire (positif → sens coût minimisation)
        inst.c = rng.uniform(1.0, 10.0, inst.n)

        v_mip, status = solve_mip_exact(inst)
        if status != "Optimal" or np.isnan(v_mip):
            continue

        res = solve_v1LD(inst, T=T)
        v_ld = res["v1LD_star"]
        gap = (v_mip - v_ld) / max(abs(v_mip), 1e-9)
        gaps.append(gap)

        conv = res["convergence"]
        n_viol = sum(1 for b in conv if b > v_mip + 1e-4)
        violations += n_viol

        print(
            f"  ep {inst.episode:3d}: v_MIP={v_mip:.4f}  v1LD={v_ld:.4f}"
            f"  gap={gap:.4f}  viol={n_viol}/{T}"
        )

    if gaps:
        gaps = np.array(gaps)
        print(f"\n  ── Résumé ──")
        print(f"  Instances testées : {len(gaps)}")
        print(f"  Gap moyen         : {gaps.mean():.4f}")
        print(f"  Gap max           : {gaps.max():.4f}")
        print(f"  Gap < 1%          : {(gaps < 0.01).sum()}/{len(gaps)} ({(gaps<0.01).mean()*100:.0f}%)")
        print(f"  Violations bound  : {violations} total")

    return gaps


# ─────────────────────────────────────────────
# Test 3 : taux de faisabilité par scénario
# ─────────────────────────────────────────────

def test3_feasibility_rate(instances):
    print("\n=== TEST 3 — Taux de faisabilité des scénarios ===")
    print("  (fraction de scénarios couverts par la relaxation LP de x∈{0,1}^n s.t. A^ix≥b^i)")

    rates = []
    for inst in instances[:10]:  # limiter pour la vitesse
        rate = _feasibility_rate(inst)
        rates.append(rate)
        needed = 1.0 - inst.epsilon
        print(
            f"  ep {inst.episode:3d}: faisabilité={rate:.3f}  "
            f"seuil requis={needed:.2f}  "
            f"{'OK' if rate >= needed else 'INFAISABLE (contrainte de chance impossible)'}"
        )

    if rates:
        rates = np.array(rates)
        print(f"\n  Taux moyen : {rates.mean():.3f}")
        print(f"  Taux min   : {rates.min():.3f}")
        print(
            f"  Interprétation : "
            + ("contrainte de chance non-triviale" if rates.mean() < 0.99 else
               "trop facile — contrainte rarement active (revoir le générateur)")
        )


def solve_lp_relaxation_ccsp(inst, fixations=None, return_sol=False):
    """
    Relaxation LP du CCSP stochastique.
    La contrainte de chance sum_k p_k * z_k >= 1 - epsilon est linéarisée :
      Pour chaque scénario k : A[k] @ x + M*(1-z_k) >= b[k]  (z_k binaire relaxé ∈ [0,1])
      sum_k p_k * z_k >= 1 - epsilon
    Variables : x ∈ [0,1]^n,  z ∈ [0,1]^N
    Objectif  : min c^T x
    """
    from scipy.optimize import linprog

    n, m, N = inst.n, inst.m, inst.N
    fixations = fixations or {}
    M = 1.0  # big-M (b est binaire ∈ {0,1}, donc M=1 suffit)

    # ── Variables : [x_0..x_{n-1}, z_0..z_{N-1}] ──────────────────────────
    n_vars = n + N

    # Objectif : min c^T x + 0^T z
    c_obj = np.concatenate([inst.c, np.zeros(N)])

    # ── Contraintes inégalité (A_ub @ v <= b_ub) ───────────────────────────
    rows_ub, rhs_ub = [], []

    # 1) Pour chaque scénario k et chaque contrainte j :
    #    -A[k,j,:] @ x - M*(1 - z_k) <=  -b[k,j]
    #    ↔  -A[k,j,:] @ x + M*z_k   <=  -b[k,j] + M
    for k in range(N):
        for j in range(m):
            row = np.zeros(n_vars)
            row[:n]     = -inst.A[k, j, :]   # -A[k,j,:]
            row[n + k]  =  M                  # +M*z_k
            rows_ub.append(row)
            rhs_ub.append(-inst.b[k, j] + M)

    # 2) Contrainte de chance : sum_k p_k * z_k >= 1 - epsilon
    #    ↔  -sum_k p_k * z_k <= -(1 - epsilon)
    row_chance = np.zeros(n_vars)
    row_chance[n:] = -inst.p
    rows_ub.append(row_chance)
    rhs_ub.append(-(1.0 - inst.epsilon))

    # 3) Fixations (branch & bound) : x_i = v  →  x_i <= v  et  -x_i <= -v
    for i, v in fixations.items():
        row_ub = np.zeros(n_vars)
        row_ub[i] = 1.0
        rows_ub.append(row_ub)
        rhs_ub.append(v)

        row_lb = np.zeros(n_vars)
        row_lb[i] = -1.0
        rows_ub.append(row_lb)
        rhs_ub.append(-v)

    A_ub = np.array(rows_ub)
    b_ub = np.array(rhs_ub)

    # Bornes : x ∈ [0,1]^n,  z ∈ [0,1]^N
    bounds = [(0, 1)] * n_vars

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if res.status != 0:
        return None

    if return_sol:
        return res.x[:n]   # seulement x

    return res.fun


def branch_and_bound_ccsp(inst):
    """
    Branch & Bound exact sur le CCSP stochastique.
    - Relaxation LP via solve_lp_relaxation_ccsp
    - Branchement sur les variables x fractionnaires
    - Retourne (v_star, info)
    """
    import math

    best_val = math.inf
    nodes    = 0
    stack    = [{}]   # pile de fixations {var_index: 0|1}

    while stack:
        fixations = stack.pop()
        nodes += 1

        lb = solve_lp_relaxation_ccsp(inst, fixations=fixations)

        if lb is None or lb >= best_val - 1e-6:
            continue  # Élagage

        x_lp = solve_lp_relaxation_ccsp(inst, fixations=fixations, return_sol=True)
        if x_lp is None:
            continue

        # Chercher la variable x la plus fractionnaire
        frac_idx  = None
        frac_dist = 0.0
        for i in range(inst.n):
            if i in fixations:
                continue
            dist = min(x_lp[i], 1 - x_lp[i])
            if dist > 1e-4 and dist > frac_dist:
                frac_dist = dist
                frac_idx  = i

        if frac_idx is None:
            # Solution entière — vérifier faisabilité de la contrainte de chance
            x_int = np.array([
                fixations.get(i, round(x_lp[i])) for i in range(inst.n)
            ])
            feasible_count = sum(
                1 for k in range(inst.N)
                if np.all(inst.A[k] @ x_int >= inst.b[k] - 1e-6)
            )
            prob_covered = np.dot(inst.p, [
                1.0 if np.all(inst.A[k] @ x_int >= inst.b[k] - 1e-6) else 0.0
                for k in range(inst.N)
            ])
            if prob_covered >= 1.0 - inst.epsilon - 1e-6:
                val = float(inst.c @ x_int)
                if val < best_val:
                    best_val = val
        else:
            # Branchement sur frac_idx (d'abord x=1 car minimisation)
            for v in (1, 0):
                stack.append({**fixations, frac_idx: v})

    return best_val, {"nodes": nodes}


def test4_bound_validation_bnb(instances, decomp = False, T=200):
    """
    TEST 4 — Validation des bornes duales via Branch & Bound (CCSP stochastique)
    Compare :
      - v_LP  : relaxation LP du CCSP (borne inférieure garantie)
      - v1LD  : borne duale par sous-gradient
      - v*    : optimum exact par Branch & Bound
    Vérifie : v_LP ≤ v* et v1LD ≤ v* (dualité faible)
    """
    print("\n=== TEST 4 — Validation des bornes via Branch & Bound ===")

    results = []
    n_valid_ld, n_valid_lp = 0, 0
    times_bnb, times_ld, times_lp = [], [], []

    for inst in instances:

        # 1. Borne LP
        t0 = time.perf_counter()
        v_lp = solve_lp_relaxation_ccsp(inst)
        times_lp.append(time.perf_counter() - t0)

        if v_lp is None:
            print(f"  ep {inst.episode:3d}: LP infaisable, skip")
            continue

        # 2. Borne v1LD
        t0 = time.perf_counter()
        res = solve_v1LD(inst, decomp=decomp,  T=T)
        times_ld.append(time.perf_counter() - t0)
        v_ld = res["v1LD_star"]

        # 3. Optimum exact B&B
        t0 = time.perf_counter()
        v_star, bnb_info = branch_and_bound_ccsp(inst)
        times_bnb.append(time.perf_counter() - t0)

        # 4. Validation
        tol = 1e-4
        ld_valid = v_ld <= v_star + tol
        lp_valid = v_lp <= v_star + tol

        gap_ld = (v_star - v_ld) / max(abs(v_star), 1e-9)
        gap_lp = (v_star - v_lp) / max(abs(v_star), 1e-9)

        n_valid_ld += int(ld_valid)
        n_valid_lp += int(lp_valid)

        print(
            f"  ep {inst.episode:3d}: "
            f"v*={v_star:.4f}  "
            f"v1LD={v_ld:.4f} (gap={gap_ld:+.4f}) {'✓' if ld_valid else '✗ VIOLATION'}  |  "
            f"LP={v_lp:.4f}  (gap={gap_lp:+.4f}) {'✓' if lp_valid else '✗ VIOLATION'}  "
            f"[B&B: {bnb_info['nodes']} nœuds, {times_bnb[-1]:.2f}s]"
        )

        results.append({
            "episode":  inst.episode,
            "v_star":   v_star,
            "v_ld":     v_ld,
            "v_lp":     v_lp,
            "gap_ld":   gap_ld,
            "gap_lp":   gap_lp,
            "ld_valid": ld_valid,
            "lp_valid": lp_valid,
            "bnb_nodes": bnb_info["nodes"],
        })

    n = len(results)
    if n == 0:
        print("  Aucune instance traitée.")
        return results

    gaps_ld = np.array([r["gap_ld"] for r in results])
    gaps_lp = np.array([r["gap_lp"] for r in results])

    print(f"\n  ── Résumé ──")
    print(f"  Instances testées       : {n}")
    print(f"  Borne v1LD valide (≤v*) : {n_valid_ld}/{n}  {'✓ OK' if n_valid_ld == n else '✗ BUG DÉTECTÉ'}")
    print(f"  Borne LP   valide (≤v*) : {n_valid_lp}/{n}  {'✓ OK' if n_valid_lp == n else '✗ BUG DÉTECTÉ'}")
    print(f"  Gap v1LD moyen          : {gaps_ld.mean():.4f}")
    print(f"  Gap LP   moyen          : {gaps_lp.mean():.4f}")
    print(f"  Gap v1LD < 1%           : {(gaps_ld < 0.01).sum()}/{n} ({(gaps_ld < 0.01).mean()*100:.0f}%)")
    print(f"  Gap v1LD < 5%           : {(gaps_ld < 0.05).sum()}/{n} ({(gaps_ld < 0.05).mean()*100:.0f}%)")
    print(f"  Nœuds B&B moyen         : {np.mean([r['bnb_nodes'] for r in results]):.1f}")
    print(f"  Temps B&B moyen         : {np.mean(times_bnb):.3f}s")
    print(f"  Temps v1LD moyen        : {np.mean(times_ld):.3f}s")
    print(f"  Temps LP  moyen         : {np.mean(times_lp):.3f}s")

    return results


#==== Debug violation borne duale v1LD sur instance micro-triviale ====

def test_micro_instance():
    """
    Test sur instance triviale à solution connue analytiquement.
    n=2, m=1, N=2, epsilon=0.5  →  v* = 1

    Scénario 0 : x_0 >= 1  →  solution minimale x=[1,0], coût=1
    Scénario 1 : x_1 >= 1  →  solution minimale x=[0,1], coût=1
    epsilon=0.5 → il suffit de couvrir 1 scénario sur 2
    v* = 1  (ex: x=[1,0] couvre scénario 0)

    ─────────────────────────────────────────────────────────────────────
    Formulation duale (Ahmed 2016, forme symétrique de Rockafellar-Wets) :
      μ ∈ R^{N×n}  avec contrainte  Σ_i p_i μ_i = 0
      θ_i(μ_i) = min { (c+μ_i)ᵀx : x ∈ {0,1}^n }
      ζ_i(μ_i) = min { (c+μ_i)ᵀx : x ∈ X^i }
      L1(μ)    = Σ_i p_i θ_i(μ_i) + min_z Σ_i p_i δ_i z_i   (h = 0)
    ─────────────────────────────────────────────────────────────────────

    Valeurs attendues à t=0 (μ = 0) :
      θ_i           = 0.0        ∀i  (x=[0,0])
      ζ_0           = 1.0        (x=[1,0])
      ζ_1           = 1.0        (x=[0,1])
      δ             = [1.0, 1.0]
      knapsack_val  = 0.5        (z_0=1, couvre 50% ≥ 1-ε)
      L1            = 0 + 0.5 = 0.5    (doit être ≤ v*=1 ✓)

    Optimum analytique : μ_1 = (1,-1), μ_2 = (-1,1)  → L1 = 1 = v*
    """
    from solver.instance_reader import CCSPInstance
    from solver.v1ld_solver import (
        solve_theta_i,
        solve_zeta_i_exact,
        solve_z_knapsack_min,
        project_mu,
    )

    print("\n=== TEST MICRO-INSTANCE (n=2, m=1, N=2, ε=0.5) ===")
    print("    v* attendu = 1.0\n")

    instance = CCSPInstance(
        episode=0,
        n=2, m=1, N=2, epsilon=0.5,
        c=np.array([1.0, 1.0]),
        A=np.array([[[1.0, 0.0]], [[0.0, 1.0]]]),  # (N, m, n)
        b=np.array([[1.0], [1.0]]),                 # (N, m)
        p=np.array([0.5, 0.5]),
    )

    n, m, N       = instance.n, instance.m, instance.N
    c, p, epsilon = instance.c, instance.p, instance.epsilon
    A, b          = instance.A, instance.b

    # μ ∈ R^{N×n}, initialisé à 0 (qui satisfait Σ p_i μ_i = 0)
    mu = np.zeros((N, n))

    all_valid = True

    for t in range(10):
        step = 1.0 / (t + 1)

        # ── (A) θ_i et ζ_i pour chaque scénario ──────────────────────────
        theta_vals = np.empty(N)
        zeta_vals  = np.empty(N)
        x_theta    = np.empty((N, n))
        x_zeta     = np.empty((N, n))

        for i in range(N):
            cost_i = c + mu[i]
            theta_vals[i], x_theta[i] = solve_theta_i(cost_i)
            zeta_vals[i],  x_zeta[i]  = solve_zeta_i_exact(cost_i, A[i], b[i])

        # ── (B) Knapsack sur z ───────────────────────────────────────────
        delta = np.maximum(0.0, zeta_vals - theta_vals)
        z_star, knapsack_val = solve_z_knapsack_min(delta, p, epsilon)

        # ── (C) Borne L1(μ) — h = 0, donc pas de terme à soustraire ─────
        L1 = float(p @ theta_vals) + knapsack_val

        # ── (D) Sous-gradient ────────────────────────────────────────────
        # x_i* = arg ζ_i si z_i* = 1, sinon arg θ_i
        x_star = np.where(z_star[:, np.newaxis] > 0.5, x_zeta, x_theta)
        x_bar  = p @ x_star
        g      = x_star - x_bar[np.newaxis, :]   # (N, n)

        # ── (E) Update + projection sur {Σ p_i μ_i = 0} ─────────────────
        mu = mu + step * g
        mu = project_mu(mu, p)

        valid = L1 <= 1.0 + 1e-6
        if not valid:
            all_valid = False

        # ── Affichage ────────────────────────────────────────────────────
        print(f"  t={t}")
        for i in range(N):
            print(f"    μ[{i}]        = {mu[i].round(4)}")
        print(f"    Σ p_i μ_i    = {(p @ mu).round(6)}   (doit être ≈ 0)")
        for i in range(N):
            print(f"    x_θ[{i}]      = {x_theta[i]}   θ_{i} = {theta_vals[i]:.4f}")
        for i in range(N):
            print(f"    x_ζ[{i}]      = {x_zeta[i]}   ζ_{i} = {zeta_vals[i]:.4f}")
        print(f"    δ            = {delta.round(4)}")
        print(f"    z*           = {z_star}   knapsack = {knapsack_val:.4f}")
        print(f"    Σ p_i θ_i    = {p @ theta_vals:.4f}")
        print(f"    x_bar        = {x_bar.round(4)}")
        print(f"    L1 (borne)   = {L1:.4f}   v*=1.0   {'✓' if valid else '✗ VIOLATION'}")
        print(f"    ||g||        = {np.linalg.norm(g):.4f}")
        print()

        if t == 0:
            checks = [
                ("θ_0       == 0.0",     abs(theta_vals[0] - 0.0) < 1e-6),
                ("θ_1       == 0.0",     abs(theta_vals[1] - 0.0) < 1e-6),
                ("ζ_0       == 1.0",     abs(zeta_vals[0]  - 1.0) < 1e-6),
                ("ζ_1       == 1.0",     abs(zeta_vals[1]  - 1.0) < 1e-6),
                ("δ         == [1, 1]",  np.allclose(delta, [1.0, 1.0])),
                ("knapsack  == 0.5",     abs(knapsack_val  - 0.5) < 1e-6),
                ("Σ p_i θ_i == 0.0",     abs(p @ theta_vals)      < 1e-6),
                # À t=0, z* = (1, 0) donc x_0* = arg ζ_0 = (1,0)
                # mais x_1* = arg θ_1 = (0,0) (pas (0,1) !).
                # → x_bar = 0.5·(1,0) + 0.5·(0,0) = (0.5, 0)
                ("x_bar     == [.5, 0]", np.allclose(x_bar, [0.5, 0.0])),
                ("L1        == 0.5",     abs(L1 - 0.5)            < 1e-6),
            ]
            print("  Vérifications analytiques à t=0 :")
            for name, ok in checks:
                print(f"    {name:30s} {'✓' if ok else '✗ ECHEC'}")
                if not ok:
                    all_valid = False
            print()

    # ── Vérification finale : convergence vers l'optimum dual ───────────
    print("  Vérification : la trajectoire L1 doit rester ≤ v* = 1")
    print(f"    → toutes les itérations conformes : {'✓' if all_valid else '✗'}")

    print(f"\n  ── Résultat : {'✓ PASS' if all_valid else '✗ FAIL — bug détecté'}\n")
    return all_valid

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["micro","small", "medium", "large"], default="small")
    parser.add_argument("--n_inst", type=int, default=10, help="nombre d'instances à tester")
    parser.add_argument("--decomp", action="store_true", help="utiliser la décomposition dans test 4 (v1LD_direct)")
    parser.add_argument("--T", type=int, default=200, help="itérations sous-gradient")
    parser.add_argument("--split", choices=["test", "val", "train"], default="test")
    args = parser.parse_args()

    filepath = f"data/covset/datasets/{args.split}/ccsp-{args.split}-{args.dataset}.txt"
    print(f"Chargement : {filepath}  ({args.n_inst} instances, T={args.T})")

    instances = load_instances(filepath, epsilon=0.1, max_episodes=args.n_inst)
    print(f"  {len(instances)} instance(s) chargée(s) : n={instances[0].n}, m={instances[0].m}, N={instances[0].N}")

    # Copie pour test 2 (on modifie c in-place dans test 2)
    import copy
    instances_t2 = copy.deepcopy(instances)

    #test1_duality_gap(instances, T=args.T)
    test4_bound_validation_bnb(instances, decomp=args.decomp, T=args.T)
    #test_micro_instance()
    #test2_random_costs(instances_t2, T=args.T)
    #test3_feasibility_rate(instances)
