"""test_lagrangian_majorant_validity.py — verification NUMERIQUE (petites instances, v* par force brute) de :
   (a) L(lambda) = sum_j p theta_j - (somme des min_feas plus petits p*delta_j), delta_j = max(0, theta_j - zeta_j),
       est une borne SUPERIEURE valide de v* ;
   (b) remplacer zeta_j par un MAJORANT zeta_bar >= zeta_j (ex. zeta+eps, theta) donne L_bar >= L >= v*  (valide, plus lache) ;
   (c) remplacer zeta_j par un MINORANT (ex. zeta-eps, ou la valeur d'un incumbent non optimal) peut donner L_bar < v*  (INVALIDE).
Probleme (comme ccpo_sld.hpp, racine, r=0) : v* = max c'x  s.c. x in D, #{j : x'Sigma_j x <= B_j} >= min_feas,
D = {x entier : sum=Q, |supp|<=k, x_i in {0}U[L,U]}. p_j = 1/N."""
import itertools
import numpy as np

n, Q, k, Lo, Up, N, min_feas = 6, 15, 3, 2, 8, 6, 4
rng = np.random.default_rng(0)


def gen_D():
    out = []
    def rec(i, rem, act, cur):
        if i == n:
            if rem == 0: out.append(tuple(cur))
            return
        rec(i + 1, rem, act, cur + [0])
        if act < k:
            for v in range(Lo, min(Up, rem) + 1):
                rec(i + 1, rem - v, act + 1, cur + [v])
    rec(0, Q, 0, [])
    return np.array(out, dtype=float)

X = gen_D()
print(f"|D| = {len(X)} (n={n}, Q={Q}, k={k}, L={Lo}, U={Up})")
p = np.full(N, 1.0 / N)
n_trials = n_lam = 0
viol = {"L exact < v*": 0, "majorant eps=0.5": 0, "majorant eps=5": 0, "majorant eps=50": 0, "majorant=theta": 0,
        "L_bar < L (majorant)": 0}
minor_viol = {"minorant eps=0.5": 0, "minorant eps=5": 0, "minorant eps=50": 0, "minorant=-1e6": 0}
minor_tot = 0
for trial in range(60):
    c = rng.uniform(1, 10, n)
    Sig, B = [], []
    R = np.zeros((N, len(X)))
    for j in range(N):
        A = rng.normal(size=(n, 2))
        S = A @ A.T + np.diag(rng.uniform(0.2, 1.0, n))
        R[j] = np.einsum("mi,ij,mj->m", X, S, X)
        B.append(np.quantile(R[j], rng.uniform(0.15, 0.6)))
    B = np.array(B)
    feas = R <= B[:, None] + 1e-9                       # (N,m)
    ok_count = feas.sum(axis=0) >= min_feas
    if not ok_count.any():
        continue
    vstar = (X[ok_count] @ c).max()
    n_trials += 1
    for _ in range(80):
        lam = rng.normal(scale=3.0, size=(N, n))
        cost = c[None, :] + lam / p[:, None] - lam.sum(axis=0, keepdims=True)
        val = cost @ X.T                                 # (N,m)
        theta = val.max(axis=1)
        zeta = np.where(feas.any(axis=1), np.where(feas, val, -np.inf).max(axis=1), -np.inf)

        def Lof(zb):
            delta = np.maximum(0.0, theta - zb)
            delta = np.where(np.isfinite(delta), delta, np.inf)
            d = np.sort(delta)[:min_feas]
            return (p * theta).sum() - (p[0] * d).sum()
        L = Lof(zeta); n_lam += 1
        if L < vstar - 1e-9: viol["L exact < v*"] += 1
        for name, eps in (("majorant eps=0.5", 0.5), ("majorant eps=5", 5.0), ("majorant eps=50", 50.0)):
            Lb = Lof(zeta + eps)
            if Lb < vstar - 1e-9: viol[name] += 1
            if Lb < L - 1e-9: viol["L_bar < L (majorant)"] += 1
        Lb = Lof(theta)
        if Lb < vstar - 1e-9: viol["majorant=theta"] += 1
        if Lb < L - 1e-9: viol["L_bar < L (majorant)"] += 1
        minor_tot += 1
        for name, eps in (("minorant eps=0.5", 0.5), ("minorant eps=5", 5.0), ("minorant eps=50", 50.0)):
            if Lof(zeta - eps) < vstar - 1e-9: minor_viol[name] += 1
        if Lof(np.full(N, -1e6)) < vstar - 1e-9: minor_viol["minorant=-1e6"] += 1
print(f"{n_trials} instances, {n_lam} valeurs de lambda")
print("VIOLATIONS (L < v*) attendues = 0 pour L exact et pour les majorants :")
for k_, v in viol.items(): print(f"   {k_:<26} {v}")
print(f"MINORANTS (invalides en general), violations sur {minor_tot} cas :")
for k_, v in minor_viol.items(): print(f"   {k_:<26} {v}")
