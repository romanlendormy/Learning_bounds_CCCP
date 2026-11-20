"""test_presolve_exact_cpp.py — correspondance C++ (solver_bab/ccpo/presolve_exact.so) / Python (presolve_box_exact)
du presolve exact certifié, et port Python / C++ de l'ancien presolve.
 1. 3000 cas aléatoires + référence exacte (énumération) : C++ jamais au-dessus de v*, = v* à l'arrondi près.
 2. (optionnel) instances collectées et instances de test du papier : mêmes décisions d'élimination C++/Python
    (exact et ancien), écart relatif des bornes."""
import ctypes, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from presolve_box_exact import box_min_certified, presolve_old
from test_presolve_box_exact import exact_enum
from ccpo_dataset import load_instances_ccpo

from ccpo_libs import lib_path  # noqa: E402


def load_episode(ep):
    """Lit un épisode CCPO (generators/ccpo, instance.txt, scenarios.txt) : n, N, Q, U, B et les N matrices Σ_j."""
    l = open(ep / "instance.txt").read().split("\n")
    h = list(map(int, l[0].split())); n, N, k, Q, L, U = h[:6]; B = float(l[1].split()[0])
    Sig = []; cur = None
    for line in open(ep / "scenarios.txt"):
        t = line.split()
        if len(t) == 1:
            cur = np.zeros((n, n)); Sig.append(cur)
        elif len(t) == 3:
            i, j, v = int(t[0]) - 1, int(t[1]) - 1, float(t[2]); cur[i, j] = cur[j, i] = v
    assert len(Sig) == N, (ep, len(Sig), N)
    return n, N, Q, U, B, Sig

lib = ctypes.CDLL(str(lib_path("presolve_exact")))
P = ctypes.POINTER(ctypes.c_double)
lib.ccpo_box_min_certified_c.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double, P, P]
lib.ccpo_box_min_legacy_c.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double, P]
lib.ccpo_box_min_legacy_c.restype = ctypes.c_double

def cpp_exact(S, Q, U):
    S = np.ascontiguousarray(S, dtype=np.float64); out = np.zeros(4)
    lib.ccpo_box_min_certified_c(S.shape[0], Q, U, S.ctypes.data_as(P), out.ctypes.data_as(P)); return out
def cpp_old(S, Q, U):
    S = np.ascontiguousarray(S, dtype=np.float64)
    return lib.ccpo_box_min_legacy_c(S.shape[0], Q, U, S.ctypes.data_as(P))

ok = True
rng = np.random.default_rng(0); wl = wg = 0.0; nb = 0
for t in range(3000):
    n = int(rng.integers(2, 9)); U = int(rng.integers(1, 10)); Q = int(rng.integers(1, n * U + 1)); kind = t % 3
    A = rng.normal(size=(n, n if kind else 2)); S = A @ A.T + np.diag(rng.uniform(1e-3 if kind == 2 else 0.1, 1.0, n))
    if kind == 1: S *= rng.uniform(1, 1e4)
    v = exact_enum(S, Q, U); lo, pr, it, opt = cpp_exact(S, Q, U); sc = max(1.0, abs(v))
    wl = max(wl, (lo - v) / sc); wg = max(wg, abs(pr - v) / sc); nb += (opt != 1.0)
print(f"[1] 3000 cas aléatoires : max (lower_cpp - v*)/v* = {wl:.2e} ; max |primal_cpp - v*|/v* = {wg:.2e} ; non optimaux = {nb}")
ok &= wl <= 1e-9 and wg <= 1e-9 and nb == 0

def compare(name, items):
    global ok
    t0 = time.time(); n_sc = dis_new = dis_old = 0; rel = 0.0; t_cpp = t_old = 0.0
    for S, Q, U, B in items:
        a = time.perf_counter(); lo_c = cpp_exact(S, Q, U)[0]; t_cpp += time.perf_counter() - a
        a = time.perf_counter(); ol_c = cpp_old(S, Q, U); t_old += time.perf_counter() - a
        lo_p = box_min_certified(S, Q, U)[0]; ol_p = presolve_old(S, Q, U)
        dis_new += (lo_c > B * (1 + 1e-9)) != (lo_p > B * (1 + 1e-9)); dis_old += (ol_c > B) != (ol_p > B)
        if np.isfinite(lo_p): rel = max(rel, abs(lo_c - lo_p) / max(1.0, abs(lo_p)))
        n_sc += 1
    print(f"[2] {name}: {n_sc} scén. | désaccords exact C++/Py = {dis_new} | ancien C++/Py = {dis_old} | "
          f"écart rel. max bornes exactes = {rel:.1e} | C++ (ctypes inclus) exact {1e6*t_cpp/n_sc:.1f} µs/scén, ancien {1e6*t_old/n_sc:.1f} µs/scén "
          f"| {time.time()-t0:.0f} s", flush=True)
    ok &= dis_new == 0 and dis_old == 0 and rel <= 1e-9

# Usage : python3 test_presolve_exact_cpp.py [<dossier de collecte> <instances par fichier>] [--test_dir <dossier test_paper CCPO>]
args = [x for x in sys.argv[1:] if not x.startswith("--")]
if "--test_dir" in sys.argv:
    T = Path(sys.argv[sys.argv.index("--test_dir") + 1]); args.remove(str(T))
    items = []
    for line in open(T / "instances_list.txt"):
        _, _, _, _, path, _ = line.split(); n, N, Q, U, B, Sig = load_episode(T / path)
        items += [(S, Q, U, B) for S in Sig]
    compare("instances de test du papier", items)
if args:
    for f in sorted(Path(args[0]).glob("instances_n*_N*.txt")):
        insts = list(load_instances_ccpo(str(f)).values())[: int(args[1])]
        compare(f.name, [(i.Sigma_dense[j], i.Q, i.U, i.B) for i in insts for j in range(i.N)])
print("STATUT", "OK" if ok else "ECHEC")
