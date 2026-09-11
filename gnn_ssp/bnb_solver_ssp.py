"""
bnb_solver_ssp.py — Wrapper ctypes pour SSPShiftProblem::solve_theta/solve_zeta.

Miroir de gnn/bnb_solver.py (MKP), mais solve_zeta y dépend d'OR-Tools CP-SAT
(coût NP-difficile en pratique — cf. benchmark manuel, ~11s pour E=8,T=10 même
avec un DFA jouet à 3 états) : contrairement au B&B maison du MKP, le lien
nécessite OR-Tools. Compile ssp_sld_wrapper.cpp automatiquement si le .so est
absent ou plus récent.

Utilisé par lagrangian_sld_ssp() dans train_SLD_ssp.py.
"""

import ctypes
import os
import subprocess
import numpy as np

_DIR      = os.path.dirname(os.path.abspath(__file__))
_REPO     = os.path.dirname(_DIR)
_SO_PATH  = os.path.join(_DIR, "_ssp_sld_wrapper.so")
_CPP_PATH = os.path.join(_DIR, "ssp_sld_wrapper.cpp")

_SOLVER_GENERIC = os.path.join(_REPO, "solver_generic")
_BUILD_CC       = os.path.join(_SOLVER_GENERIC, "build_cc")
_ORTOOLS        = os.path.join(_REPO, "external", "ortools")

_CXX = "/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/x86_64-pc-linux-gnu/gcc-bin/12/c++"
if not os.path.isfile(_CXX):
    _CXX = "g++"  # fallback hors Compute Canada (Nibi/Rorqual)

_lib = None


def _needs_recompile() -> bool:
    if not os.path.exists(_SO_PATH):
        return True
    return os.path.getmtime(_CPP_PATH) > os.path.getmtime(_SO_PATH)


def _compile():
    lib_static = os.path.join(_BUILD_CC, "libsld_generic_core.a")
    ortools_so = os.path.join(_ORTOOLS, "lib", "libortools.so.9.6.2534")
    if not os.path.isfile(lib_static):
        raise RuntimeError(
            f"{lib_static} introuvable — compiler d'abord solver_generic "
            f"(cf. compile_cc.sh, target sld_generic_core)."
        )
    if not os.path.isfile(ortools_so):
        raise RuntimeError(f"{ortools_so} introuvable.")

    cmd = [
        _CXX, "-O3", "-DNDEBUG", "-std=gnu++17", "-march=native", "-fwrapv",
        "-shared", "-fPIC",
        "-DHAVE_CONFIG_H", "-DOR_TOOLS_AS_DYNAMIC_LIB", "-DUSE_BOP", "-DUSE_CBC",
        "-DUSE_CLP", "-DUSE_GLOP", "-DUSE_LP_PARSER", "-DUSE_PDLP", "-DUSE_SCIP",
        "-I", _SOLVER_GENERIC,
        "-I", os.path.join(_SOLVER_GENERIC, "ssp"),
        "-isystem", os.path.join(_ORTOOLS, "include"),
        "-isystem", os.path.join(_ORTOOLS, "include", "coin"),
        "-isystem", os.path.join(_ORTOOLS, "include", "eigen3"),
        _CPP_PATH, "-o", _SO_PATH,
        "-Wl,-rpath," + os.path.join(_ORTOOLS, "lib"),
        lib_static, ortools_so, "-ldl", "-lpthread", "-lrt",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Compilation de ssp_sld_wrapper.cpp échouée :\n{result.stderr}")


def _load() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    if _needs_recompile():
        _compile()

    ortools_lib_dir = os.path.join(_ORTOOLS, "lib")
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if ortools_lib_dir not in ld_path.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = ortools_lib_dir + os.pathsep + ld_path

    lib = ctypes.CDLL(_SO_PATH)

    lib.ssp_sld_create.restype  = ctypes.c_void_p
    lib.ssp_sld_create.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_int),   # n_trans, trans_flat
        ctypes.c_int, ctypes.POINTER(ctypes.c_int),   # n_acc, accepting
        ctypes.c_int,                                 # initial
        ctypes.c_int, ctypes.POINTER(ctypes.c_double),# T, salary
        ctypes.c_int, ctypes.POINTER(ctypes.c_int),   # N, demand_flat
        ctypes.POINTER(ctypes.c_double),               # p
        ctypes.c_double, ctypes.c_int,                # epsilon, E
        ctypes.POINTER(ctypes.c_int),                  # mask_flat (E*T,) -1=libre
    ]
    lib.ssp_sld_destroy.argtypes = [ctypes.c_void_p]
    lib.ssp_sld_n_vars.restype   = ctypes.c_int
    lib.ssp_sld_n_vars.argtypes  = [ctypes.c_void_p]

    lib.ssp_sld_solve_theta.restype  = ctypes.c_double
    lib.ssp_sld_solve_theta.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    lib.ssp_sld_solve_zeta.restype  = ctypes.c_double
    lib.ssp_sld_solve_zeta.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]

    _lib = lib
    return lib


class SSPShiftProblemPy:
    """
    Wrapper Python d'une instance SSPShiftProblem (DFA + salaire + demande).

    Construction coûteuse relativement (déroule le DAG des transitions) mais
    faite une seule fois par instance — réutiliser le même objet pour tous
    les appels solve_theta/solve_zeta d'un même graphe B&B (une seule instance
    de base par inst_id, indépendante du nœud/masque).
    """

    def __init__(self, transitions, accepting, initial, salary, demand, p, epsilon, E,
                 mask=None):
        """
        mask : (E,T) int, -1=libre sinon symbole imposé (0=O,1=P,2=R,3=W).
               None = tout libre (équivalent SSPShiftProblem non masqué —
               NE PAS utiliser pour un vrai nœud B&B, seulement pour du
               sanity-check hors-branchement).
        """
        lib = _load()
        n_trans = len(transitions)
        trans_flat = (ctypes.c_int * (3 * n_trans))(
            *[v for tri in transitions for v in tri])
        n_acc = len(accepting)
        acc_arr = (ctypes.c_int * n_acc)(*accepting)

        T = len(salary)
        salary_arr = (ctypes.c_double * T)(*[float(s) for s in salary])

        N = len(demand)
        demand_flat = (ctypes.c_int * (N * T))(
            *[int(v) for row in demand for v in row])
        p_arr = (ctypes.c_double * N)(*[float(v) for v in p])

        if mask is None:
            mask_flat = (ctypes.c_int * (E * T))(*([-1] * (E * T)))
        else:
            mask_flat = (ctypes.c_int * (E * T))(
                *[int(v) for row in mask for v in row])

        self._lib = lib
        self._handle = lib.ssp_sld_create(
            n_trans, trans_flat, n_acc, acc_arr, int(initial),
            T, salary_arr, N, demand_flat, p_arr, float(epsilon), int(E),
            mask_flat)
        self.n_vars = lib.ssp_sld_n_vars(self._handle)

    def solve_theta(self, cost: np.ndarray) -> tuple[float, np.ndarray]:
        """cost : (n_vars,) float64. Retourne (valeur, x (n_vars,) float64 0/1)."""
        cost_c = (ctypes.c_double * self.n_vars)(*cost.astype(np.float64))
        x_out  = (ctypes.c_double * self.n_vars)()
        val = self._lib.ssp_sld_solve_theta(self._handle, cost_c, x_out)
        return val, np.frombuffer(x_out, dtype=np.float64).copy()

    def solve_zeta(self, i: int, cost: np.ndarray) -> tuple[float, np.ndarray]:
        """cost : (n_vars,) float64. Retourne (valeur, x (n_vars,) float64 0/1)."""
        cost_c = (ctypes.c_double * self.n_vars)(*cost.astype(np.float64))
        x_out  = (ctypes.c_double * self.n_vars)()
        val = self._lib.ssp_sld_solve_zeta(self._handle, int(i), cost_c, x_out)
        return val, np.frombuffer(x_out, dtype=np.float64).copy()

    def __del__(self):
        try:
            if getattr(self, "_handle", None) is not None:
                self._lib.ssp_sld_destroy(self._handle)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# DLD — miroir de ce qui précède, ciblant SSPDLDProblemBnB (ssp_dld_wrapper.cpp)
# ─────────────────────────────────────────────────────────────────────────────

_SO_PATH_DLD  = os.path.join(_DIR, "_ssp_dld_wrapper.so")
_CPP_PATH_DLD = os.path.join(_DIR, "ssp_dld_wrapper.cpp")

_lib_dld = None


def _needs_recompile_dld() -> bool:
    if not os.path.exists(_SO_PATH_DLD):
        return True
    return os.path.getmtime(_CPP_PATH_DLD) > os.path.getmtime(_SO_PATH_DLD)


def _compile_dld():
    lib_static = os.path.join(_BUILD_CC, "libsld_generic_core.a")
    ortools_so = os.path.join(_ORTOOLS, "lib", "libortools.so.9.6.2534")
    if not os.path.isfile(lib_static):
        raise RuntimeError(f"{lib_static} introuvable — compiler d'abord solver_generic.")
    if not os.path.isfile(ortools_so):
        raise RuntimeError(f"{ortools_so} introuvable.")

    cmd = [
        _CXX, "-O3", "-DNDEBUG", "-std=gnu++17", "-march=native", "-fwrapv",
        "-shared", "-fPIC",
        "-DHAVE_CONFIG_H", "-DOR_TOOLS_AS_DYNAMIC_LIB", "-DUSE_BOP", "-DUSE_CBC",
        "-DUSE_CLP", "-DUSE_GLOP", "-DUSE_LP_PARSER", "-DUSE_PDLP", "-DUSE_SCIP",
        "-I", _SOLVER_GENERIC,
        "-I", os.path.join(_SOLVER_GENERIC, "ssp"),
        "-isystem", os.path.join(_ORTOOLS, "include"),
        "-isystem", os.path.join(_ORTOOLS, "include", "coin"),
        "-isystem", os.path.join(_ORTOOLS, "include", "eigen3"),
        _CPP_PATH_DLD, "-o", _SO_PATH_DLD,
        "-Wl,-rpath," + os.path.join(_ORTOOLS, "lib"),
        lib_static, ortools_so, "-ldl", "-lpthread", "-lrt",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Compilation de ssp_dld_wrapper.cpp échouée :\n{result.stderr}")


def _load_dld() -> ctypes.CDLL:
    global _lib_dld
    if _lib_dld is not None:
        return _lib_dld
    if _needs_recompile_dld():
        _compile_dld()

    ortools_lib_dir = os.path.join(_ORTOOLS, "lib")
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if ortools_lib_dir not in ld_path.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = ortools_lib_dir + os.pathsep + ld_path

    lib = ctypes.CDLL(_SO_PATH_DLD)

    lib.ssp_dld_create.restype  = ctypes.c_void_p
    lib.ssp_dld_create.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_int),
        ctypes.c_int, ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int, ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.ssp_dld_destroy.argtypes = [ctypes.c_void_p]
    lib.ssp_dld_n_vars.restype   = ctypes.c_int
    lib.ssp_dld_n_vars.argtypes  = [ctypes.c_void_p]

    lib.ssp_dld_solve_theta.restype  = ctypes.c_double
    lib.ssp_dld_solve_theta.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    lib.ssp_dld_solve_zeta.restype  = ctypes.c_double
    lib.ssp_dld_solve_zeta.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]

    _lib_dld = lib
    return lib


class SSPDLDProblemPy:
    """Wrapper Python de SSPDLDProblemBnB — miroir de SSPShiftProblemPy, avec un
    indice de période j en plus sur solve_theta/solve_zeta (décomposition DLD)."""

    def __init__(self, transitions, accepting, initial, salary, demand, p, epsilon, E,
                 mask=None):
        lib = _load_dld()
        n_trans = len(transitions)
        trans_flat = (ctypes.c_int * (3 * n_trans))(
            *[v for tri in transitions for v in tri])
        n_acc = len(accepting)
        acc_arr = (ctypes.c_int * n_acc)(*accepting)

        T = len(salary)
        salary_arr = (ctypes.c_double * T)(*[float(s) for s in salary])

        N = len(demand)
        demand_flat = (ctypes.c_int * (N * T))(
            *[int(v) for row in demand for v in row])
        p_arr = (ctypes.c_double * N)(*[float(v) for v in p])

        if mask is None:
            mask_flat = (ctypes.c_int * (E * T))(*([-1] * (E * T)))
        else:
            mask_flat = (ctypes.c_int * (E * T))(
                *[int(v) for row in mask for v in row])

        self._lib = lib
        self._handle = lib.ssp_dld_create(
            n_trans, trans_flat, n_acc, acc_arr, int(initial),
            T, salary_arr, N, demand_flat, p_arr, float(epsilon), int(E),
            mask_flat)
        self.n_vars = lib.ssp_dld_n_vars(self._handle)

    def solve_theta(self, j: int, cost: np.ndarray) -> tuple[float, np.ndarray]:
        cost_c = (ctypes.c_double * self.n_vars)(*cost.astype(np.float64))
        x_out  = (ctypes.c_double * self.n_vars)()
        val = self._lib.ssp_dld_solve_theta(self._handle, int(j), cost_c, x_out)
        return val, np.frombuffer(x_out, dtype=np.float64).copy()

    def solve_zeta(self, i: int, j: int, cost: np.ndarray) -> tuple[float, np.ndarray]:
        cost_c = (ctypes.c_double * self.n_vars)(*cost.astype(np.float64))
        x_out  = (ctypes.c_double * self.n_vars)()
        val = self._lib.ssp_dld_solve_zeta(self._handle, int(i), int(j), cost_c, x_out)
        return val, np.frombuffer(x_out, dtype=np.float64).copy()

    def __del__(self):
        try:
            if getattr(self, "_handle", None) is not None:
                self._lib.ssp_dld_destroy(self._handle)
        except Exception:
            pass
