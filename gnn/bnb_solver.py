"""
bnb_solver.py — Wrapper ctypes pour solve_zeta_batch (C++).

Compile bnb_solver.cpp automatiquement si le .so est absent ou plus récent.
Utilisé par lagrangian_v1ld() dans train_gnn_v1ld.py.
"""

import ctypes
import os
import subprocess
import numpy as np

_DIR     = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_DIR, "_bnb_solver_lib.so")
_CPP_PATH = os.path.join(_DIR, "bnb_solver.cpp")

_lib = None


def _needs_recompile() -> bool:
    if not os.path.exists(_SO_PATH):
        return True
    return os.path.getmtime(_CPP_PATH) > os.path.getmtime(_SO_PATH)


def _compile():
    result = subprocess.run(
        ["g++", "-O3", "-std=c++17", "-shared", "-fPIC", "-fopenmp",
         "-o", _SO_PATH, _CPP_PATH],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        # Fallback sans OpenMP si le compilateur ne le supporte pas
        result = subprocess.run(
            ["g++", "-O3", "-std=c++17", "-shared", "-fPIC",
             "-o", _SO_PATH, _CPP_PATH],
            capture_output=True, text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Compilation de bnb_solver.cpp échouée :\n{result.stderr}"
        )


def _load() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    if _needs_recompile():
        _compile()
    lib = ctypes.CDLL(_SO_PATH)
    lib.solve_zeta_batch.restype = None
    lib.solve_zeta_batch.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),  # cost
        ctypes.POINTER(ctypes.c_double),  # A
        ctypes.POINTER(ctypes.c_double),  # b_res
        ctypes.POINTER(ctypes.c_double),  # x_out
    ]
    _lib = lib
    return _lib


def solve_zeta_batch(
    cost_np:  np.ndarray,   # (N_z, n_g) float64
    A_np:     np.ndarray,   # (N_z, m, n_g) float64
    b_res_np: np.ndarray,   # (N_z, m) float64
) -> np.ndarray:            # (N_z, n_g) float32
    """Résout N_z sous-problèmes ζ (knapsack B&B) en C++."""
    lib = _load()
    N_z, n_g = cost_np.shape
    m = b_res_np.shape[1]

    cost_c  = np.ascontiguousarray(cost_np,  dtype=np.float64)
    A_c     = np.ascontiguousarray(A_np,     dtype=np.float64)
    b_res_c = np.ascontiguousarray(b_res_np, dtype=np.float64)
    x_out   = np.zeros((N_z, n_g), dtype=np.float64)

    lib.solve_zeta_batch(
        ctypes.c_int(N_z), ctypes.c_int(m), ctypes.c_int(n_g),
        cost_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        A_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        b_res_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        x_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    return x_out.astype(np.float32)
