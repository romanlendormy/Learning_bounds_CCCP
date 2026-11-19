"""
ccpo_lagrangian_autograd.py — torch.autograd.Function maison pour L(λ) du CCPO.

Enveloppe evaluate_ccpo_lagrangian_c (solver_bab/ccpo/ccpo_lagrangian_bind.cpp,
compilé en .so, appelé via ctypes -- même patron que gnn/bnb_solver.cpp côté
MKP) via le théorème de l'enveloppe : forward = appel C++ (calcule L(λ) ET le
sous-gradient exact en une seule passe -- cf. evaluate_ccpo_lagrangian dans
ccpo_sld.hpp) ; backward = renvoie directement ce sous-gradient (pas de
différentiation automatique à travers theta_solve/zeta_solve, tous deux du
code combinatoire non différentiable).

Solidité : L(λ) est une borne supérieure VALIDE de la valeur optimale pour
N'IMPORTE QUEL λ réel (dualité faible) -- un GNN qui propose n'importe quoi ne peut jamais produire une borne
invalide, seule sa SERRURE dépend de la qualité de λ. C'est exactement ce qui
permet d'entraîner un réseau à MINIMISER L(λ) directement comme perte, sans
label externe (cf. train_ccpo_gnn.py).
"""

from __future__ import annotations

import ctypes
import os

import numpy as np
import torch

import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ccpo_libs import lib_path  # noqa: E402

_LIB_PATH = os.environ.get("CCPO_LAGRANGIAN_SO") or str(lib_path("ccpo_lagrangian"))
_LIB = ctypes.CDLL(os.path.abspath(_LIB_PATH))
_LIB.evaluate_ccpo_lagrangian_c.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),  # cost_base
    ctypes.POINTER(ctypes.c_double),  # Sigma
    ctypes.POINTER(ctypes.c_double),  # B_vec
    ctypes.POINTER(ctypes.c_uint8),   # infeasible_scenario
    ctypes.POINTER(ctypes.c_double),  # lambda
    ctypes.POINTER(ctypes.c_int),     # order
    ctypes.POINTER(ctypes.c_double),  # R_ext -- correctif terme croisé fixé×libre
    ctypes.POINTER(ctypes.c_double),  # subgrad_out
    ctypes.POINTER(ctypes.c_double),  # L_out
    ctypes.POINTER(ctypes.c_uint8),   # feasible_out
]
_LIB.evaluate_ccpo_lagrangian_c.restype = None

_LIB.evaluate_ccpo_lagrangian_omp_c.argtypes = _LIB.evaluate_ccpo_lagrangian_c.argtypes
_LIB.evaluate_ccpo_lagrangian_omp_c.restype = None

_LIB.ccpo_omp_set_num_threads.argtypes = [ctypes.c_int]
_LIB.ccpo_omp_set_num_threads.restype = None

_LIB.ccpo_omp_diag.argtypes = []
_LIB.ccpo_omp_diag.restype = None


def ccpo_omp_diag() -> None:
    _LIB.ccpo_omp_diag()


_LIB.ccpo_zeta_timing_diag_c.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),  # cost_base
    ctypes.POINTER(ctypes.c_double),  # Sigma
    ctypes.POINTER(ctypes.c_double),  # B_vec
    ctypes.POINTER(ctypes.c_uint8),   # infeasible_scenario
    ctypes.POINTER(ctypes.c_double),  # lambda
    ctypes.POINTER(ctypes.c_int),     # order
    ctypes.POINTER(ctypes.c_double),  # R_ext
    ctypes.POINTER(ctypes.c_double),  # t_ms_out
]
_LIB.ccpo_zeta_timing_diag_c.restype = None


def ccpo_zeta_timing_diag(
    n: int, N: int, k: int, Q: int, L: int, U: int,
    cost_base: np.ndarray, Sigma: np.ndarray, B_vec: np.ndarray,
    infeasible_scenario: np.ndarray, lam: np.ndarray, order: np.ndarray,
    R_ext: np.ndarray | None = None,
) -> np.ndarray:
    """DIAGNOSTIC : temps individuel (ms) de chaque zeta_solve(j), j=0..N-1
    (sequentiel). Teste la distribution du cout entre scenarios (stragglers ?)."""
    cost_base = np.ascontiguousarray(cost_base, dtype=np.float64)
    Sigma = np.ascontiguousarray(Sigma, dtype=np.float64)
    B_vec = np.ascontiguousarray(B_vec, dtype=np.float64)
    infeasible_u8 = np.ascontiguousarray(infeasible_scenario, dtype=np.uint8)
    lam = np.ascontiguousarray(lam, dtype=np.float64)
    order_i32 = np.ascontiguousarray(order, dtype=np.int32)
    if R_ext is None:
        R_ext = np.zeros((N, n), dtype=np.float64)
    R_ext = np.ascontiguousarray(R_ext, dtype=np.float64)
    t_ms_out = np.zeros(N, dtype=np.float64)
    _LIB.ccpo_zeta_timing_diag_c(
        n, N, k, Q, L, U,
        _dptr(cost_base), _dptr(Sigma), _dptr(B_vec),
        infeasible_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        _dptr(lam),
        order_i32.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        _dptr(R_ext),
        _dptr(t_ms_out),
    )
    return t_ms_out


_LIB.ccpo_zeta_omp_timeline_diag_c.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),  # cost_base
    ctypes.POINTER(ctypes.c_double),  # Sigma
    ctypes.POINTER(ctypes.c_double),  # B_vec
    ctypes.POINTER(ctypes.c_uint8),   # infeasible_scenario
    ctypes.POINTER(ctypes.c_double),  # lambda
    ctypes.POINTER(ctypes.c_int),     # order
    ctypes.POINTER(ctypes.c_double),  # R_ext
]
_LIB.ccpo_zeta_omp_timeline_diag_c.restype = None


def ccpo_zeta_omp_timeline_diag(
    n: int, N: int, k: int, Q: int, L: int, U: int,
    cost_base: np.ndarray, Sigma: np.ndarray, B_vec: np.ndarray,
    infeasible_scenario: np.ndarray, lam: np.ndarray, order: np.ndarray,
    R_ext: np.ndarray | None = None,
) -> None:
    """DIAGNOSTIC : imprime sur stderr le timeline (thread id, t0, t1) de
    chaque zeta_solve(j) DANS la boucle OpenMP reelle -- revele si les
    threads se recouvrent (vrai parallelisme) ou s'executent en serie."""
    cost_base = np.ascontiguousarray(cost_base, dtype=np.float64)
    Sigma = np.ascontiguousarray(Sigma, dtype=np.float64)
    B_vec = np.ascontiguousarray(B_vec, dtype=np.float64)
    infeasible_u8 = np.ascontiguousarray(infeasible_scenario, dtype=np.uint8)
    lam = np.ascontiguousarray(lam, dtype=np.float64)
    order_i32 = np.ascontiguousarray(order, dtype=np.int32)
    if R_ext is None:
        R_ext = np.zeros((N, n), dtype=np.float64)
    R_ext = np.ascontiguousarray(R_ext, dtype=np.float64)
    _LIB.ccpo_zeta_omp_timeline_diag_c(
        n, N, k, Q, L, U,
        _dptr(cost_base), _dptr(Sigma), _dptr(B_vec),
        infeasible_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        _dptr(lam),
        order_i32.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        _dptr(R_ext),
    )


def ccpo_omp_set_num_threads(n: int) -> None:
    """A appeler APRES import torch si torch.set_num_threads() a ete appele
    dans ce processus -- torch ecrase omp_set_num_threads() globalement au
    niveau de libgomp (partagee par tout le processus), ce qui neutralise
    silencieusement OMP_NUM_THREADS pour ccpo_lagrangian.so aussi (cf. commentaire ccpo_lagrangian_bind.cpp::ccpo_omp_set_num_threads)."""
    _LIB.ccpo_omp_set_num_threads(int(n))

_LIB.solve_ccpo_sld_c.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),  # cost_base
    ctypes.POINTER(ctypes.c_double),  # Sigma
    ctypes.POINTER(ctypes.c_double),  # B_vec
    ctypes.POINTER(ctypes.c_uint8),   # infeasible_scenario
    ctypes.POINTER(ctypes.c_double),  # R_ext
    ctypes.POINTER(ctypes.c_double),  # lambda_init
    ctypes.c_int,                     # T
    ctypes.c_double,                  # rel_tol
    ctypes.c_int,                     # tol_window
    ctypes.POINTER(ctypes.c_double),  # lambda_star_out
    ctypes.POINTER(ctypes.c_double),  # L_out
    ctypes.POINTER(ctypes.c_uint8),   # feasible_out
]
_LIB.solve_ccpo_sld_c.restype = None


def _dptr(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def evaluate_ccpo_lagrangian_raw(
    n: int, N: int, k: int, Q: int, L: int, U: int, min_feas: int,
    cost_base: np.ndarray,
    Sigma: np.ndarray,
    B_vec: np.ndarray,
    infeasible_scenario: np.ndarray,
    lam: np.ndarray,
    order: np.ndarray,
    R_ext: np.ndarray | None = None,
    omp_threads: int = 0,
) -> tuple[float, np.ndarray, bool]:
    """omp_threads > 0 : boucle scénario parallélisée OpenMP
    (evaluate_ccpo_lagrangian_omp_c, résultats identiques au bit près à la
    version séquentielle). Le nombre de threads est fixé DANS le thread
    appelant (ICV OpenMP par thread hôte : un omp_set_num_threads fait depuis
    le thread principal ne s'applique pas aux threads du ThreadPoolExecutor).

    Appel direct (numpy in, numpy out) -- pas de torch ici, cf. CCPOLagrangian pour l'usage autograd.

    R_ext : (N,n) -- correction linéaire externe par scénario (terme croisé
    fixé×libre, cf. ccpo_sld.hpp "CORRECTIF -- terme croisé"). None ou omis =
    zéros, valide UNIQUEMENT pour un sous-problème RACINE (aucune variable
    fixée par l'arbre englobant) -- pour un nœud résiduel réel, l'appelant
    DOIT fournir le R_ext calculé côté C++ (more()) ou reconstruit à
    l'identique côté Python (cf. ccpo_graph.CCPOResidualInstance.R_ext)."""
    cost_base = np.ascontiguousarray(cost_base, dtype=np.float64)
    Sigma = np.ascontiguousarray(Sigma, dtype=np.float64)
    B_vec = np.ascontiguousarray(B_vec, dtype=np.float64)
    infeasible_u8 = np.ascontiguousarray(infeasible_scenario, dtype=np.uint8)
    lam = np.ascontiguousarray(lam, dtype=np.float64)
    order_i32 = np.ascontiguousarray(order, dtype=np.int32)
    if R_ext is None:
        R_ext = np.zeros((N, n), dtype=np.float64)
    R_ext = np.ascontiguousarray(R_ext, dtype=np.float64)

    subgrad_out = np.zeros((N, n), dtype=np.float64)
    L_out = np.zeros(1, dtype=np.float64)
    feasible_out = np.zeros(1, dtype=np.uint8)

    if omp_threads > 0:
        _LIB.ccpo_omp_set_num_threads(int(omp_threads))
        fn = _LIB.evaluate_ccpo_lagrangian_omp_c
    else:
        fn = _LIB.evaluate_ccpo_lagrangian_c
    fn(
        n, N, k, Q, L, U, min_feas,
        _dptr(cost_base), _dptr(Sigma), _dptr(B_vec),
        infeasible_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        _dptr(lam),
        order_i32.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        _dptr(R_ext),
        _dptr(subgrad_out), _dptr(L_out),
        feasible_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )
    return float(L_out[0]), subgrad_out, bool(feasible_out[0])


def evaluate_ccpo_lagrangian_omp_raw(
    n: int, N: int, k: int, Q: int, L: int, U: int, min_feas: int,
    cost_base: np.ndarray,
    Sigma: np.ndarray,
    B_vec: np.ndarray,
    infeasible_scenario: np.ndarray,
    lam: np.ndarray,
    order: np.ndarray,
    R_ext: np.ndarray | None = None,
) -> tuple[float, np.ndarray, bool]:
    """IDENTIQUE à evaluate_ccpo_lagrangian_raw (même contrat), mais appelle
    evaluate_ccpo_lagrangian_omp_c (boucle scenario parallelisee OpenMP, cf.
    ccpo_sld.hpp) -- pour tester correction (doit renvoyer EXACTEMENT les
    memes L/subgrad/feasible que la version sequentielle, aucune tolerance
    numerique attendue puisque chaque scenario est independant) et gain de
    temps ()."""
    cost_base = np.ascontiguousarray(cost_base, dtype=np.float64)
    Sigma = np.ascontiguousarray(Sigma, dtype=np.float64)
    B_vec = np.ascontiguousarray(B_vec, dtype=np.float64)
    infeasible_u8 = np.ascontiguousarray(infeasible_scenario, dtype=np.uint8)
    lam = np.ascontiguousarray(lam, dtype=np.float64)
    order_i32 = np.ascontiguousarray(order, dtype=np.int32)
    if R_ext is None:
        R_ext = np.zeros((N, n), dtype=np.float64)
    R_ext = np.ascontiguousarray(R_ext, dtype=np.float64)

    subgrad_out = np.zeros((N, n), dtype=np.float64)
    L_out = np.zeros(1, dtype=np.float64)
    feasible_out = np.zeros(1, dtype=np.uint8)

    _LIB.evaluate_ccpo_lagrangian_omp_c(
        n, N, k, Q, L, U, min_feas,
        _dptr(cost_base), _dptr(Sigma), _dptr(B_vec),
        infeasible_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        _dptr(lam),
        order_i32.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        _dptr(R_ext),
        _dptr(subgrad_out), _dptr(L_out),
        feasible_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )
    return float(L_out[0]), subgrad_out, bool(feasible_out[0])


def solve_ccpo_sld_raw(
    n: int, N: int, k: int, Q: int, L: int, U: int, min_feas: int,
    cost_base: np.ndarray, Sigma: np.ndarray, B_vec: np.ndarray,
    infeasible_scenario: np.ndarray, R_ext: np.ndarray,
    T: int = 50, rel_tol: float = 1e-3, tol_window: int = 15,
    lambda_init: np.ndarray | None = None,
) -> tuple[np.ndarray, float, bool]:
    """Sous-gradient SLD complet (T itérations) -- calcule λ* de référence,
    utilisé pour les cibles du préchauffage supervisé (cf.
    warmup_on_roots côté MKP). Retourne (lambda_star (N,n), L_star, feasible)."""
    cost_base = np.ascontiguousarray(cost_base, dtype=np.float64)
    Sigma = np.ascontiguousarray(Sigma, dtype=np.float64)
    B_vec = np.ascontiguousarray(B_vec, dtype=np.float64)
    infeasible_u8 = np.ascontiguousarray(infeasible_scenario, dtype=np.uint8)
    R_ext = np.ascontiguousarray(R_ext, dtype=np.float64)
    if lambda_init is None:
        lambda_init = np.zeros((N, n), dtype=np.float64)
    lambda_init = np.ascontiguousarray(lambda_init, dtype=np.float64)

    lambda_star_out = np.zeros((N, n), dtype=np.float64)
    L_out = np.zeros(1, dtype=np.float64)
    feasible_out = np.zeros(1, dtype=np.uint8)

    _LIB.solve_ccpo_sld_c(
        n, N, k, Q, L, U, min_feas,
        _dptr(cost_base), _dptr(Sigma), _dptr(B_vec),
        infeasible_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        _dptr(R_ext), _dptr(lambda_init),
        T, rel_tol, tol_window,
        _dptr(lambda_star_out), _dptr(L_out),
        feasible_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )
    return lambda_star_out, float(L_out[0]), bool(feasible_out[0])


class CCPOLagrangian(torch.autograd.Function):
    """
    forward(lam) -> L(lam) (scalaire), pour une instance résiduelle fixée
    (passée via `ctx` par les autres arguments, non différentiés).

    backward : théorème de l'enveloppe -- le sous-gradient calculé au forward
    (en même temps que L, une seule passe C++) EST le gradient exact de L par
    rapport à λ à x* fixé (l'argmax/argmin de theta_solve/zeta_solve). Aucune
    différentiation à travers ces routines combinatoires n'est nécessaire ni
    possible -- exactement le même principe que lagrangian_sld_cpp côté MKP.

    Si le sous-problème est prouvé infaisable (feasible=False), retourne
    L=NaN et un gradient nul -- l'appelant (boucle d'entraînement) doit
    filtrer ces exemples AVANT le backward, pas s'appuyer sur ce cas limite
    pour propager un signal utile (sur SLDResult::feasible).
    """

    @staticmethod
    def forward(ctx, lam: torch.Tensor, n, N, k, Q, L, U, min_feas,
                cost_base_np, Sigma_np, B_vec_np, infeasible_np, order_np, R_ext_np):
        lam_np = lam.detach().cpu().numpy().astype(np.float64)
        L_val, subgrad_np, feasible = evaluate_ccpo_lagrangian_raw(
            n, N, k, Q, L, U, min_feas, cost_base_np, Sigma_np, B_vec_np,
            infeasible_np, lam_np, order_np, R_ext_np,
        )
        if not feasible:
            ctx.feasible = False
            ctx.save_for_backward(torch.zeros_like(lam))
            return torch.tensor(float("nan"), dtype=lam.dtype, device=lam.device)

        ctx.feasible = True
        subgrad = torch.from_numpy(subgrad_np).to(dtype=lam.dtype, device=lam.device)
        ctx.save_for_backward(subgrad)
        return torch.tensor(L_val, dtype=lam.dtype, device=lam.device)

    @staticmethod
    def backward(ctx, grad_output):
        (subgrad,) = ctx.saved_tensors
        # dL/dlam = subgrad (deja calcule au forward) ; chaine avec grad_output
        # (scalaire, typiquement 1.0 puisque L est directement la perte).
        return (grad_output * subgrad,) + (None,) * 13


def ccpo_lagrangian(lam: torch.Tensor, n: int, N: int, k: int, Q: int, L: int, U: int,
                     min_feas: int, cost_base_np: np.ndarray, Sigma_np: np.ndarray,
                     B_vec_np: np.ndarray, infeasible_np: np.ndarray,
                     order_np: np.ndarray, R_ext_np: np.ndarray | None = None) -> torch.Tensor:
    """Point d'entrée pratique : ccpo_lagrangian(lam, ...) -> L(lam), différentiable en lam.

    R_ext_np : (N,n) -- correction linéaire externe (cf. evaluate_ccpo_lagrangian_raw).
    None = zéros, valide UNIQUEMENT à la racine (cf. docstring ci-dessus)."""
    if R_ext_np is None:
        R_ext_np = np.zeros((N, n), dtype=np.float64)
    return CCPOLagrangian.apply(
        lam, n, N, k, Q, L, U, min_feas, cost_base_np, Sigma_np, B_vec_np, infeasible_np,
        order_np, R_ext_np,
    )


_LIB.ccpo_zeta_order_diag_c.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_uint8),
]
_LIB.ccpo_zeta_order_diag_c.restype = None


def ccpo_zeta_order_diag(n, N, k, Q, L, U, mode, cost_base, Sigma, B_vec, infeasible, lam, order, R_ext):
    """DIAGNOSTIC : par scenario (t_ms, noeuds explores, valeur zeta, faisable). mode 0 = ordre
    `order` commun ; mode 1 = ordre argsort(-cost_j) propre a chaque scenario."""
    cost_base = np.ascontiguousarray(cost_base, dtype=np.float64)
    Sigma = np.ascontiguousarray(Sigma, dtype=np.float64)
    B_vec = np.ascontiguousarray(B_vec, dtype=np.float64)
    inf8 = np.ascontiguousarray(infeasible, dtype=np.uint8)
    lam = np.ascontiguousarray(lam, dtype=np.float64)
    order32 = np.ascontiguousarray(order, dtype=np.int32)
    R_ext = np.ascontiguousarray(R_ext, dtype=np.float64)
    t = np.zeros(N); nodes = np.zeros(N, dtype=np.int64); val = np.zeros(N); feas = np.zeros(N, dtype=np.uint8)
    _LIB.ccpo_zeta_order_diag_c(
        n, N, k, Q, L, U, mode, _dptr(cost_base), _dptr(Sigma), _dptr(B_vec),
        inf8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), _dptr(lam),
        order32.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), _dptr(R_ext),
        _dptr(t), nodes.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), _dptr(val),
        feas.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
    return t, nodes, val, feas.astype(bool)


def ccpo_zeta_warm_diag(n, N, k, Q, L, U, warm_mode, cost_base, Sigma, B_vec, infeasible, lam, order, R_ext):
    """DIAGNOSTIC : zeta_solve par scenario avec warm_mode (0 historique, 1 incumbent glouton, 2 + plafond theta).
    Retourne (t_ms, noeuds, valeur zeta, faisable, valeur de l'incumbent glouton ou NaN)."""
    cost_base = np.ascontiguousarray(cost_base, dtype=np.float64)
    Sigma = np.ascontiguousarray(Sigma, dtype=np.float64)
    B_vec = np.ascontiguousarray(B_vec, dtype=np.float64)
    inf8 = np.ascontiguousarray(infeasible, dtype=np.uint8)
    lam = np.ascontiguousarray(lam, dtype=np.float64)
    order32 = np.ascontiguousarray(order, dtype=np.int32)
    R_ext = np.ascontiguousarray(R_ext, dtype=np.float64)
    t = np.zeros(N); nodes = np.zeros(N, dtype=np.int64); val = np.zeros(N)
    feas = np.zeros(N, dtype=np.uint8); warm = np.zeros(N)
    # argtypes posés ICI (pas à l'import) : l'ancien ccpo_lagrangian.so n'a pas cette fonction.
    f = _LIB.ccpo_zeta_warm_diag_c
    f.argtypes = _LIB.ccpo_zeta_order_diag_c.argtypes + [ctypes.POINTER(ctypes.c_double)]
    f.restype = None
    _LIB.ccpo_zeta_warm_diag_c(
        n, N, k, Q, L, U, warm_mode, _dptr(cost_base), _dptr(Sigma), _dptr(B_vec),
        inf8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), _dptr(lam),
        order32.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), _dptr(R_ext),
        _dptr(t), nodes.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), _dptr(val),
        feas.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), _dptr(warm))
    return t, nodes, val, feas.astype(bool), warm


def ccpo_zeta_branch_diag(n, N, k, Q, L, U, strategy, warm_mode, cost_base, Sigma, B_vec, infeasible, lam, order, R_ext):
    """DIAGNOSTIC : zeta_solve_ex par scenario selon la strategie de branchement (cf. ccpo_zeta_branch_diag_c).
    Retourne (t_ms, noeuds, valeur zeta, faisable)."""
    cost_base = np.ascontiguousarray(cost_base, dtype=np.float64)
    Sigma = np.ascontiguousarray(Sigma, dtype=np.float64)
    B_vec = np.ascontiguousarray(B_vec, dtype=np.float64)
    inf8 = np.ascontiguousarray(infeasible, dtype=np.uint8)
    lam = np.ascontiguousarray(lam, dtype=np.float64)
    order32 = np.ascontiguousarray(order, dtype=np.int32)
    R_ext = np.ascontiguousarray(R_ext, dtype=np.float64)
    t = np.zeros(N); nodes = np.zeros(N, dtype=np.int64); val = np.zeros(N); feas = np.zeros(N, dtype=np.uint8)
    f = _LIB.ccpo_zeta_branch_diag_c  # argtypes ici : l'ancien .so n'a pas cette fonction
    f.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_uint8)]
    f.restype = None
    f(n, N, k, Q, L, U, strategy, warm_mode, _dptr(cost_base), _dptr(Sigma), _dptr(B_vec),
      inf8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), _dptr(lam),
      order32.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), _dptr(R_ext),
      _dptr(t), nodes.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), _dptr(val),
      feas.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
    return t, nodes, val, feas.astype(bool)


def ccpo_zeta_set_config(warm_mode: int, branching: int) -> None:
    """Regle zeta_solve (defaut de production : 2, 1 ; ancien comportement : 0, 0). Hors section parallele."""
    _LIB.ccpo_zeta_set_config_c.argtypes = [ctypes.c_int, ctypes.c_int]
    _LIB.ccpo_zeta_set_config_c.restype = None
    _LIB.ccpo_zeta_set_config_c(int(warm_mode), int(branching))


def ccpo_zeta_get_config():
    _LIB.ccpo_zeta_get_config_c.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    _LIB.ccpo_zeta_get_config_c.restype = None
    w, b = ctypes.c_int(), ctypes.c_int()
    _LIB.ccpo_zeta_get_config_c(ctypes.byref(w), ctypes.byref(b))
    return w.value, b.value


def ccpo_zeta_set_config2(warm_mode: int, branching: int, infeas_prune: int, inherit_infeas: int) -> None:
    """Idem ccpo_zeta_set_config + detection d'infaisabilite dans zeta_solve_ex et heritage (defaut de production : 2,1,1,1)."""
    _LIB.ccpo_zeta_set_config2_c.argtypes = [ctypes.c_int] * 4
    _LIB.ccpo_zeta_set_config2_c.restype = None
    _LIB.ccpo_zeta_set_config2_c(int(warm_mode), int(branching), int(infeas_prune), int(inherit_infeas))


def solve_ccpo_sld_projected(
    n: int, N: int, k: int, Q: int, L: int, U: int, min_feas: int,
    cost_base: np.ndarray, Sigma: np.ndarray, B_vec: np.ndarray,
    infeasible_scenario: np.ndarray, order: np.ndarray, R_ext: np.ndarray,
    mode: str, T: int = 50, rel_tol: float = 1e-3, tol_window: int = 15,
    omp_threads: int = 0,
) -> tuple[np.ndarray, float, bool, int]:
    """Sous-gradient SLD PROJETÉ sur la famille de λ que le GNN peut produire
    -- cibles de préchauffage cohérentes.

    Même schéma que solve_ccpo_sld (ccpo_sld.hpp) : λ0=0, pas 1/sqrt(it+1),
    meilleur λ retenu, arrêt sur amélioration relative de best_L sur une
    fenêtre. Seule différence : projection du sous-gradient selon `mode` :
      "E0"  : lignes presolve-infaisables gelées à 0 (sortie GNN non centrée) ;
      "E0c" : E0 + Σ_{j∈E} λ_j = 0 (sortie GNN centrée sur E, --center_lam).
    λ part de 0 et chaque pas reste dans le sous-espace -> le λ retenu y est
    exactement (à l'arrondi près). Pas d'héritage des preuves d'infaisabilité
    (n'affecte que le coût de calcul, pas L).
    Retourne (lambda_best (N,n), L_best, feasible, n_eval)."""
    if mode not in ("E0", "E0c"):
        raise ValueError(f"mode de projection inconnu : {mode}")
    infeas = np.ascontiguousarray(infeasible_scenario, dtype=np.uint8)
    elig = np.where(infeas == 0)[0]
    lam = np.zeros((N, n), dtype=np.float64)
    best_L, best_lam = float("inf"), lam.copy()
    hist: list[float] = []
    n_eval = 0
    for it in range(T):
        Lv, g, feas = evaluate_ccpo_lagrangian_raw(
            n, N, k, Q, L, U, min_feas, cost_base, Sigma, B_vec, infeas, lam, order, R_ext,
            omp_threads=omp_threads)
        n_eval += 1
        if not feas or not np.isfinite(Lv):
            return lam, float("nan"), False, n_eval
        if Lv < best_L:
            best_L, best_lam = float(Lv), lam.copy()
        hist.append(best_L)
        if len(hist) > tol_window:
            old = hist[-1 - tol_window]
            if abs(old - best_L) / max(1.0, abs(old)) < rel_tol:
                break
        g = np.array(g, dtype=np.float64, copy=True)
        g[infeas != 0] = 0.0
        if mode == "E0c" and len(elig):
            g[elig] -= g[elig].mean(axis=0, keepdims=True)
        lam = lam - (1.0 / np.sqrt(it + 1.0)) * g
    return best_lam, best_L, True, n_eval
