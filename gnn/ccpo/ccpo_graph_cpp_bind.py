"""
ccpo_graph_cpp_bind.py — binding ctypes pour solver_bab/ccpo/ccpo_graph.so
(port C++ de ccpo_graph.py, cf. ccpo_graph_cpp.hpp/ccpo_graph_bind.cpp).
Utilise par test_ccpo_graph_cpp_vs_python.py pour valider numeriquement
l'equivalence AVANT integration au driver B&B+GNN batche.
"""
from __future__ import annotations

import ctypes
import os

import numpy as np

import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ccpo_libs import lib_path  # noqa: E402

_LIB_PATH = str(lib_path("ccpo_graph"))
_LIB = ctypes.CDLL(_LIB_PATH)

_LIB.ccpo_build_node_features_c.argtypes = [
    ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),  # cost_free
    ctypes.POINTER(ctypes.c_double),  # Sigma_free_flat
    ctypes.POINTER(ctypes.c_double),  # B_res
    ctypes.POINTER(ctypes.c_double),  # R_ext_flat
    ctypes.POINTER(ctypes.c_uint8),   # infeasible
    ctypes.c_int, ctypes.c_int,       # Q_res, U
    ctypes.c_double, ctypes.c_double, # c_scale, sigma_scale
    ctypes.POINTER(ctypes.c_float),   # feats_out
    ctypes.POINTER(ctypes.c_int),     # eligible_out
]
_LIB.ccpo_build_node_features_c.restype = None

_LIB.ccpo_build_context_features_c.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),  # B_res
    ctypes.POINTER(ctypes.c_uint8),   # infeasible
    ctypes.c_int, ctypes.c_int,       # depth, n_total
    ctypes.c_double, ctypes.c_double, # frac_fixed_active, cost_fixed
    ctypes.c_double, ctypes.c_double, ctypes.c_double,  # c_scale, sigma_scale, epsilon
    ctypes.POINTER(ctypes.c_float),   # ctx_out
]
_LIB.ccpo_build_context_features_c.restype = None

_LIB.ccpo_count_edges_c.argtypes = [ctypes.c_int, ctypes.c_int]
_LIB.ccpo_count_edges_c.restype = ctypes.c_int64

_LIB.ccpo_build_edges_c.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
_LIB.ccpo_build_edges_c.restype = None


def build_node_features_cpp(n_free, N, cost_free, Sigma_free, B_res, R_ext, infeasible, Q_res, U,
                             c_scale, sigma_scale):
    cost_free = np.ascontiguousarray(cost_free, dtype=np.float64)
    Sigma_free = np.ascontiguousarray(Sigma_free, dtype=np.float64)
    B_res = np.ascontiguousarray(B_res, dtype=np.float64)
    R_ext = np.ascontiguousarray(R_ext, dtype=np.float64)
    infeasible_u8 = np.ascontiguousarray(infeasible, dtype=np.uint8)
    n_elig = int((~infeasible.astype(bool)).sum())
    feats_out = np.zeros((n_elig * n_free, 10), dtype=np.float32)
    eligible_out = np.zeros(n_elig, dtype=np.int32)
    _LIB.ccpo_build_node_features_c(
        n_free, N,
        cost_free.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        Sigma_free.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        B_res.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        R_ext.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        infeasible_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        Q_res, U, c_scale, sigma_scale,
        feats_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        eligible_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )
    return feats_out, eligible_out


def build_context_features_cpp(n_free, N, k_res, Q_res, U, B_res, infeasible, depth, n_total,
                                frac_fixed_active, cost_fixed, c_scale, sigma_scale, epsilon):
    B_res = np.ascontiguousarray(B_res, dtype=np.float64)
    infeasible_u8 = np.ascontiguousarray(infeasible, dtype=np.uint8)
    ctx_out = np.zeros(11, dtype=np.float32)
    _LIB.ccpo_build_context_features_c(
        n_free, N, k_res, Q_res, U,
        B_res.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        infeasible_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        depth, n_total, frac_fixed_active, cost_fixed, c_scale, sigma_scale, epsilon,
        ctx_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return ctx_out


def build_edges_cpp(n_free, n_elig):
    n_edges = int(_LIB.ccpo_count_edges_c(n_free, n_elig))
    edge_index_out = np.zeros(2 * n_edges, dtype=np.int64)
    _LIB.ccpo_build_edges_c(n_free, n_elig, edge_index_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)))
    # layout C++ : [src(0..n_edges-1), dst(0..n_edges-1)] -- reshape vers (2,n_edges)
    return edge_index_out.reshape(2, n_edges)


# ── v2 : features NODE_DIM_V2, contexte versionné, poids d'arêtes façon CCMKP ──
_LIB.ccpo_build_node_features_v2_c.argtypes = _LIB.ccpo_build_node_features_c.argtypes
_LIB.ccpo_build_node_features_v2_c.restype = None
_LIB.ccpo_build_context_features_v_c.argtypes = _LIB.ccpo_build_context_features_c.argtypes + [ctypes.c_int]
_LIB.ccpo_build_context_features_v_c.restype = None
_LIB.ccpo_build_edge_weights_mkp_c.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
_LIB.ccpo_build_edge_weights_mkp_c.restype = None

NODE_DIM_V2 = 26


def build_node_features_v2_cpp(n_free, N, cost_free, Sigma_free, B_res, R_ext, infeasible, Q_res, U,
                                c_scale, sigma_scale):
    cost_free = np.ascontiguousarray(cost_free, dtype=np.float64)
    Sigma_free = np.ascontiguousarray(Sigma_free, dtype=np.float64)
    B_res = np.ascontiguousarray(B_res, dtype=np.float64)
    R_ext = np.ascontiguousarray(R_ext, dtype=np.float64)
    infeasible_u8 = np.ascontiguousarray(infeasible, dtype=np.uint8)
    n_elig = int((~infeasible.astype(bool)).sum())
    feats_out = np.zeros((n_elig * n_free, NODE_DIM_V2), dtype=np.float32)
    eligible_out = np.zeros(n_elig, dtype=np.int32)
    _LIB.ccpo_build_node_features_v2_c(
        n_free, N,
        cost_free.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        Sigma_free.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        B_res.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        R_ext.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        infeasible_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        Q_res, U, c_scale, sigma_scale,
        feats_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        eligible_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )
    return feats_out, eligible_out


def build_context_features_v_cpp(n_free, N, k_res, Q_res, U, B_res, infeasible, depth, n_total,
                                  frac_fixed_active, cost_fixed, c_scale, sigma_scale, epsilon,
                                  feat_version):
    B_res = np.ascontiguousarray(B_res, dtype=np.float64)
    infeasible_u8 = np.ascontiguousarray(infeasible, dtype=np.uint8)
    ctx_out = np.zeros(11, dtype=np.float32)
    _LIB.ccpo_build_context_features_v_c(
        n_free, N, k_res, Q_res, U,
        B_res.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        infeasible_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        depth, n_total, frac_fixed_active, cost_fixed, c_scale, sigma_scale, epsilon,
        ctx_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), int(feat_version),
    )
    return ctx_out


def build_edge_weights_mkp_cpp(n_free, n_elig):
    n_edges = int(_LIB.ccpo_count_edges_c(n_free, n_elig))
    w = np.zeros(n_edges, dtype=np.float32)
    _LIB.ccpo_build_edge_weights_mkp_c(n_free, n_elig, w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    return w


# ── v3 : features NODE_DIM_V3 (normalisations CCMKP), contexte v3 ──
_LIB.ccpo_build_node_features_v3_c.argtypes = _LIB.ccpo_build_node_features_c.argtypes
_LIB.ccpo_build_node_features_v3_c.restype = None
_LIB.ccpo_build_context_features_v3_c.argtypes = _LIB.ccpo_build_context_features_v_c.argtypes + [ctypes.c_double]
_LIB.ccpo_build_context_features_v3_c.restype = None

NODE_DIM_V3 = 28


def build_node_features_v3_cpp(n_free, N, cost_free, Sigma_free, B_res, R_ext, infeasible, Q_res, U,
                                c_scale, B_orig):
    cost_free = np.ascontiguousarray(cost_free, dtype=np.float64)
    Sigma_free = np.ascontiguousarray(Sigma_free, dtype=np.float64)
    B_res = np.ascontiguousarray(B_res, dtype=np.float64)
    R_ext = np.ascontiguousarray(R_ext, dtype=np.float64)
    infeasible_u8 = np.ascontiguousarray(infeasible, dtype=np.uint8)
    n_elig = int((~infeasible.astype(bool)).sum())
    feats_out = np.zeros((n_elig * n_free, NODE_DIM_V3), dtype=np.float32)
    eligible_out = np.zeros(n_elig, dtype=np.int32)
    _LIB.ccpo_build_node_features_v3_c(
        n_free, N,
        cost_free.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        Sigma_free.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        B_res.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        R_ext.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        infeasible_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        Q_res, U, c_scale, B_orig,
        feats_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        eligible_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )
    return feats_out, eligible_out


def build_context_features_v3_cpp(n_free, N, k_res, Q_res, U, B_res, infeasible, depth, n_total,
                                   frac_fixed_active, cost_fixed, c_scale, sigma_scale, epsilon,
                                   feat_version, B_orig):
    B_res = np.ascontiguousarray(B_res, dtype=np.float64)
    infeasible_u8 = np.ascontiguousarray(infeasible, dtype=np.uint8)
    ctx_out = np.zeros(11, dtype=np.float32)
    _LIB.ccpo_build_context_features_v3_c(
        n_free, N, k_res, Q_res, U,
        B_res.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        infeasible_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        depth, n_total, frac_fixed_active, cost_fixed, c_scale, sigma_scale, epsilon,
        ctx_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), int(feat_version), float(B_orig),
    )
    return ctx_out
