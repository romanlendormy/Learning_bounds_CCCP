"""
ccpo_graph.py — Construction du graphe CCPO pour l'entrée du GNN.

Sur le patron de gnn/mkp/gnn_sd_edit_native.cpp (MKP), topologie "sd_edit" :
un nœud par paire (scénario libre j, actif libre i), clique actif-actif par
scénario, étoile scénario-scénario ancrée sur un scénario de référence
(au lieu d'une clique complète -- O(n_g·N) arêtes au lieu de O(n_g·N²)).

Adapté à CCPO (pas de dimension `m` comme MKP -- une seule contrainte
quadratique par scénario) : les
features remplacent les `m` colonnes de MKP (A_ijk/b, efficacité, rang
Dantzig par dimension...) par des features tirées de la structure de risque
propre à CCPO (Σ_j[i][i], corrélations avec les autres actifs, tension du
budget résiduel). Pas d'accès à β_j/σ_f,j (choix délibéré, cf.
ccpo_sld.hpp : la sous-matrice dense reste la seule source de vérité) --
tout est calculé à partir de Σ_j dense, comme le fait déjà zeta_solve.

Comme MKP : aucun index/rang brut (k/n_g, i/N) -- "rangs arbitraires,
brisent l'invariance par permutation". Seul un rang DÉRIVÉ (ratio
rendement/risque, cf. branch_order de gecode_bab_ccpo.hpp) est inclus.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Data

NODE_DIM = 10  # cf. build_node_features (10e feature : R_ext, cf. correctif terme croisé)
CONTEXT_DIM = 11
NODE_DIM_V2 = 26  # cf. build_node_features_v2 (agrégats façon CCMKP)
NODE_DIM_V3 = 28  # cf. build_node_features_v3 (normalisations alignées sur CCMKP)

# Modes de pondération des arêtes (DOIVENT matcher ccpo_graph_cpp.hpp::EdgeMode et
# l'attribut edge_mode du module scripté) : 0 = aucune (poids 1), 1 = Σ_ab/σ
# (ablation D, Python seulement), 2 = façon CCMKP (clique 1/n_free, étoile 1).
EDGE_MODES = {"none": 0, "sigma": 1, "mkp": 2}


def node_dim_for(feat_version: int) -> int:
    return {1: NODE_DIM, 2: NODE_DIM_V2, 3: NODE_DIM_V3}[feat_version]


@dataclass
class CCPOResidualInstance:
    """Sous-problème résiduel passé au constructeur de graphe (mêmes quantités
    que celles calculées dans CCPOSpaceSLD::more() côté C++ -- cf.
    gecode_bab_ccpo.hpp)."""

    n_free: int
    N: int
    k_res: int
    Q_res: int
    L: int
    U: int
    cost_free: np.ndarray       # (n_free,) -- coût lagrangien courant (AVEC λ, pas cost_base seul)
    Sigma_free: np.ndarray      # (N, n_free, n_free) dense par scénario
    B_res: np.ndarray           # (N,) budget résiduel par scénario
    R_ext: np.ndarray           # (N, n_free) -- correction linéaire externe par scénario
                                 # (terme croisé fixé×libre, cf. ccpo_sld.hpp
                                 # / CCPOSpaceSLD::d_full) -- zéros à la racine.
    infeasible: np.ndarray      # (N,) bool -- presolve, scénarios jamais éligibles
    depth: int                  # nombre de variables q déjà fixées (0 = racine)
    n_total: int                # n de l'instance complète (pas n_free)
    frac_fixed_active: float    # fraction d'actifs déjà fixés ACTIFS (fixed_count_active/n_total)
    cost_fixed: float           # profit déjà fixé (fp_running)
    c_scale: float              # échelle de normalisation des coûts (ex. S_c de l'instance)
    sigma_scale: float          # échelle de normalisation du risque (ex. Sigma_bar*S de l'instance)
    epsilon: float              # ε de la contrainte de chance
    B_orig: float = 0.0         # budget de risque D'ORIGINE (avant fixations) -- features v3 seulement


def _dantzig_like_rank(cost_free: np.ndarray, risk_proxy: np.ndarray) -> np.ndarray:
    """Rang décroissant du ratio coût/risque, normalisé dans [0,1] -- analogue
    du dantzig_rank de MKP (jamais un index brut, toujours un rang DÉRIVÉ)."""
    n = len(cost_free)
    ratio = cost_free / np.maximum(risk_proxy, 1e-9)
    order = np.argsort(-ratio)
    rank = np.empty(n)
    rank[order] = np.arange(n) / max(1, n - 1)
    return rank


def build_node_features(inst: CCPOResidualInstance) -> np.ndarray:
    """Retourne un tableau (N_eligible * n_free, NODE_DIM) -- un nœud par
    paire (scénario ÉLIGIBLE j, actif libre i). Les scénarios presolve-
    infaisables n'ont pas de nœud (jamais soumis à zeta_solve)."""
    n_free, N = inst.n_free, inst.N
    eligible = np.where(~inst.infeasible)[0]
    n_elig = len(eligible)

    diag = np.diagonal(inst.Sigma_free, axis1=1, axis2=2)  # (N, n_free) -- Σ_j[i][i]
    row_sum = inst.Sigma_free.sum(axis=2) - diag           # (N, n_free) -- Σ_{k≠i} Σ_j[i][k]
    row_sq_sum = (inst.Sigma_free**2).sum(axis=2) - diag**2
    denom = max(1, n_free - 1)
    corr_mean = row_sum / denom
    corr_var = np.maximum(row_sq_sum / denom - corr_mean**2, 0.0)
    corr_std = np.sqrt(corr_var)

    dantzig = _dantzig_like_rank(inst.cost_free, np.mean(diag, axis=0) + 1e-9)

    diag_mean_over_scen = diag.mean(axis=0)   # (n_free,) -- stabilité du risque idiosyncratique
    diag_std_over_scen = diag.std(axis=0)

    feats = np.zeros((n_elig * n_free, NODE_DIM), dtype=np.float32)
    row = 0
    for j in eligible:
        f_c = inst.cost_free / inst.c_scale
        f_diag = diag[j] / inst.sigma_scale
        f_diag_over_bres = diag[j] / max(inst.B_res[j], 1e-9)
        f_corr_mean = corr_mean[j] / inst.sigma_scale
        f_corr_std = corr_std[j] / inst.sigma_scale
        f_dantzig = dantzig
        f_diag_mean_scen = diag_mean_over_scen / inst.sigma_scale
        f_diag_std_scen = diag_std_over_scen / inst.sigma_scale
        f_bres_tension = np.full(n_free, inst.B_res[j] / max(inst.sigma_scale * inst.Q_res**2, 1e-9))
        # Terme croisé fixé×libre (cf. ccpo_sld.hpp) : nul à la racine,
        # signale à quel point la contrainte de CET actif, pour CE scénario, a
        # été décalée par les actifs déjà fixés actifs par l'arbre englobant.
        f_r_ext = inst.R_ext[j] / max(inst.sigma_scale * inst.U, 1e-9)

        feats[row:row + n_free, 0] = f_c
        feats[row:row + n_free, 1] = f_diag
        feats[row:row + n_free, 2] = f_diag_over_bres
        feats[row:row + n_free, 3] = f_corr_mean
        feats[row:row + n_free, 4] = f_corr_std
        feats[row:row + n_free, 5] = f_dantzig
        feats[row:row + n_free, 6] = f_diag_mean_scen
        feats[row:row + n_free, 7] = f_diag_std_scen
        feats[row:row + n_free, 8] = f_bres_tension
        feats[row:row + n_free, 9] = f_r_ext
        row += n_free

    return feats


def _rank_desc_stable(x: np.ndarray) -> np.ndarray:
    """Rang décroissant normalisé dans [0,1], égalités départagées par indice
    croissant (tri STABLE -- identique à std::stable_sort côté C++)."""
    n = len(x)
    order = np.argsort(-x, kind="stable")
    rank = np.empty(n)
    rank[order] = np.arange(n) / max(1, n - 1)
    return rank


def _pop_std(x: np.ndarray, axis: int) -> np.ndarray:
    """Écart-type population (ddof=0), DEUX PASSES (même formule que le C++)."""
    m = x.mean(axis=axis, keepdims=True)
    return np.sqrt(((x - m) ** 2).mean(axis=axis))


def build_node_features_v2(inst: CCPOResidualInstance) -> np.ndarray:
    """Features v2 (NODE_DIM_V2 = 26), un nœud par (scénario ÉLIGIBLE j, actif libre i).
    Toutes les statistiques sur les scénarios portent sur les ÉLIGIBLES seulement
    (correctif F3). Grandeurs normalisées, invariantes par permutation (aucun index).

    Notations : d_ji = Σ_j[i][i], cm_ji / cs_ji = moyenne / écart-type de Σ_j[i][k≠i]
    (comme v1), q_max = max(min(U, Q_res), 0) (nombre de lots maximal d'un actif au nœud),
    r_ji = min(q_max² d_ji / max(B_res_j, 1e-9), 10) : part du budget de risque résiduel
    consommée par l'actif i SEUL à q_max lots (analogue de A/b_res MKP, écrêté à 10 comme
    f_frac). Sans q_max² (v1, col. 2) le ratio valait ~4e-3 en médiane : mal conditionné.
    Saturation (col. 13) : q_max² d_ji > B_res_j (l'actif seul à q_max sature le budget).

      0  c_i / c_scale                         13 frac_i(q_max² d_ji > B_res_j) [scénario]
      1  d_ji / σ                              14 moyenne_{a≠b} Σ_j[a][b] / σ  [scénario]
      2  r_ji                                  15 écart-type_{a≠b} Σ_j[a][b]/σ [scénario]
      3  cm_ji / σ                             16 écart-type_i d_ji / σ        [scénario]
      4  cs_ji / σ                             17 moy_E cm_·i / σ              [actif]
      5  rang global c_i / moy_E d_·i          18 é-t_E cm_·i / σ              [actif]
      6  moy_E d_·i / σ          [actif]       19 moy_E cs_·i / σ              [actif]
      7  é-t_E d_·i / σ          [actif]       20 é-t_E cs_·i / σ              [actif]
      8  B_res_j / (σ Q_res²)    [scénario]    21 moy_E r_·i                   [actif]
      9  R_ext_ji / (σ U)                      22 é-t_E r_·i                   [actif]
      10 min_i r_ji              [scénario]    23 moy_E R_ext_·i / (σU)        [actif]
      11 moyenne_i r_ji          [scénario]    24 é-t_E R_ext_·i / (σU)        [actif]
      12 écart-type_i r_ji       [scénario]    25 rang de c_i / d_ji dans le scénario j
    """
    n_free = inst.n_free
    eligible = np.where(~inst.infeasible)[0]
    n_elig = len(eligible)
    feats = np.zeros((n_elig * n_free, NODE_DIM_V2), dtype=np.float32)
    if n_elig == 0:
        return feats
    sig = inst.sigma_scale
    S = inst.Sigma_free[eligible]                          # (E, n, n)
    B = inst.B_res[eligible]                               # (E,)
    Rx = inst.R_ext[eligible] / max(sig * inst.U, 1e-9)    # (E, n)

    d = np.diagonal(S, axis1=1, axis2=2).copy()            # (E, n)
    denom = max(1, n_free - 1)
    row_sum = S.sum(axis=2) - d
    row_sq = (S ** 2).sum(axis=2) - d ** 2
    cm = row_sum / denom
    cs = np.sqrt(np.maximum(row_sq / denom - cm ** 2, 0.0))
    qm2 = float(max(min(inst.U, inst.Q_res), 0)) ** 2
    r = np.minimum(qm2 * d / np.maximum(B, 1e-9)[:, None], 10.0)

    # Hors-diagonale par scénario (n(n-1) entrées, les deux sens).
    if n_free > 1:
        off = ~np.eye(n_free, dtype=bool)
        offv = S[:, off]                                   # (E, n(n-1))
        off_mean = offv.mean(axis=1)
        off_std = _pop_std(offv, axis=1)
    else:
        off_mean = np.zeros(n_elig)
        off_std = np.zeros(n_elig)

    d_mean_E = d.mean(axis=0)
    dantzig = _rank_desc_stable(inst.cost_free / np.maximum(d_mean_E + 1e-9, 1e-9))

    per_asset = [d_mean_E / sig, _pop_std(d, 0) / sig,
                 cm.mean(axis=0) / sig, _pop_std(cm, 0) / sig,
                 cs.mean(axis=0) / sig, _pop_std(cs, 0) / sig,
                 r.mean(axis=0), _pop_std(r, 0),
                 Rx.mean(axis=0), _pop_std(Rx, 0)]

    f = feats.reshape(n_elig, n_free, NODE_DIM_V2)
    f[:, :, 0] = inst.cost_free[None, :] / inst.c_scale
    f[:, :, 1] = d / sig
    f[:, :, 2] = r
    f[:, :, 3] = cm / sig
    f[:, :, 4] = cs / sig
    f[:, :, 5] = dantzig[None, :]
    f[:, :, 6] = per_asset[0][None, :]
    f[:, :, 7] = per_asset[1][None, :]
    f[:, :, 8] = (B / max(sig * inst.Q_res ** 2, 1e-9))[:, None]
    f[:, :, 9] = Rx
    f[:, :, 10] = r.min(axis=1)[:, None]
    f[:, :, 11] = r.mean(axis=1)[:, None]
    f[:, :, 12] = _pop_std(r, 1)[:, None]
    f[:, :, 13] = (qm2 * d > B[:, None]).mean(axis=1)[:, None]
    f[:, :, 14] = (off_mean / sig)[:, None]
    f[:, :, 15] = (off_std / sig)[:, None]
    f[:, :, 16] = (_pop_std(d, 1) / sig)[:, None]
    for c, v in zip(range(17, 25), per_asset[2:]):
        f[:, :, c] = v[None, :]
    for s in range(n_elig):
        f[s, :, 25] = _rank_desc_stable(inst.cost_free / np.maximum(d[s], 1e-9))
    return feats


def build_node_features_v3(inst: CCPOResidualInstance) -> np.ndarray:
    """Features v3 (NODE_DIM_V3 = 28) : v2 avec les normalisations de CCMKP
    (gnn/mkp/train_SLD_sd_edit.py::build_graph). Unité de risque = budget
    D'ORIGINE B (MKP : A/b d'origine) au lieu de σ (moyenne globale des variances).
    Avec a_ji = U² d_ji / B (part du budget d'origine consommée par l'actif i seul
    à U lots, analogue de f_A = A/b ; U fixe comme b, donc indépendant du nœud) :

      0  c_i / c_scale                          14 U²·moyenne_{a≠b} Σ_j[a][b] / B  [scénario]
      1  a_ji                                   15 U²·é-t_{a≠b} Σ_j[a][b] / B      [scénario]
      2  r_ji = min(q_max² d_ji/B_res_j, 10)    16 é-t_i a_ji                      [scénario]
      3  U² cm_ji / B                           17 moy_E U² cm_·i / B              [actif]
      4  U² cs_ji / B                           18 é-t_E U² cm_·i / B              [actif]
      5  rang global c_i / moy_E d_·i           19 moy_E U² cs_·i / B              [actif]
      6  moy_E a_·i              [actif]        20 é-t_E U² cs_·i / B              [actif]
      7  é-t_E a_·i              [actif]        21 moy_E r_·i                      [actif]
      8  B_res_j / B   (= f_cap MKP)            22 é-t_E r_·i                      [actif]
      9  U·R_ext_ji / B                         23 moy_E U·R_ext_·i / B            [actif]
      10 min_i r_ji              [scénario]     24 é-t_E U·R_ext_·i / B            [actif]
      11 moyenne_i r_ji          [scénario]     25 rang de c_i / d_ji dans le scénario j
      12 écart-type_i r_ji       [scénario]     26 log1p((c_i/c_scale) / a_ji)  (= f_eff MKP)
      13 frac_i(q_max² d_ji > B_res_j)          27 1[B_res_j / B < 0.1]  (= cap_sat MKP) [scénario]

    Colonnes 2, 5, 10-13, 21-22, 25 : identiques à v2. DOIT rester identique à
    ccpo_graph_cpp.hpp::build_node_features_v3_cpp."""
    n_free = inst.n_free
    eligible = np.where(~inst.infeasible)[0]
    n_elig = len(eligible)
    feats = np.zeros((n_elig * n_free, NODE_DIM_V3), dtype=np.float32)
    if n_elig == 0:
        return feats
    B0 = max(float(inst.B_orig), 1e-9)
    u2b = float(inst.U) ** 2 / B0
    S = inst.Sigma_free[eligible]                          # (E, n, n)
    B = inst.B_res[eligible]                               # (E,)
    Rx = inst.R_ext[eligible] * (float(inst.U) / B0)       # (E, n)

    d = np.diagonal(S, axis1=1, axis2=2).copy()            # (E, n)
    denom = max(1, n_free - 1)
    row_sum = S.sum(axis=2) - d
    row_sq = (S ** 2).sum(axis=2) - d ** 2
    cm = row_sum / denom
    cs = np.sqrt(np.maximum(row_sq / denom - cm ** 2, 0.0))
    qm2 = float(max(min(inst.U, inst.Q_res), 0)) ** 2
    r = np.minimum(qm2 * d / np.maximum(B, 1e-9)[:, None], 10.0)
    a = d * u2b
    acm = cm * u2b
    acs = cs * u2b
    cap = B / B0

    if n_free > 1:
        off = ~np.eye(n_free, dtype=bool)
        offv = S[:, off]
        off_mean = offv.mean(axis=1)
        off_std = _pop_std(offv, axis=1)
    else:
        off_mean = np.zeros(n_elig)
        off_std = np.zeros(n_elig)

    d_mean_E = d.mean(axis=0)
    dantzig = _rank_desc_stable(inst.cost_free / np.maximum(d_mean_E + 1e-9, 1e-9))
    cn = inst.cost_free / inst.c_scale

    per_asset = [a.mean(axis=0), _pop_std(a, 0),
                 acm.mean(axis=0), _pop_std(acm, 0),
                 acs.mean(axis=0), _pop_std(acs, 0),
                 r.mean(axis=0), _pop_std(r, 0),
                 Rx.mean(axis=0), _pop_std(Rx, 0)]

    f = feats.reshape(n_elig, n_free, NODE_DIM_V3)
    f[:, :, 0] = cn[None, :]
    f[:, :, 1] = a
    f[:, :, 2] = r
    f[:, :, 3] = acm
    f[:, :, 4] = acs
    f[:, :, 5] = dantzig[None, :]
    f[:, :, 6] = per_asset[0][None, :]
    f[:, :, 7] = per_asset[1][None, :]
    f[:, :, 8] = cap[:, None]
    f[:, :, 9] = Rx
    f[:, :, 10] = r.min(axis=1)[:, None]
    f[:, :, 11] = r.mean(axis=1)[:, None]
    f[:, :, 12] = _pop_std(r, 1)[:, None]
    f[:, :, 13] = (qm2 * d > B[:, None]).mean(axis=1)[:, None]
    f[:, :, 14] = (off_mean * u2b)[:, None]
    f[:, :, 15] = (off_std * u2b)[:, None]
    f[:, :, 16] = _pop_std(a, 1)[:, None]
    for c, v in zip(range(17, 25), per_asset[2:]):
        f[:, :, c] = v[None, :]
    for s in range(n_elig):
        f[s, :, 25] = _rank_desc_stable(inst.cost_free / np.maximum(d[s], 1e-9))
    f[:, :, 26] = np.log1p(cn[None, :] / np.maximum(a, 1e-9))
    f[:, :, 27] = (cap < 0.1).astype(np.float64)[:, None]
    return feats


def build_context_features(inst: CCPOResidualInstance, feat_version: int = 1) -> np.ndarray:
    """Vecteur de contexte global (CONTEXT_DIM,) -- analogue des 9 features
    de contexte de MKP, plus 2 spécifiques à CCPO (tension de cardinalité et
    de budget de lots, absentes chez MKP).

    v3 (alignement CCMKP) : ctx[2] = min_E B_res/B (MKP : min b_res/b),
    ctx[7] = log1p(n_elig), ctx[8] = n_elig/n_libre (MKP : N_z, scénarios libres)."""
    n_free, N = inst.n_free, inst.N
    n_eligible = int((~inst.infeasible).sum())
    if feat_version >= 3:
        tightest = float(np.min(inst.B_res[~inst.infeasible] / max(inst.B_orig, 1e-9))
                         ) if n_eligible > 0 else 0.0
    else:
        tightest = float(np.min(inst.B_res[~inst.infeasible] / max(inst.sigma_scale * inst.Q_res**2, 1e-9))
                         ) if n_eligible > 0 else 0.0
    n_scen = n_eligible if feat_version >= 3 else N

    return np.array([
        inst.depth / max(1, inst.n_total),                      # depth_norm
        inst.frac_fixed_active,                                  # frac_fixed
        tightest,                                                 # tension budget de risque la + serree
        inst.cost_fixed / inst.c_scale,                           # coût déjà fixé, normalisé
        inst.epsilon,                                             # ε
        (n_eligible / max(1, N)) if feat_version >= 2 else 0.0,  # v2 : fraction de scénarios éligibles (v1 : 0, réservé)
        np.log1p(n_free),                                         # log(n_libre)
        np.log1p(n_scen),                                         # log(N) (v3 : log(n_elig))
        n_scen / max(1, n_free),                                  # N/n_libre (v3 : n_elig/n_libre)
        inst.k_res / max(1, n_free),                              # tension de cardinalité (SPÉCIFIQUE CCPO)
        inst.Q_res / max(1, n_free * inst.U),                     # tension de budget de lots (SPÉCIFIQUE CCPO)
    ], dtype=np.float32)


def build_edges(inst: CCPOResidualInstance) -> np.ndarray:
    """Arêtes (2, n_edges) -- clique actif-actif par scénario + étoile
    scénario-scénario ancrée sur le premier scénario éligible, à actif fixé
    (topologie "sd_edit" de MKP, cf. docstring du module)."""
    n_free = inst.n_free
    eligible = np.where(~inst.infeasible)[0]
    n_elig = len(eligible)
    if n_elig == 0:
        return np.zeros((2, 0), dtype=np.int64)

    edges = []

    # Clique actif-actif, par scénario (bloc de n_free nœuds consécutifs).
    idx_pairs = np.array([(a, b) for a in range(n_free) for b in range(n_free) if a != b])
    for s in range(n_elig):
        base = s * n_free
        if len(idx_pairs) > 0:
            edges.append(np.stack([idx_pairs[:, 0] + base, idx_pairs[:, 1] + base]))

    # Étoile scénario-scénario, à actif fixé : scénario 0 (du sous-ensemble
    # éligible) relié à tous les autres, pour chaque actif i.
    if n_elig > 1:
        for i in range(n_free):
            anchor = 0 * n_free + i
            others = np.arange(1, n_elig) * n_free + i
            edges.append(np.stack([np.full(n_elig - 1, anchor), others]))
            edges.append(np.stack([others, np.full(n_elig - 1, anchor)]))  # non-dirigé

    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    return np.concatenate(edges, axis=1).astype(np.int64)


def build_edge_weights(inst: CCPOResidualInstance) -> np.ndarray:
    """Poids d'arete (n_edges,), MEME ordre que build_edges : clique actif-actif
    par scenario eligible -> Sigma_j[a][b]/sigma_scale (l'interaction de risque
    entre les actifs a et b, exactement ce qui entre dans q'Sigma_j q) ; etoile
    inter-scenarios -> 1.0. Utilise seulement si CCPOGNN(use_edge_weight=True)
    (ablation D : goulot d'information -- les aretes
    ne portaient aucun poids, le GNN ne voyait que des statistiques de Sigma)."""
    n_free = inst.n_free
    eligible = np.where(~inst.infeasible)[0]
    n_elig = len(eligible)
    if n_elig == 0:
        return np.zeros(0, dtype=np.float32)
    A = np.array([a for a in range(n_free) for b in range(n_free) if a != b], dtype=np.int64)
    B = np.array([b for a in range(n_free) for b in range(n_free) if a != b], dtype=np.int64)
    ws = []
    for s in range(n_elig):
        j = eligible[s]
        ws.append(inst.Sigma_free[j][A, B] / inst.sigma_scale)
    if n_elig > 1:
        for _ in range(n_free):
            ws.append(np.ones(n_elig - 1))
            ws.append(np.ones(n_elig - 1))
    return np.concatenate(ws).astype(np.float32)


def build_edge_weights_mkp(n_free: int, n_elig: int) -> np.ndarray:
    """Poids façon CCMKP (train_SLD_sd_edit.build_graph), MÊME ordre que build_edges :
    clique intra-scénario 1/n_free, étoile inter-scénarios 1. Ne dépend que de
    (n_free, n_elig) -- mis en cache côté C++ avec la topologie."""
    if n_elig == 0:
        return np.zeros(0, dtype=np.float32)
    w_clique = np.full(n_elig * n_free * (n_free - 1), 1.0 / n_free, dtype=np.float32)
    w_star = np.ones(2 * n_free * (n_elig - 1) if n_elig > 1 else 0, dtype=np.float32)
    return np.concatenate([w_clique, w_star])


def build_graph(inst: CCPOResidualInstance, feat_version: int = 1, edge_mode: str = "sigma") -> Data:
    """Assemble un objet torch_geometric.data.Data prêt pour le GNN.

    feat_version : 1 (10 features, modèles existants) | 2 (NODE_DIM_V2) | 3 (NODE_DIM_V3).
    edge_mode : poids stockés dans data.edge_weight -- "sigma" (défaut historique,
    ignoré si le modèle a use_edge_weight=False), "mkp" ou "none"."""
    x = torch.from_numpy(build_node_features(inst) if feat_version == 1
                         else build_node_features_v2(inst) if feat_version == 2
                         else build_node_features_v3(inst))
    edge_index = torch.from_numpy(build_edges(inst))
    ctx = torch.from_numpy(build_context_features(inst, feat_version)).unsqueeze(0)  # (1, CONTEXT_DIM)

    eligible = np.where(~inst.infeasible)[0]
    n_elig = len(eligible)
    scenario_id = torch.arange(n_elig).repeat_interleave(inst.n_free)
    asset_id = torch.arange(inst.n_free).repeat(n_elig)

    data = Data(x=x, edge_index=edge_index)
    if edge_mode == "sigma":
        ew = build_edge_weights(inst)
    elif edge_mode == "mkp":
        ew = build_edge_weights_mkp(inst.n_free, n_elig)
    elif edge_mode == "none":
        ew = np.ones(edge_index.shape[1], dtype=np.float32)
    else:
        raise ValueError(f"edge_mode inconnu : {edge_mode}")
    data.edge_weight = torch.from_numpy(ew)
    data.ctx = ctx
    data.scenario_id = scenario_id
    data.asset_id = asset_id
    data.n_free = inst.n_free
    data.n_scenarios_eligible = n_elig
    data.eligible_scenario_idx = torch.from_numpy(eligible.astype(np.int64))
    return data
