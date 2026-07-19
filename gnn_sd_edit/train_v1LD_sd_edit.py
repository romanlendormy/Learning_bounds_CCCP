"""
train_v1LD_sd_edit.py — Variante de train_gnn_v1ld.py avec graphe "style Swann/Darius"
(solver_swann_darius.cpp) : la seule différence avec gnn/train_gnn_v1ld.py est la
construction des arêtes "inter-scénarios" dans build_graph() (voir plus bas), tout
le reste (features, contexte, modèle GNN, borne L1(λ), vérification sous-gradient)
est strictement identique et n'en dépend pas.

Rappel de la structure de graphe de solver_swann_darius.cpp (solver_swann_darius.cpp
L417-442) : nœuds = (contrainte j, item i) ; deux types d'arêtes :
  1. intra-contrainte : pour j fixé, clique complète sur les items i (poids 1/n)
  2. inter-contraintes : pour i fixé, étoile — chaque copie (j, i) n'est reliée
     qu'à la copie de référence (j=0, i), PAS à toutes les autres copies (poids 1)
Ici l'axe "contrainte" de swann/darius correspond à l'axe "scénario" (i_idx) de
V1LD, et l'axe "item" correspond à l'axe "item actif" (k_idx). Le train_gnn_v1ld.py
original construit une clique complète sur les DEUX axes ; cette variante remplace
la clique inter-scénarios par une étoile ancrée au premier scénario libre.

train_gnn_v1ld.py — Chargement et entraînement GNN pour la borne V1LD sur CC-MKP.

Format des fichiers d'entrée (générés par bab_mkp_gecode_collect v1ld, commit c2fdc2f) :

  instances_n{n}.txt — une ligne par instance résolue à optimalité :
    inst_id, m, n, N, epsilon, opt,
    c[n],
    b[i][j] pour i in N, j in m     (N*m valeurs, row-major),
    A[i][j][k] pour i in N, j in m, k in n  (N*m*n valeurs)

  nodes_n{n}.txt — une ligne par nœud B&B collecté :
    inst_id, depth, curr_cost, violated_count,
    fix_ones[n]  (0 = non-fixé-à-1, 1 = fixé-à-1)
    b_res[N*m]   (capacités résiduelles pour tous N scénarios, row-major)

  Total champs : 4 + n + N*m  (ex. n=5, m=2, N=100 → 209)

  violated_count : nombre de scénarios i tels que b_res[i][j] < 0 pour au moins un j
                   (capacité violée par les items fixés à 1 → z_i forcé à 1).
  fix_ones[k] ∈ {0,1}: 0 = non fixé à 1 (actif), 1 = fixé à 1.

Structure du graphe produit (un graphe par nœud B&B) :
  - Scénarios libres : free_z = {i : b_res[i][j] >= 0 pour tout j}
    N_z = |free_z| ; n_fixed_z1 = N - N_z  (scénarios forcés à z=1)
  - Nœuds   : N_z * n_g paires (scénario libre i_idx, item actif k_idx)
              indexé par i_idx * n_g + k_idx
  - Features : 5m + 3 :
      [c_k, A_ijk(m), clip(A_ijk/b_res,10)(m), log1p(c_k/A_ijk)(m), dantzig_rank(m), b_res/b(m),
       min_j(b_res/b)_i, mean_j(b_res/b)_i]  ← 2 features scénario-niveau, court-circuitent le MP
      (k/n_g et i/N_z supprimés : rangs arbitraires, brisent l'invariance par permutation)
  - Arêtes   : item_varies (clique par scénario, poids 1/n_g)
               scenario_varies (ÉTOILE ancrée au scénario 0, poids 1 — PAS une
               clique, cf. solver_swann_darius.cpp)
  - Contexte : [depth, frac_fixés, tightest, curr_cost, epsilon, n_fixed_z1/N, log(n_g), log(N_z), N_z/n_g]
  - Target   : minimiser L1(λ) la borne V1LD
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import importlib, sys
_pyg_data_mod = importlib.import_module("torch_geometric.data")
pyg_data = _pyg_data_mod
DataLoader = importlib.import_module("torch_geometric.loader").DataLoader
from torch_geometric.nn import GatedGraphConv
from torch_geometric.nn import global_mean_pool, global_max_pool

# ── Solveur B&B C++ (optionnel, fallback Python si absent) ───────────────────
# Réutilise le binaire compilé de gnn/ (non dupliqué ici, inchangé par sd_edit).
import importlib.util as _ilu, pathlib as _pl
_bnb_spec = _ilu.spec_from_file_location(
    "bnb_solver", _pl.Path(__file__).parent.parent / "gnn" / "bnb_solver.py"
)
try:
    _bnb_mod = _ilu.module_from_spec(_bnb_spec)
    _bnb_spec.loader.exec_module(_bnb_mod)
    _cpp_solve_zeta_batch = _bnb_mod.solve_zeta_batch
    _USE_CPP_BNB = True
    print("[bnb_solver] C++ B&B chargé (solve_zeta_batch)", file=sys.stderr)
except Exception as _e:
    _cpp_solve_zeta_batch = None
    _USE_CPP_BNB = False
    print(f"[bnb_solver] fallback Python B&B ({_e})", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# Chargement des instances
# ─────────────────────────────────────────────────────────────────────────────

def load_instances(instances_file: str | Path) -> dict[int, dict]:
    """
    Lit instances_n{n}.txt et retourne un dict {inst_id: {n,m,N,c,b,A,opt}}.
    b  : ndarray (N, m)
    A  : ndarray (N, m, n)
    c  : ndarray (n,)
    """
    instances: dict[int, dict] = {}
    with open(instances_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(x) for x in line.split(",")]

            inst_id = int(vals[0])
            m       = int(vals[1])
            n       = int(vals[2])
            N       = int(vals[3])
            epsilon = vals[4]
            opt     = vals[5]

            c = np.array(vals[6 : 6 + n], dtype=np.float32)

            off = 6 + n
            b = np.array(vals[off : off + N * m], dtype=np.float32).reshape(N, m)

            off += N * m
            A = np.array(vals[off : off + N * m * n], dtype=np.float32).reshape(N, m, n)

            instances[inst_id] = {
                "n": n, "m": m, "N": N,
                "epsilon": epsilon, "opt": opt,
                "c": c, "b": b, "A": A,
            }
    return instances


# ─────────────────────────────────────────────────────────────────────────────
# Construction du graphe pour un nœud B&B
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(inst: dict, node_rec: dict) -> Optional[pyg_data.Data]:
    """
    Construit un graphe torch_geometric à partir d'une instance et d'un état
    de nœud B&B V1LD.

    Doit accepter un seul format de node_rec :

    Format B — nouveau (nodes_n10+ dans train_v1ld/, 5+n+N champs) :
      x_mask[k] ∈ {0,1,2} : 0=fixé_0, 1=fixé_1, 2=libre
      z_mask[i] ∈ {0,1,2} : 0=fixé_0, 1=fixé_1, 2=libre
      b_res[N, m]          : recalculé depuis instances + x_mask
      → free_z = indices où z_mask == 2
      → n_fixed_z1 = count(z_mask == 1)

    Nœuds : N_z × n_g paires (scénario libre, item actif).
    Features (5m + 3) : [c_k, A_ijk(m), clip(A_ijk/b_res,10)(m), log1p(c_k/A_ijk)(m),
                          dantzig_rank(m), b_res/b(m), min_j(b_res/b)_i, mean_j(b_res/b)_i]
    Arêtes : item_varies (clique par scénario, poids 1/n_g) ;
             scenario_varies (étoile ancrée au scénario 0, poids 1 — style
             solver_swann_darius.cpp, pas une clique).

    Retourne None si n_g == 0 ou N_z == 0.
    """
    n, m, N  = inst["n"], inst["m"], inst["N"]
    c, b, A  = inst["c"], inst["b"], inst["A"]
    x_mask   = node_rec["x_mask"]   # (n,)  int8
    b_res    = node_rec["b_res"]    # (N, m)  float32

    z_mask = node_rec.get("z_mask", None)   # présent seulement en format B

    if z_mask is not None:
        # ── Format B : x_mask ternaire, z_mask explicite ──────────────────────
        active_x   = np.where(x_mask == 2)[0]   # items libres (ni fixé_0 ni fixé_1)
        free_z     = np.where(z_mask == 2)[0]
        n_fixed_z1 = int((z_mask == 1).sum())
    else:
        # ── Format A : x_mask binaire, free_z déduit de b_res ─────────────────
        active_x   = np.where(x_mask == 0)[0]   # non-fixé-à-1
        free_z     = np.where(np.all(b_res >= -1e-9, axis=1))[0]
        n_fixed_z1 = N - len(free_z)

    n_g = len(active_x)
    N_z = len(free_z)
    if n_g == 0 or N_z == 0:
        return None

    # ── Sous-matrices : scénarios libres × items actifs ───────────────────────
    b_res_z = b_res[free_z, :]                       # (N_z, m)
    A_g     = A[np.ix_(free_z, range(m), active_x)]  # (N_z, m, n_g)
    c_g     = c[active_x]                            # (n_g,)
    b_z     = b[free_z, :]                           # (N_z, m)

    curr_cost   = node_rec["curr_cost"]
    c_scale     = max(float(curr_cost) + float(c_g.sum()), 1.0)
    b_orig_safe = np.maximum(b_z, 1e-9)

    # ── Features des nœuds (N_z * n_g, 5m + 3) ───────────────────────────────
    c_2d = np.broadcast_to(c_g[None, :], (N_z, n_g))   # (N_z, n_g)

    A_T    = A_g.transpose(0, 2, 1).copy()              # (N_z, n_g, m)
    bres_T = b_res_z[:, np.newaxis, :]                  # (N_z, 1, m)

    f_c    = c_2d[:, :, None] / c_scale                                        # (N_z, n_g, 1)
    f_A    = A_T / b_orig_safe[:, np.newaxis, :]                               # (N_z, n_g, m)
    f_frac = np.minimum(
        A_T / np.maximum(np.broadcast_to(bres_T, (N_z, n_g, m)), 1e-9),
        10.0)                                                                    # (N_z, n_g, m) clippé à 10
    f_eff  = np.log1p(
        (c_2d[:, :, None] / c_scale) / np.maximum(f_A, 1e-9))                 # (N_z, n_g, m) log(1+eff)

    eff_dantzig  = c_g[None, None, :] / np.maximum(A_g, 1e-9)                 # (N_z, m, n_g)
    dantzig_rank = (
        np.argsort(np.argsort(-eff_dantzig, axis=2), axis=2)
        / max(n_g - 1, 1)
    ).astype(np.float32)                                                        # (N_z, m, n_g)
    f_rank = dantzig_rank.transpose(0, 2, 1).copy()                            # (N_z, n_g, m)

    cap_frac = b_res_z / np.maximum(b_z, 1e-9)                                # (N_z, m)
    f_cap    = np.broadcast_to(cap_frac[:, None, :], (N_z, n_g, m)).copy()    # (N_z, n_g, m)

    # Features scénario-niveau : min et mean de cap_frac sur les contraintes j.
    # Même valeur pour tous les items k d'un même scénario i → le head peut
    # différencier les λ par scénario sans dépendre du message-passing.
    scen_min_cap  = cap_frac.min(axis=1)                                        # (N_z,)
    scen_mean_cap = cap_frac.mean(axis=1)                                       # (N_z,)
    f_scen_min  = np.broadcast_to(
        scen_min_cap[:, None, None],  (N_z, n_g, 1)).copy()                    # (N_z, n_g, 1)
    f_scen_mean = np.broadcast_to(
        scen_mean_cap[:, None, None], (N_z, n_g, 1)).copy()                    # (N_z, n_g, 1)

    # Features item-niveau : moyenne et écart-type de f_A sur les scénarios i.
    # Capturent la "difficulté universelle" de l'item k dans la contrainte j,
    # indépendamment du scénario courant.
    f_A_mean_scen = np.broadcast_to(
        f_A.mean(axis=0)[None, :, :], (N_z, n_g, m)).copy()                   # (N_z, n_g, m)
    f_A_std_scen  = np.broadcast_to(
        f_A.std(axis=0)[None, :, :],  (N_z, n_g, m)).copy()                   # (N_z, n_g, m)

    # Features scénario-niveau : std et count de saturation de cap_frac sur j.
    # Caractérisent "à quel point ce scénario est difficile à couvrir",
    # information clé pour la sélection y* dans la borne V1LD.
    cap_std_scen = cap_frac.std(axis=1)                                        # (N_z,)
    cap_sat_scen = (cap_frac < 0.1).sum(axis=1).astype(np.float32) / m        # (N_z,) normalisé ∈[0,1]
    f_cap_std_scen = np.broadcast_to(
        cap_std_scen[:, None, None], (N_z, n_g, 1)).copy()                    # (N_z, n_g, 1)
    f_cap_sat_scen = np.broadcast_to(
        cap_sat_scen[:, None, None], (N_z, n_g, 1)).copy()                    # (N_z, n_g, 1)

    X = np.concatenate(
        [f_c, f_A, f_frac, f_eff, f_rank, f_cap,
         f_scen_min, f_scen_mean,
         f_A_mean_scen, f_A_std_scen,
         f_cap_std_scen, f_cap_sat_scen], axis=-1
    ).reshape(-1, 7 * m + 5).astype(np.float32)                                # (N_z*n_g, 7m+5)

    # ── Arêtes (vectorisées) ──────────────────────────────────────────────────
    i_all = np.arange(N_z, dtype=np.int64)
    k_all = np.arange(n_g, dtype=np.int64)

    # 1. Item varie (même scénario i_idx) : poids 1/n_g
    ki, kj   = np.triu_indices(n_g, k=1)
    i_rep    = np.repeat(i_all, len(ki))
    ki_rep   = np.tile(ki, N_z);  kj_rep = np.tile(kj, N_z)
    src1_a   = i_rep * n_g + ki_rep;  src1_b = i_rep * n_g + kj_rep
    src1     = np.concatenate([src1_a, src1_b])
    dst1     = np.concatenate([src1_b, src1_a])
    w1       = np.full(len(src1), 1.0 / n_g, dtype=np.float32)

    # 2. Scénario varie (même item k_idx) : ÉTOILE style swann/darius, PAS une
    #    clique — chaque scénario i>0 n'est relié qu'au scénario de référence
    #    i=0 (analogue à la copie "contrainte 0" de solver_swann_darius.cpp
    #    L435-442), pas à tous les autres scénarios. Poids 1 (comme le code
    #    C++ de référence), pas 1/N_z : ce n'est plus une moyenne sur N_z-1
    #    voisins mais un unique lien vers l'ancre.
    ii       = np.arange(1, N_z, dtype=np.int64)          # scénarios "rayons"
    k_rep    = np.repeat(k_all, len(ii))
    ii_rep   = np.tile(ii, n_g)
    src2_a   = 0 * n_g + k_rep;   src2_b = ii_rep * n_g + k_rep   # ancre → rayon
    src2     = np.concatenate([src2_a, src2_b])
    dst2     = np.concatenate([src2_b, src2_a])                  # bidirectionnel
    w2       = np.full(len(src2), 1.0, dtype=np.float32)

    edge_index  = torch.tensor(
        np.stack([np.concatenate([src1, src2]), np.concatenate([dst1, dst2])]),
        dtype=torch.long)
    edge_weight = torch.tensor(np.concatenate([w1, w2]), dtype=torch.float32)

    # ── Contexte (features au niveau du graphe) ───────────────────────────────
    depth_norm     = float(node_rec["depth"]) / max(n, 1)
    frac_fixed     = 1.0 - n_g / n
    curr_cost_norm = float(curr_cost) / c_scale
    tightest       = float(np.min(cap_frac))

    context = torch.tensor(
        [depth_norm, frac_fixed, tightest, curr_cost_norm,
         float(inst["epsilon"]), n_fixed_z1 / N,
         float(np.log(n_g)), float(np.log(N_z)), float(N_z) / float(n_g)],
        dtype=torch.float32)

    return pyg_data.Data(
        x             = torch.from_numpy(X),
        edge_index    = edge_index,
        edge_weight   = edge_weight,
        context       = context,
        opt           = torch.tensor(float(inst["opt"]), dtype=torch.float32),
        curr_cost_raw = torch.tensor(float(curr_cost), dtype=torch.float32),
        c_scale       = torch.tensor(c_scale, dtype=torch.float32),
        n_g           = n_g,
        N             = N,
        N_z           = N_z,
        m             = m,
        n_fixed_z1    = n_fixed_z1,
        epsilon       = torch.tensor(float(inst["epsilon"]), dtype=torch.float32),
        # Matrices pour le calcul de L1(λ) — scénarios libres seulement
        A_g     = torch.from_numpy(A_g.copy()),       # (N_z, m, n_g)
        b_res_g = torch.from_numpy(b_res_z.copy()),   # (N_z, m)
        c_g     = torch.from_numpy(c_g.copy()),       # (n_g,)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Chargement du dataset complet
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(
    instances_file: str | Path,
    nodes_file: str | Path,
    max_graphs: Optional[int] = None,
) -> list[pyg_data.Data]:
    """
    Charge tous les états de nœuds B&B comme graphes PyG.

    Détecte automatiquement le format :
      Format A (ancien, 4+n+N*m champs) :
        inst_id, depth, curr_cost, violated_count, fix_ones[n], b_res[N*m]
      Format B (5+n+N champs, bab_mkp_gecode_collect) :
        inst_id, depth, curr_cost, ub_node, incumbent, x_mask[n]{0/1/2}, z_mask[N]{0/1/2}
      Format C (3+n+N champs, bab_mkp_gecode_base_collect) :
        inst_id, depth, curr_cost, x_mask[n]{0/1/2}, z_mask[N]{0/1/2}
    """
    instances = load_instances(instances_file)
    graphs: list[pyg_data.Data] = []
    skipped = 0

    with open(nodes_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            vals = [float(x) for x in line.split(",")]
            inst_id   = int(vals[0])
            depth     = int(vals[1])
            curr_cost = vals[2]

            if inst_id not in instances:
                skipped += 1
                continue

            inst = instances[inst_id]
            n, m, N = inst["n"], inst["m"], inst["N"]

            n_fmt_A = 4 + n + N * m
            n_fmt_B = 5 + n + N
            n_fmt_C = 3 + n + N

            if len(vals) == n_fmt_C:
                # ── Format C : sans ub_node/incumbent (collect_gecode_base) ──────
                x_mask = np.array(vals[3 : 3 + n], dtype=np.int8)
                z_mask = np.array(vals[3 + n : 3 + n + N], dtype=np.int8)
                fixed_to_1 = np.where(x_mask == 1)[0]
                b_res = inst["b"].copy()
                if len(fixed_to_1):
                    b_res -= inst["A"][:, :, fixed_to_1].sum(axis=2)
                node_rec = {
                    "depth":     depth,
                    "curr_cost": curr_cost,
                    "x_mask":    x_mask,
                    "z_mask":    z_mask,
                    "b_res":     b_res,
                }
            elif len(vals) >= n_fmt_B and len(vals) < n_fmt_A:
                # ── Format B : x_mask ternaire + z_mask, b_res à reconstruire ──
                x_mask = np.array(vals[5 : 5 + n], dtype=np.int8)
                z_mask = np.array(vals[5 + n : 5 + n + N], dtype=np.int8)
                fixed_to_1 = np.where(x_mask == 1)[0]
                b_res = inst["b"].copy()
                if len(fixed_to_1):
                    b_res -= inst["A"][:, :, fixed_to_1].sum(axis=2)
                node_rec = {
                    "depth":     depth,
                    "curr_cost": curr_cost,
                    "x_mask":    x_mask,
                    "z_mask":    z_mask,
                    "b_res":     b_res,
                }
            elif len(vals) >= n_fmt_A:
                # ── Format A : fix_ones binaire + b_res explicite ─────────────
                x_mask = np.array(vals[4 : 4 + n], dtype=np.int8)
                b_res  = np.array(vals[4 + n : 4 + n + N * m],
                                  dtype=np.float32).reshape(N, m)
                node_rec = {
                    "depth":     depth,
                    "curr_cost": curr_cost,
                    "x_mask":    x_mask,
                    "b_res":     b_res,
                }
            else:
                skipped += 1
                continue

            g = build_graph(inst, node_rec)
            if g is not None:
                g.inst_id = torch.tensor(inst_id, dtype=torch.long)
                graphs.append(g)
            else:
                skipped += 1

            if max_graphs and len(graphs) >= max_graphs:
                break

    print(f"[load_dataset] {len(graphs)} graphes chargés, {skipped} ignorés")
    return graphs


# ─────────────────────────────────────────────────────────────────────────────
# Chargement brut (sans construction PyG) + datasets lazy/eager
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_nodes(
    instances_file: str | Path,
    nodes_file: str | Path,
    max_records: Optional[int] = None,
) -> tuple[dict, list]:
    """
    Parse instances + nœuds sans construire les graphes PyG.
    Supporte Format B (5+n+N) et Format C (3+n+N, bab_mkp_gecode_base).
    Retourne (instances_dict, raw_records).
    """
    instances = load_instances(instances_file)
    records: list[dict] = []
    skipped = 0

    with open(nodes_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            vals = [float(x) for x in line.split(",")]
            inst_id   = int(vals[0])
            depth     = int(vals[1])
            curr_cost = vals[2]

            if inst_id not in instances:
                skipped += 1
                continue

            inst = instances[inst_id]
            n, N = inst["n"], inst["N"]

            n_fmt_B = 5 + n + N
            n_fmt_C = 3 + n + N

            if len(vals) == n_fmt_C:
                x_mask = np.array(vals[3 : 3 + n], dtype=np.int8)
                z_mask = np.array(vals[3 + n : 3 + n + N], dtype=np.int8)
            elif len(vals) >= n_fmt_B:
                x_mask = np.array(vals[5 : 5 + n], dtype=np.int8)
                z_mask = np.array(vals[5 + n : 5 + n + N], dtype=np.int8)
            else:
                skipped += 1
                continue

            if (x_mask == 2).sum() == 0 or (z_mask == 2).sum() == 0:
                skipped += 1
                continue

            records.append({
                "inst_id":   inst_id,
                "depth":     depth,
                "curr_cost": curr_cost,
                "x_mask":    x_mask,
                "z_mask":    z_mask,
            })
            if max_records and len(records) >= max_records:
                break

    print(f"[load_raw_nodes] {len(records)} enregistrements chargés, {skipped} ignorés")
    return instances, records


class LazyGraphDataset(torch.utils.data.Dataset):
    """Dataset PyTorch qui construit les graphes PyG à la volée."""

    def __init__(self, instances: dict, raw_records: list):
        self.instances   = instances
        self.raw_records = raw_records

    def __len__(self) -> int:
        return len(self.raw_records)

    def __getitem__(self, idx: int) -> pyg_data.Data:
        rec  = self.raw_records[idx]
        inst = self.instances[rec["inst_id"]]

        x_mask     = rec["x_mask"]
        fixed_to_1 = np.where(x_mask == 1)[0]
        b_res      = inst["b"].copy()
        if len(fixed_to_1):
            b_res -= inst["A"][:, :, fixed_to_1].sum(axis=2)

        node_rec = {
            "depth":     rec["depth"],
            "curr_cost": rec["curr_cost"],
            "x_mask":    x_mask,
            "z_mask":    rec["z_mask"],
            "b_res":     b_res,
        }

        g = build_graph(inst, node_rec)
        if g is None:
            raise ValueError(
                f"build_graph a retourné None pour inst_id={rec['inst_id']} "
                f"depth={rec['depth']} — ce record aurait dû être filtré."
            )
        g.inst_id = torch.tensor(rec["inst_id"], dtype=torch.long)
        if "n_orig" in rec:
            g.n_orig = torch.tensor(rec["n_orig"], dtype=torch.long)
        return g

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def inst_ids(self) -> list[int]:
        return [rec["inst_id"] for rec in self.raw_records]

    def filter_by_inst_ids(self, ids_set: set) -> "LazyGraphDataset":
        filtered = [rec for rec in self.raw_records if rec["inst_id"] in ids_set]
        return LazyGraphDataset(self.instances, filtered)

    def subset(self, start: int, stop: int) -> "LazyGraphDataset":
        return LazyGraphDataset(self.instances, self.raw_records[start:stop])

    def sample(self, n: int, rng: np.random.Generator) -> "LazyGraphDataset":
        n = min(n, len(self))
        idx = rng.choice(len(self), n, replace=False)
        return LazyGraphDataset(self.instances, [self.raw_records[i] for i in idx])


class EagerGraphDataset:
    """Construit tous les graphes PyG en mémoire au démarrage."""

    def __init__(self, lazy: LazyGraphDataset):
        self.graphs:    list = []
        self._inst_ids: list = []
        self._depths:   list = []
        n_skip = 0
        for i in range(len(lazy)):
            try:
                g = lazy[i]
            except Exception:
                n_skip += 1
                continue
            self.graphs.append(g)
            rec = lazy.raw_records[i]
            self._inst_ids.append(rec["inst_id"])
            self._depths.append(rec["depth"])
        if n_skip:
            print(f"  {n_skip} enregistrements ignorés (build_graph=None)", flush=True)

    @classmethod
    def _make(cls, graphs, inst_ids, depths):
        obj = cls.__new__(cls)
        obj.graphs     = list(graphs)
        obj._inst_ids  = list(inst_ids)
        obj._depths    = list(depths)
        return obj

    def __len__(self):          return len(self.graphs)
    def __getitem__(self, idx): return self.graphs[idx]
    def inst_ids(self):         return self._inst_ids

    def filter_by_inst_ids(self, ids_set):
        triples = [(g, i, d) for g, i, d in
                   zip(self.graphs, self._inst_ids, self._depths)
                   if i in ids_set]
        if not triples:
            return EagerGraphDataset._make([], [], [])
        gs, ids, ds = zip(*triples)
        return EagerGraphDataset._make(gs, ids, ds)


# ─────────────────────────────────────────────────────────────────────────────
# Modèle GNN
# ─────────────────────────────────────────────────────────────────────────────

class GNN(nn.Module):
    """
    Prédit λ[i,k] ∈ R par nœud (scénario libre i, item actif k).

    Architecture (7m+5 node features, context 9-dim, max_pool optionnel) :
      embed      : X(7m+5) → hidden → ReLU → hidden → ReLU → LayerNorm
      conv1      : GatedGraphConv(hidden, 2 it.) + LayerNorm
      conv2      : GatedGraphConv(hidden, 2 it.) + LayerNorm + skip(h1)
      proj       : hidden → 4h → ReLU → 2h                    → h_v
      pool item  : nœud → item k [mean] (+ [max] si use_max_pool) → h_item_k
      pool scén  : nœud → scénario i [mean] (+ [max] si use_max_pool) → h_scen_i
      pool graph : scénario → graphe [mean] → g_pool (2h)
      norm_ctx   : LayerNorm sur les 9 features de contexte (calculé une seule fois)
      head       : [h_v | h_item_k | h_scen_i | g_pool | ctx] → h → ReLU → h → ReLU → 1

    `h_item_k` (ajouté 2026-07-07, `diag_scenario_variance.py`) = moyenne
    (+max) de `h_v` sur les N_z scénarios, à item k fixé — taille n_g,
    indépendante de N, analogue direct du consensus `x̄_k` de la mise à jour
    sous-gradient (`verify_v1ld` : `lam[i] += step*(x_bar - xi)`), jamais
    représenté explicitement dans le head avant ce changement (cf.
    NOTES_N_scaling_investigation.md, "piste retenue").

    `h_scen_i` réintroduit le même jour après un test isolé (retrait de
    g_pool ET h_scen_i à la fois) qui a dégradé la qualité de la borne
    malgré un ICC plus proche de la cible — signe que h_scen_i, bien que
    redondant en information avec `h_v` (déjà vu via les arêtes
    item_varies), sert de raccourci d'optimisation utile à budget d'epochs
    fixé. `g_pool` réintroduit ensuite pour tester si une tête plus riche
    (rien retiré, seulement h_item_k ajouté à l'architecture originale) fait
    au moins aussi bien — ablation antérieure avait montré que g_pool seul
    ne différenciait ni scénarios ni items, mais son retrait combiné à celui
    de h_scen_i avait dégradé la borne, donc prudence : on ne retire plus
    rien, on ajoute seulement.

    Paramètres :
      m (int)             : nombre de contraintes (détermine NODE_DIM = 7m+5)
      hidden              : dimension des couches cachées (défaut 64)
      use_max_pool (bool) : si True, pool item/scén = [mean | max]
    """
    CONTEXT_DIM = 9  # [depth, frac_fixed, tightest, curr_cost, epsilon, n_fixed_z1/N, log(n_g), log(N_z), N_z/n_g]

    def __init__(self, m: int = 2, hidden: int = 64,
                 use_max_pool: bool = False):
        super().__init__()
        self.m            = m
        self.hidden       = hidden
        self.use_max_pool = use_max_pool
        node_dim    = 7 * m + 5

        self.embed = nn.Sequential(
            nn.Linear(node_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
        )
        self.norm_embed = nn.LayerNorm(hidden)
        self.conv1 = GatedGraphConv(hidden, num_layers=2)
        self.norm1 = nn.LayerNorm(hidden)
        self.conv2 = GatedGraphConv(hidden, num_layers=2)
        self.norm2 = nn.LayerNorm(hidden)
        self.proj  = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.ReLU(),
            nn.Linear(4 * hidden, 2 * hidden),
        )
        self.norm_ctx = nn.LayerNorm(self.CONTEXT_DIM)
        # pool item / scén : 2h chacun (mean seul) ou 4h (mean+max)
        item_pool_dim = 4 * hidden if use_max_pool else 2 * hidden
        scen_pool_dim = 4 * hidden if use_max_pool else 2 * hidden

        # h_v(2h) + h_item_k(item_pool_dim) + h_scen_i(scen_pool_dim) + g_pool(2h) + ctx(9)
        head_in = 2 * hidden + item_pool_dim + scen_pool_dim + 2 * hidden + self.CONTEXT_DIM
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, data: pyg_data.Data, return_hidden: bool = False):
        x, ei, ew = data.x, data.edge_index, data.edge_weight
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        n_v      = x.size(0)
        dev      = x.device
        n_graphs = int(batch.max().item()) + 1

        # ── Contexte normalisé (une seule fois — réutilisé par la tête) ─────────
        ctx = data.context
        if ctx.dim() == 1:
            ctx = ctx.unsqueeze(0)
        ctx_norm = self.norm_ctx(ctx)                                  # (1 or B, 9)

        # ── Message passing ───────────────────────────────────────────────────
        h_embed = self.norm_embed(self.embed(x))
        h = h_embed
        h1  = self.norm1(self.conv1(h,  ei, ew))
        h2  = self.norm2(self.conv2(h1, ei, ew)) + h1
        h_v = self.proj(h2)                                        # (V, 2h)

        # ── Pooling item-consensus (x̄_k) et scénario (h_scen_i) ───────────────
        # Nœuds ordonnés i_idx*n_g + k_idx ; n_g consécutifs = un scénario, donc
        # k_idx = local_idx % n_g identifie l'item indépendamment du scénario,
        # et i_idx = local_idx // n_g identifie le scénario indépendamment de
        # l'item. NB : suppose n_g uniforme sur le batch (hypothèse déjà faite
        # par l'ancien pooling scénario — pas une nouvelle limitation).
        n_g = data.n_g if isinstance(data.n_g, int) else int(data.n_g.item())

        counts   = torch.bincount(batch, minlength=n_graphs)
        node_off = torch.zeros(n_graphs, dtype=torch.long, device=dev)
        node_off[1:] = counts[:-1].cumsum(0)
        local_idx = torch.arange(n_v, dtype=torch.long, device=dev) - node_off[batch]

        item_off   = torch.arange(n_graphs, device=dev) * n_g
        item_batch = item_off[batch] + (local_idx % n_g)            # (V,)

        h_item_mean = global_mean_pool(h_v, item_batch)               # (Σ n_g, 2h)
        if self.use_max_pool:
            h_item_max = global_max_pool(h_v, item_batch)          # (Σ n_g, 2h)
            h_item = torch.cat([h_item_mean, h_item_max], dim=-1)  # (Σ n_g, 4h)
        else:
            h_item = h_item_mean                                    # (Σ n_g, 2h)
        h_item_exp = h_item[item_batch]                            # (V, 2h or 4h)

        n_scen_per = counts // n_g
        scen_off   = torch.zeros(n_graphs, dtype=torch.long, device=dev)
        scen_off[1:] = n_scen_per[:-1].cumsum(0)
        scen_batch = scen_off[batch] + local_idx // n_g            # (V,)

        h_scen_mean = global_mean_pool(h_v, scen_batch)              # (Σ N_z, 2h)
        if self.use_max_pool:
            h_scen_max = global_max_pool(h_v, scen_batch)         # (Σ N_z, 2h)
            h_scen = torch.cat([h_scen_mean, h_scen_max], dim=-1) # (Σ N_z, 4h)
        else:
            h_scen = h_scen_mean                                   # (Σ N_z, 2h)
        h_scen_exp = h_scen[scen_batch]                            # (V, 2h or 4h)

        ctx_exp = ctx_norm[batch]                                          # (V, 9)

        g_pool  = global_mean_pool(h_v, batch)                        # (B, 2h)
        g_exp   = g_pool[batch]                                            # (V, 2h)

        cat     = torch.cat([h_v, h_item_exp, h_scen_exp, g_exp, ctx_exp], dim=-1)
        h_out   = self.head(cat).squeeze(-1)

        lam = h_out * data.c_scale / float(data.N)
        if return_hidden:
            # Diagnostic uniquement (bottleneck de pooling) : h_v (par nœud),
            # h_item_mean (par item), h_scen_mean (par scénario), avant concat max.
            extra = {"x": x, "h_embed": h, "h1": h1, "h2": h2,
                     "h_v": h_v, "h_item_mean": h_item_mean, "item_batch": item_batch,
                     "h_scen_mean": h_scen_mean, "scen_batch": scen_batch,
                     "g_pool": g_pool, "h_out": h_out}
            return lam, extra
        return lam


# ─────────────────────────────────────────────────────────────────────────────
# Solveur knapsack multi-dimensionnel exact (B&B)
# ─────────────────────────────────────────────────────────────────────────────

def _solve_zeta_multi_bb(
    cost: np.ndarray,   # (n_g,) coûts lagrangiens
    A_i:  np.ndarray,   # (m, n_g) matrice de contraintes du scénario i
    b_i:  np.ndarray,   # (m,) capacités résiduelles
) -> tuple[float, np.ndarray]:
    """
    Knapsack 0/1 multi-dimensionnel exact par Branch & Bound.
    Port de mkp_problem_bnb.cpp.

    max cost^T x  s.t.  A_i x ≤ b_i,  x ∈ {0,1}^n_g
    Retourne (valeur_optimale, x* ∈ {0,1}^n_g).
    """
    n_g = len(cost)
    m   = len(b_i)
    x_out    = np.zeros(n_g, dtype=np.float32)
    val_free = 0.0

    active = []
    for k in range(n_g):
        if cost[k] <= 0.0:
            continue
        if np.all(A_i[:, k] < 1e-9):
            x_out[k] = 1.0
            val_free += cost[k]
        else:
            active.append(k)

    if not active:
        return val_free, x_out

    na = len(active)

    def surrogate(k):
        w = sum(A_i[j, k] / b_i[j] for j in range(m) if b_i[j] > 1e-12)
        return cost[k] / w if w > 1e-12 else 1e18

    active.sort(key=surrogate, reverse=True)

    p     = np.array([cost[k] for k in active], dtype=np.float64)
    A_act = np.array([[A_i[j, k] for k in active] for j in range(m)],
                     dtype=np.float64)

    sort_j = []
    for j in range(m):
        eff = [p[ki] / A_act[j, ki] if A_act[j, ki] > 1e-12 else 1e18
               for ki in range(na)]
        sort_j.append(sorted(range(na), key=lambda x, e=eff: -e[x]))

    best   = [val_free]
    best_x = [np.zeros(na, dtype=bool)]
    cur_x  = np.zeros(na, dtype=bool)

    def lp_bound_j(j, idx, rem_j, cur_val):
        v, r = cur_val, rem_j
        for ki in sort_j[j]:
            if ki < idx:
                continue
            a = A_act[j, ki]
            if a < 1e-12:
                v += p[ki]
            elif a <= r + 1e-12:
                v += p[ki]; r -= a
            else:
                v += p[ki] * r / a; break
        return v

    def upper_bound(idx, rem_b, cur_val):
        ub = 1e30
        for j in range(m):
            ub = min(ub, lp_bound_j(j, idx, rem_b[j], cur_val))
        return ub

    def bb(idx, rem_b, cur_val):
        if upper_bound(idx, rem_b, cur_val) <= best[0] + 1e-9:
            return
        if idx == na:
            if cur_val > best[0] + 1e-9:
                best[0] = cur_val
                best_x[0] = cur_x.copy()
            return
        feasible = all(A_act[j, idx] <= rem_b[j] + 1e-12 for j in range(m))
        if feasible:
            for j in range(m): rem_b[j] -= A_act[j, idx]
            cur_x[idx] = True
            bb(idx + 1, rem_b, cur_val + p[idx])
            cur_x[idx] = False
            for j in range(m): rem_b[j] += A_act[j, idx]
        bb(idx + 1, rem_b, cur_val)

    bb(0, b_i.copy().astype(np.float64), val_free)

    for ki, k in enumerate(active):
        if best_x[0][ki]:
            x_out[k] = 1.0

    return best[0], x_out


# ─────────────────────────────────────────────────────────────────────────────
# Borne L1(λ) — implémentation différentiable de V1LD
# ─────────────────────────────────────────────────────────────────────────────

def lagrangian_v1ld(lam: torch.Tensor, data: pyg_data.Data) -> torch.Tensor:
    """
    Calcule L1(λ) sur les scénarios libres, différentiable en λ.

    lam     : (N_z*n_g,) ou (N_z, n_g) — sortie brute du GNN
    data    : graphe avec A_g (N_z, m, n_g), b_res_g (N_z, m), c_g (n_g,),
              N (total scénarios), N_z (scénarios libres), n_fixed_z1

    Algorithme (théorème de l'enveloppe) :
      1. cost_i[k] = c[k] + λ_ik / p_i − Λ_sum[k]   (p_i = 1/N)
      2. θ_i = Σ max(0, cost_i[k])
      3. ζ_i = knapsack B&B multi-dimensionnel (detach)
      4. Δ_i = max(0, θ_i − ζ_i)
      5. required_prob = max(0, (1-ε) - n_fixed_z1/N)
      6. y* = sélection gloutonne des N_z scénarios libres (detach)
      7. L1 ≈ p Σ_i θ_i(λ) − p Σ_i y*_i Δ_i(λ)   linéaire en λ
    """
    N_z        = data.N_z
    n_g        = data.n_g
    N          = data.N
    n_fixed_z1 = data.n_fixed_z1
    dev        = lam.device

    lam_2d  = lam.view(N_z, n_g)

    A_g     = data.A_g.to(dev)      # (N_z, m, n_g)
    b_res_g = data.b_res_g.to(dev)  # (N_z, m)
    c_g     = data.c_g.to(dev)      # (n_g,)
    epsilon = data.epsilon.item()

    p_unif = 1.0 / N   # probabilité de chaque scénario (sur N total)

    # (A) Λ_sum = Σ_i λ_i   (sur N_z scénarios libres)
    Lambda_sum = lam_2d.sum(dim=0)                              # (n_g,)

    # (B) cost_i[k] = c[k] + λ_ik/p_i − Λ_sum[k]
    cost = c_g.unsqueeze(0) + lam_2d / p_unif - Lambda_sum.unsqueeze(0)  # (N_z, n_g)

    # ── Sous-problèmes (hors graphe de calcul) ────────────────────────────────
    cost_np = cost.detach().cpu().numpy()
    A_np    = A_g.cpu().numpy()
    bres_np = b_res_g.cpu().numpy()

    x_theta_np = (cost_np > 0).astype(np.float32)

    if _USE_CPP_BNB:
        x_zeta_np = _cpp_solve_zeta_batch(
            cost_np.astype(np.float64),
            A_np.astype(np.float64),
            bres_np.astype(np.float64),
        )
    else:
        x_zeta_np = np.zeros_like(x_theta_np)
        for i in range(N_z):
            _, xz = _solve_zeta_multi_bb(cost_np[i], A_np[i], bres_np[i])
            x_zeta_np[i] = xz

    # (C) Δ_i et y* greedy
    theta_np = np.sum(np.maximum(cost_np, 0.0), axis=1)
    zeta_np  = np.sum(cost_np * x_zeta_np, axis=1)
    Delta_np = np.maximum(0.0, theta_np - zeta_np)

    required_prob = max(0.0, (1.0 - epsilon) - n_fixed_z1 * p_unif)
    order   = np.argsort(Delta_np)
    y_star  = np.zeros(N_z, dtype=np.float32)
    cumprob = 0.0
    for idx in order:
        if required_prob <= 0.0:
            break
        y_star[idx] = 1.0
        cumprob += p_unif
        if cumprob >= required_prob - 1e-9:
            break

    # (D) L1 linéaire en λ à x*, y* fixés
    x_theta_t = torch.tensor(x_theta_np, device=dev)
    x_zeta_t  = torch.tensor(x_zeta_np,  device=dev)
    y_star_t  = torch.tensor(y_star,     device=dev)

    theta_lam = (cost * x_theta_t).sum(dim=1)
    delta_lam = (cost * (x_theta_t - x_zeta_t)).sum(dim=1)

    L1_theta = (p_unif * theta_lam).sum()
    val_lam  = (p_unif * y_star_t * delta_lam).sum()
    return L1_theta - val_lam


# ─────────────────────────────────────────────────────────────────────────────
# Vérification par sous-gradient Python pur — port de v1ld_solver.cpp
# ─────────────────────────────────────────────────────────────────────────────

def verify_v1ld(
    data: pyg_data.Data,
    T: int = 200,
    precision: float | None = None,
    patience:  int | None   = None,
    lam_init:  np.ndarray | None = None,
    return_n_iters: bool = False,
    return_lam: bool = False,
    history_out: list | None = None,
) -> float | tuple:
    """
    Reproduit l'algorithme v1ld_solver.cpp en Python pur (pas 1/√(t+1))
    et retourne la meilleure L1 vue.

    lam_init : (N_z, n_g) float64 ; None = cold start λ=0.
    history_out : si fourni, la valeur de `best` (meilleure L1 vue jusqu'à
        l'itération t inclusivement) y est appendée à chaque itération —
        permet de retracer la trajectoire complète sans rappeler la fonction
        avec des T croissants. Ignore patience/precision pour l'arrêt anticipé
        si on veut la trajectoire jusqu'à T (les passer à None/None).
    """
    N_z        = data.N_z
    n_g        = data.n_g
    N          = data.N
    n_fixed_z1 = data.n_fixed_z1
    epsilon    = data.epsilon.item()
    A_np       = data.A_g.numpy()      # (N_z, m, n_g)
    b_np       = data.b_res_g.numpy()  # (N_z, m)
    c_np       = data.c_g.numpy()      # (n_g,)

    p_unif        = 1.0 / N
    required_prob = max(0.0, (1.0 - epsilon) - n_fixed_z1 * p_unif)

    lam = (lam_init.copy() if lam_init is not None
           else np.zeros((N_z, n_g), dtype=np.float64))

    best       = np.inf
    best_lam   = lam.copy()
    no_improve = 0
    n_iters    = T

    for t in range(T):
        Lsum = lam.sum(axis=0)
        cost = c_np + lam / p_unif - Lsum               # (N_z, n_g)

        x_theta    = (cost > 0).astype(np.float64)
        theta_vals = np.sum(np.maximum(cost, 0), axis=1)

        if _USE_CPP_BNB:
            x_zeta = _cpp_solve_zeta_batch(
                cost.astype(np.float64),
                A_np.astype(np.float64),
                b_np.astype(np.float64),
            ).astype(np.float64)
        else:
            x_zeta = np.zeros_like(x_theta)
            for i in range(N_z):
                _, xz = _solve_zeta_multi_bb(
                    cost[i].astype(np.float32),
                    A_np[i].astype(np.float32),
                    b_np[i].astype(np.float32))
                x_zeta[i] = xz.astype(np.float64)

        zeta_vals = np.sum(cost * x_zeta, axis=1)
        Delta     = np.maximum(0.0, theta_vals - zeta_vals)

        order   = np.argsort(Delta)
        y_star  = np.zeros(N_z)
        cumprob = 0.0
        for idx in order:
            if required_prob <= 0.0:
                break
            y_star[idx] = 1.0
            cumprob += p_unif
            if cumprob >= required_prob - 1e-9:
                break

        L1 = p_unif * theta_vals.sum() - p_unif * (y_star * Delta).sum()

        if L1 < best - (precision or 0.0):
            best = L1; best_lam = lam.copy(); no_improve = 0
        else:
            no_improve += 1
        if history_out is not None:
            history_out.append(best)
        if patience is not None and precision is not None and no_improve >= patience:
            n_iters = t + 1
            break

        x_bar = np.zeros(n_g)
        for i in range(N_z):
            xi = x_zeta[i] if y_star[i] > 0.5 else x_theta[i]
            x_bar += p_unif * xi

        step = 1.0 / np.sqrt(t + 1)

        for i in range(N_z):
            xi = x_zeta[i] if y_star[i] > 0.5 else x_theta[i]
            lam[i] += step * (x_bar - xi)

    if return_n_iters and return_lam:
        return best, n_iters, best_lam
    if return_lam:
        return best, best_lam
    if return_n_iters:
        return best, n_iters
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Overfit sur un graphe — validation de l'architecture
# ─────────────────────────────────────────────────────────────────────────────

def overfit_one_graph(data: pyg_data.Data, n_epochs: int = 1000, lr: float = 1e-3):
    """Entraîne le GNN sur un seul graphe avec la vraie borne V1LD comme loss."""
    device = torch.device("cpu")
    data   = data.to(device)

    model     = GNN(m=data.m).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)

    opt_val = data.opt.item()
    with torch.no_grad():
        lam0    = torch.zeros(data.N_z * data.n_g)
        l1_init = lagrangian_v1ld(lam0, data).item()

    print(f"\n{'='*60}")
    print(f"Overfit — n_g={data.n_g}  N_z={data.N_z}  N={data.N}  m={data.m}")
    print(f"  opt={opt_val:.2f}   L1(λ=0)={l1_init:.2f}")
    print(f"  graphe : {data.x.shape[0]} nœuds, {data.edge_index.shape[1]} arêtes")
    print(f"{'='*60}")
    print(f"{'epoch':>6}  {'L1(λ_GNN)':>12}  {'gap_vs_opt%':>12}  {'gap_vs_L1_0%':>13}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        lam  = model(data)
        loss = lagrangian_v1ld(lam, data)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 50 == 0 or epoch == 1:
            b     = loss.item()
            g_opt = (b - opt_val) / opt_val * 100 if opt_val > 0 else float("nan")
            g_l0  = (b - l1_init) / abs(l1_init) * 100 if l1_init != 0 else float("nan")
            print(f"{epoch:6d}  {b:12.4f}  {g_opt:+11.2f}%  {g_l0:+12.2f}%")

    print(f"{'='*60}")
    print(f"Borne finale : {loss.item():.4f}  (opt={opt_val:.2f}, L1_0={l1_init:.2f})")


# ─────────────────────────────────────────────────────────────────────────────
# Test rapide
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/train_v1LD")
    n_val    = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    mode     = sys.argv[3] if len(sys.argv) > 3 else "inspect"

    inst_file  = data_dir / f"instances_n{n_val}.txt"
    nodes_file = data_dir / f"nodes_n{n_val}.txt"

    graphs = load_dataset(inst_file, nodes_file, max_graphs=5)

    for i, g in enumerate(graphs):
        print(
            f"  graph {i:3d} | nodes={g.x.shape[0]:6d} | "
            f"edges={g.edge_index.shape[1]:8d} | "
            f"n_g={g.n_g} N_z={g.N_z}/{g.N} m={g.m} | "
            f"depth={int(g.context[0].item())} | opt={g.opt.item():.2f}"
        )

    if mode in ("verify", "overfit") and graphs:
        g0 = graphs[0]
        lam0    = torch.zeros(g0.N_z * g0.n_g)
        l1_init = lagrangian_v1ld(lam0, g0).item()
        print(f"\n── Vérification V1LD (T=100 itérations subgradient) ──")
        print(f"  L1(λ=0)          : {l1_init:.4f}")
        py_best = verify_v1ld(g0, T=100)
        print(f"  subgradient T=100 : {py_best:.4f}")
        print(f"  opt               : {g0.opt.item():.4f}")
        print(f"  amélioration      : {(l1_init - py_best) / l1_init * 100:.1f}%")

    if mode == "overfit" and graphs:
        overfit_one_graph(graphs[0])
