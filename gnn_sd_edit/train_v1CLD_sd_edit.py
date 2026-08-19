"""
train_v1CLD_sd_edit.py — GNN "style sd_edit" (arêtes en étoile) pour la borne
V1CLD sur CC-MKP.

Contexte et décisions de design (session du 2026-08-16, cf. discussion avec
l'utilisateur avant l'écriture de ce fichier) :

  Un prototype existait déjà : gnn/train_gnn_v1cld.py. Il construit un graphe
  à un nœud par triplet (scénario, contrainte, item) — l'idée de base demandée
  ici — mais avec deux limites jamais corrigées :
    1. arêtes en CLIQUE COMPLÈTE sur les 3 axes (coût O(n_g²·m + N_z²·n_g +
       m²·n_g) par graphe) — jamais mis à l'échelle (pas de
       train_scaled_v1cld.py, aucun checkpoint produit), contrairement à V1LD
       qui est passé de la clique à l'étoile (sd_edit) précisément pour tenir
       à grande échelle.
    2. features de nœud incluant k/n_g, j/m, i/N_z (position BRUTE dans le
       tableau, sans lien avec le problème) — exactement le défaut que V1LD a
       corrigé dans sa propre variante sd_edit ("rangs arbitraires, brisent
       l'invariance par permutation", cf. train_v1LD_sd_edit.py).

  Ce fichier reconstruit le raisonnement plutôt que de copier le prototype :

  Nœuds : un par triplet (i, j, k) avec
            i ∈ scénarios LIBRES (z_mask==2)   → N_z
            j ∈ [0, m)                          → m (toutes les contraintes,
                                                    m est petit et fixe, pas
                                                    besoin de filtrer)
            k ∈ items LIBRES (x_mask==2)        → n_g
  Total : N_z * m * n_g nœuds. Indexé par i*(m*n_g) + j*n_g + k (scénario
  majeur, puis contrainte, puis item) — même convention que le prototype,
  nécessaire pour dériver les groupements de pooling par arithmétique pure
  (cf. GNN.forward), et pour une éventuelle ré-implémentation C++ (même
  arithmétique d'indexation que gnn_sd_edit_native.hpp::build_graph_cpp).

  Pourquoi un nœud par (scénario, contrainte, item) et pas juste (scénario,
  item) comme V1LD ? Parce que V1CLD résout ζ_i par DP 1D INDÉPENDAMMENT pour
  CHAQUE contrainte j (contrairement à V1LD qui résout un knapsack joint sur
  les m contraintes pour chaque scénario) — le multiplicateur de Lagrange a
  donc un indice de plus : λ[i,j,k] au lieu de λ[i,k]. Le graphe suit cette
  structure duale plutôt que de l'imposer arbitrairement.

  Features de nœud (6, cf. discussion — volontairement minimal pour ce
  premier jet, des raccourcis statistiques type V1LD (scen_min_cap,
  f_A_mean_scen, etc.) pourront être ajoutés PAR ABLATION si l'entraînement
  en montre le besoin, pas d'emblée) :
    [c_k/c_scale, A_ijk, clip(A_ijk/b_res_ij,10), log1p((c_k/c_scale)/A_ijk),
     dantzig_rank, b_res_ij/b_ij]
  dantzig_rank est un rang DÉRIVÉ DES DONNÉES (position de k dans le
  classement décroissant d'efficacité c_k/A_ijk au sein du couple (i,j)) — il
  reste attaché au bon item si on permute le stockage, contrairement à k/n_g.
  c_scale = max(curr_cost + Σc_g, 1.0) : normalisation de l'échelle des
  profits (portée de V1LD), pour que les features/la sortie du réseau restent
  d'ordre 1 quel que soit la taille de l'instance — important pour
  l'entraînement multi-config (mélange de n/N différents dans un batch).

  Arêtes (3 types, chacune ne relie que des nœuds qui diffèrent sur UN SEUL
  axe) :
    - item      (i,j fixés, k varie)  : CLIQUE complète, poids 1/n_g.
                  Coût O(n_g²) par (i,j) — n_g reste petit (≤ n), pas un
                  problème de scalabilité comme les deux autres axes.
    - scénario  (j,k fixés, i varie)  : ÉTOILE ancrée à i=0, poids 1 — comme
                  V1LD sd_edit / solver_swann_darius.cpp. C'est l'axe qui
                  peut devenir très grand (N jusqu'à 1000+), la clique
                  (O(N_z²)) y est prohibitive.
    - contrainte (i,k fixés, j varie) : ÉTOILE ancrée à j=0, poids 1. m est
                  petit et fixe (m=5 dans ce projet) — une clique y coûterait
                  seulement O(m²)=O(25), donc ce n'est PAS un enjeu de
                  scalabilité ici. L'étoile est reprise par cohérence de
                  design avec les deux autres axes (décision utilisateur),
                  pas par nécessité de coût sur cet axe précis — à garder en
                  tête si m devait un jour ne plus être petit/fixe.

  Pooling : multi-vues comme V1LD sd_edit, étendu à 3 axes — nœud→item,
  nœud→scénario, nœud→contrainte (NOUVEAU), nœud→graphe, tous concaténés
  (avec le nœud lui-même et le contexte) dans la tête. C'est l'extension
  naturelle du pooling de V1LD (qui n'avait que 2 vues, item+scénario,
  puisque la contrainte y est aplatie dans les features) à un axe de plus.

  Contexte (9, identique à V1LD sd_edit — rien de spécifique à l'axe
  contrainte à ajouter puisque m est fixe/petit et déjà implicite dans
  data.m) :
    [depth/n, frac_fixés, tightest, curr_cost/c_scale, epsilon,
     n_fixed_1_z/N, log(n_g), log(N_z), N_z/n_g]

  Architecture du réseau : embed → GatedGraphConv×2 (+LayerNorm+skip) → proj
  → pooling multi-vues → tête. Le cœur (embed/conv/proj/tête) est repris À
  L'IDENTIQUE de V1LD sd_edit (le message-passing ne "sait" pas ce que
  représentent les arêtes, seulement leur topologie/poids) ; seuls NODE_DIM
  (6, pas 7m+5 : la contrainte n'est plus aplatie dans les features) et le
  pooling (3 vues au lieu de 2) changent.

  Cible : implicite — minimiser L1(λ) V1CLD (théorème de l'enveloppe), comme
  V1LD et comme le prototype. lagrangian_v1cld()/verify_v1cld() ci-dessous
  sont repris QUASI VERBATIM de gnn/train_gnn_v1cld.py (math déjà validée,
  corrigée le 2026-08-13 pour le centrage par scénario — cf.
  PREUVE_v1CLD_centrage_par_scenario.md) : c'est la partie du prototype qu'on
  NE remet PAS en cause, seule la construction du graphe change.

  Compatibilité C++ native (prévue dès maintenant, cf. décision utilisateur) :
  l'indexation des nœuds (i*(m*n_g)+j*n_g+k) et les 3 règles d'arêtes
  (clique/étoile/étoile) sont volontairement de l'arithmétique simple,
  réimplémentable en C++ sans dépendance Python — même schéma que
  gnn_sd_edit_native.hpp/.cpp pour V1LD. Pas d'implémentation C++ dans ce
  fichier (travail Python/recherche pour l'instant), mais aucun choix ici ne
  devrait bloquer un futur portage.

  Fichier séparé de gnn/train_gnn_v1cld.py et de gnn_sd_edit/train_v1LD_sd_edit.py
  — AUCUNE modification de ces deux fichiers (même convention que le passage
  V1LD homogène → sd_edit).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

# Importer uniquement le sous-module nécessaire pour éviter le conflit
# torch_geometric.datasets → transformers → tensorflow → NumPy 1.x
import importlib, sys
_pyg_data_mod = importlib.import_module("torch_geometric.data")
pyg_data = _pyg_data_mod
DataLoader = importlib.import_module("torch_geometric.loader").DataLoader
from torch_geometric.nn import GatedGraphConv
from torch_geometric.nn import global_mean_pool, global_max_pool

# ── Solveur B&B C++ (optionnel, fallback Python si absent) ───────────────────
# Réutilise le binaire compilé de gnn/ (non dupliqué ici, inchangé par ce fichier).
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
# Chargement des instances (format partagé avec V1LD — inchangé)
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
# Construction du graphe pour un nœud B&B — cœur de ce fichier
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(inst: dict, node_rec: dict) -> Optional[pyg_data.Data]:
    """
    Construit un graphe torch_geometric pour la borne V1CLD à partir d'une
    instance et d'un état de nœud B&B.

    node_rec attendu (même format que train_v1LD_sd_edit.py — "Format B") :
      x_mask[k] ∈ {0,1,2} : 0=fixé_0, 1=fixé_1, 2=libre
      z_mask[i] ∈ {0,1,2} : 0=fixé_0, 1=fixé_1, 2=libre
      b_res[N, m]          : capacités résiduelles, précalculées par l'appelant
                              (cf. LazyGraphDataset) depuis instances + x_mask

    Nœuds : N_z * m * n_g triplets (scénario libre i, contrainte j, item libre k).
    Features (6) : cf. docstring de module.
    Arêtes : item=clique(1/n_g), contrainte=étoile@0(1), scénario=étoile@0(1).
    Contexte (9) : cf. docstring de module.

    Retourne None si n_g == 0 ou N_z == 0 (rien à borner à ce nœud).
    """
    n, m, N  = inst["n"], inst["m"], inst["N"]
    c, b, A  = inst["c"], inst["b"], inst["A"]
    x_mask   = node_rec["x_mask"]   # (n,)  int8
    b_res    = node_rec["b_res"]    # (N, m)  float32
    z_mask   = node_rec["z_mask"]   # (N,)  int8

    active_x   = np.where(x_mask == 2)[0]           # items libres
    free_z     = np.where(z_mask == 2)[0]            # scénarios libres
    n_fixed_1_z = int((z_mask == 1).sum())           # scénarios déjà satisfaits (z=1)

    n_g = len(active_x)
    N_z = len(free_z)
    if n_g == 0 or N_z == 0:
        return None

    # ── Sous-matrices : scénarios libres × TOUTES les contraintes × items libres ──
    b_res_z = b_res[free_z, :]                        # (N_z, m)
    b_z     = b[free_z, :]                             # (N_z, m)
    A_g     = A[np.ix_(free_z, range(m), active_x)]    # (N_z, m, n_g)
    c_g     = c[active_x]                               # (n_g,)

    curr_cost = node_rec["curr_cost"]
    c_scale   = max(float(curr_cost) + float(c_g.sum()), 1.0)

    # ── Features des nœuds (N_z * m * n_g, 6) ────────────────────────────────
    c_brd    = np.broadcast_to(c_g[None, None, :], (N_z, m, n_g)).astype(np.float32)
    bres_brd = np.broadcast_to(b_res_z[:, :, None], (N_z, m, n_g)).astype(np.float32)
    b_brd    = np.broadcast_to(b_z[:, :, None],     (N_z, m, n_g)).astype(np.float32)

    f_c    = c_brd / c_scale                                                   # profit normalisé
    f_A    = A_g.astype(np.float32)                                            # poids brut
    f_frac = np.minimum(A_g / np.maximum(bres_brd, 1e-9), 10.0).astype(np.float32)  # clippé à 10, comme V1LD
    f_eff  = np.log1p(f_c / np.maximum(f_A, 1e-9)).astype(np.float32)          # efficacité log-stabilisée

    # Rang Dantzig normalisé ∈ [0,1] : position de k dans le classement
    # décroissant d'efficacité c_k/A_ijk au sein du couple (i,j). Dérivé des
    # données (suit l'item si on permute), PAS une position brute — gardé.
    eff_dantzig  = c_brd / np.maximum(A_g, 1e-9)
    dantzig_rank = (np.argsort(np.argsort(-eff_dantzig, axis=2), axis=2)
                    / max(n_g - 1, 1)).astype(np.float32)                      # (N_z, m, n_g)

    f_cap = (b_res_z / np.maximum(b_z, 1e-9))                                  # (N_z, m)
    f_cap_brd = np.broadcast_to(f_cap[:, :, None], (N_z, m, n_g)).astype(np.float32)

    X = np.stack([f_c, f_A, f_frac, f_eff, dantzig_rank, f_cap_brd], axis=-1)
    X = X.reshape(-1, 6).astype(np.float32)   # (N_z*m*n_g, 6)

    # ── Arêtes (vectorisées, pas de boucle Python) ───────────────────────────
    # Indice de nœud : flat(i,j,k) = i*(m*n_g) + j*n_g + k

    # 1. Item varie (i,j fixés) : CLIQUE complète, poids 1/n_g — n_g reste
    #    petit, pas un enjeu de scalabilité (contrairement aux 2 autres axes).
    ki, kj  = np.triu_indices(n_g, k=1)
    E_k     = len(ki)
    base_ij = (np.repeat(np.arange(N_z, dtype=np.int64), m) * (m * n_g)
               + np.tile(np.arange(m, dtype=np.int64), N_z) * n_g)   # (N_z*m,)
    base_rep = np.repeat(base_ij, E_k)
    ki_rep   = np.tile(ki, N_z * m)
    kj_rep   = np.tile(kj, N_z * m)
    src1 = np.concatenate([base_rep + ki_rep, base_rep + kj_rep])
    dst1 = np.concatenate([base_rep + kj_rep, base_rep + ki_rep])
    w1   = np.full(len(src1), 1.0 / n_g, dtype=np.float32)

    # 2. Contrainte varie (i,k fixés) : ÉTOILE ancrée à j=0, poids 1 — comme
    #    l'étoile scénario de V1LD sd_edit, appliquée ici par cohérence de
    #    design (m est petit/fixe : une clique coûterait à peine plus cher,
    #    cf. docstring de module).
    i_ik = np.repeat(np.arange(N_z, dtype=np.int64), n_g)   # (N_z*n_g,)
    k_ik = np.tile(np.arange(n_g, dtype=np.int64), N_z)
    anchor_ik = i_ik * (m * n_g) + k_ik                      # j=0 → terme nul
    if m > 1:
        j_spokes  = np.arange(1, m, dtype=np.int64)                       # (m-1,)
        spoke_idx = anchor_ik[:, None] + (j_spokes * n_g)[None, :]       # (N_z*n_g, m-1)
        anchor_rep = np.broadcast_to(anchor_ik[:, None], spoke_idx.shape)
        src2 = np.concatenate([anchor_rep.ravel(), spoke_idx.ravel()])
        dst2 = np.concatenate([spoke_idx.ravel(), anchor_rep.ravel()])
    else:
        src2 = np.empty(0, dtype=np.int64)
        dst2 = np.empty(0, dtype=np.int64)
    w2 = np.full(len(src2), 1.0, dtype=np.float32)

    # 3. Scénario varie (j,k fixés) : ÉTOILE ancrée à i=0, poids 1 — identique
    #    à V1LD sd_edit / solver_swann_darius.cpp (axe potentiellement grand,
    #    N jusqu'à 1000+, la clique y serait O(N_z²)).
    j_jk = np.repeat(np.arange(m, dtype=np.int64), n_g)     # (m*n_g,)
    k_jk = np.tile(np.arange(n_g, dtype=np.int64), m)
    anchor_jk = j_jk * n_g + k_jk                            # i=0 → terme nul
    if N_z > 1:
        i_spokes   = np.arange(1, N_z, dtype=np.int64)                    # (N_z-1,)
        spoke_idx3 = anchor_jk[None, :] + (i_spokes * (m * n_g))[:, None]  # (N_z-1, m*n_g)
        anchor_rep3 = np.broadcast_to(anchor_jk[None, :], spoke_idx3.shape)
        src3 = np.concatenate([anchor_rep3.ravel(), spoke_idx3.ravel()])
        dst3 = np.concatenate([spoke_idx3.ravel(), anchor_rep3.ravel()])
    else:
        src3 = np.empty(0, dtype=np.int64)
        dst3 = np.empty(0, dtype=np.int64)
    w3 = np.full(len(src3), 1.0, dtype=np.float32)

    edge_index  = torch.tensor(
        np.stack([np.concatenate([src1, src2, src3]),
                  np.concatenate([dst1, dst2, dst3])]),
        dtype=torch.long)
    edge_weight = torch.tensor(np.concatenate([w1, w2, w3]), dtype=torch.float32)

    # ── Contexte (features au niveau du graphe, 9 — identique V1LD sd_edit) ──
    depth_norm     = float(node_rec["depth"]) / max(n, 1)
    frac_fixed     = 1.0 - n_g / n
    curr_cost_norm = float(curr_cost) / c_scale
    tightest       = float(np.min(f_cap))

    context = torch.tensor(
        [depth_norm, frac_fixed, tightest, curr_cost_norm,
         float(inst["epsilon"]), n_fixed_1_z / N,
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
        N             = N,       # N original (pour p_i = 1/N)
        N_z           = N_z,     # scénarios libres dans ce nœud
        m             = m,
        n_fixed_1_z   = n_fixed_1_z,
        epsilon       = torch.tensor(float(inst["epsilon"]), dtype=torch.float32),
        # Matrices pour le calcul de L1(λ) (lagrangian_v1cld ci-dessous)
        A_g     = torch.from_numpy(A_g.copy()),      # (N_z, m, n_g)
        b_res_g = torch.from_numpy(b_res_z.copy()),  # (N_z, m)
        c_g     = torch.from_numpy(c_g.copy()),      # (n_g,)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Chargement lazy : enregistrements bruts + dataset à la volée
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_nodes(
    instances_file: str | Path,
    nodes_file: str | Path,
    max_records: Optional[int] = None,
) -> tuple[dict, list]:
    """
    Parse instances + nœuds sans construire les graphes PyG.
    Supporte Format B (5+n+N) et Format C (3+n+N, bab_mkp_gecode_base) — même
    format de collecte que V1LD, le format des nœuds B&B ne dépend pas de la
    borne utilisée pour guider la recherche pendant la collecte.
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

    def __len__(self):          return len(self.graphs)
    def __getitem__(self, idx): return self.graphs[idx]
    def inst_ids(self):         return self._inst_ids


# ─────────────────────────────────────────────────────────────────────────────
# Modèle GNN — prédit λ[i,j,k] par nœud (scénario libre, contrainte, item libre)
# ─────────────────────────────────────────────────────────────────────────────

class GNN(nn.Module):
    """
    Prédit λ[i,j,k] ∈ R par nœud. Pas de softplus en sortie (λ ∈ R dans V1CLD,
    comme V1LD).

    Architecture (NODE_DIM=6, contexte 9-dim, max_pool optionnel) :
      embed      : X(6) → hidden → ReLU → hidden → ReLU → LayerNorm
      conv1      : GatedGraphConv(hidden, 2 it.) + LayerNorm
      conv2      : GatedGraphConv(hidden, 2 it.) + LayerNorm + skip(h1)
      proj       : hidden → 4h → ReLU → 2h                         → h_v
      pool item       : nœud → item k        [mean(+max)] → h_item
      pool scénario   : nœud → scénario i    [mean(+max)] → h_scen
      pool contrainte : nœud → contrainte j  [mean(+max)] → h_constraint  (NOUVEAU)
      pool graphe     : nœud → graphe        [mean]       → g_pool
      head       : [h_v | h_item | h_scen | h_constraint | g_pool | ctx]
                   → h → ReLU → h → ReLU → 1

    Repris à l'identique de train_v1LD_sd_edit.py::GNN sauf NODE_DIM (6, pas
    7m+5 — la contrainte n'est plus aplatie dans les features) et l'ajout de
    la vue de pooling par contrainte (extension directe du pooling multi-vues
    de V1LD à un axe de plus). Pas de paramètre `m` au constructeur : NODE_DIM
    ne dépend plus de m (contrairement à V1LD), m n'intervient qu'à l'exécution
    via data.m pour dériver les groupements de pooling.
    """
    NODE_DIM    = 6
    CONTEXT_DIM = 9  # [depth/n, frac_fixed, tightest, curr_cost/c_scale, epsilon, n_fixed_1_z/N, log(n_g), log(N_z), N_z/n_g]

    def __init__(self, hidden: int = 64, use_max_pool: bool = False):
        super().__init__()
        self.hidden       = hidden
        self.use_max_pool = use_max_pool

        self.embed = nn.Sequential(
            nn.Linear(self.NODE_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),        nn.ReLU(),
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

        # pool item / scén / contrainte : 2h chacun (mean seul) ou 4h (mean+max)
        pool_dim = 4 * hidden if use_max_pool else 2 * hidden

        # h_v(2h) + h_item(pool_dim) + h_scen(pool_dim) + h_constraint(pool_dim) + g_pool(2h) + ctx(9)
        head_in = 2 * hidden + 3 * pool_dim + 2 * hidden + self.CONTEXT_DIM
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

        ctx = data.context
        if ctx.dim() == 1:
            ctx = ctx.unsqueeze(0)
        ctx_norm = self.norm_ctx(ctx)                                  # (1 or B, 9)

        # ── Message passing (identique à V1LD sd_edit) ──────────────────────
        h_embed = self.norm_embed(self.embed(x))
        h  = h_embed
        h1 = self.norm1(self.conv1(h,  ei, ew))
        h2 = self.norm2(self.conv2(h1, ei, ew)) + h1
        h_v = self.proj(h2)                                            # (V, 2h)

        # ── Groupements de pooling par arithmétique d'indexation ────────────
        # Nœuds ordonnés i*(m*n_g) + j*n_g + k (scénario majeur, puis
        # contrainte, puis item) ⇒ à local_idx dans son graphe :
        #   k = local_idx % n_g
        #   reste = local_idx // n_g  =  i*m + j
        #   j = reste % m
        #   i = reste // m
        # NB : suppose n_g ET m uniformes sur le batch (m l'est toujours dans
        # ce projet — fixe globalement ; n_g varie par nœud B&B mais est
        # supposé constant au sein d'UN batch, même hypothèse que V1LD).
        n_g = data.n_g if isinstance(data.n_g, int) else int(data.n_g.item())
        m   = data.m   if isinstance(data.m,   int) else int(data.m.item())

        counts   = torch.bincount(batch, minlength=n_graphs)
        node_off = torch.zeros(n_graphs, dtype=torch.long, device=dev)
        node_off[1:] = counts[:-1].cumsum(0)
        local_idx = torch.arange(n_v, dtype=torch.long, device=dev) - node_off[batch]

        # -- item (k) --
        item_off   = torch.arange(n_graphs, device=dev) * n_g
        item_batch = item_off[batch] + (local_idx % n_g)                # (V,)

        # -- contrainte (j) --
        remaining        = local_idx // n_g                             # i*m + j
        constraint_off   = torch.arange(n_graphs, device=dev) * m
        constraint_batch = constraint_off[batch] + (remaining % m)      # (V,)

        # -- scénario (i) --
        n_scen_per = counts // (m * n_g)
        scen_off   = torch.zeros(n_graphs, dtype=torch.long, device=dev)
        scen_off[1:] = n_scen_per[:-1].cumsum(0)
        scen_batch = scen_off[batch] + (remaining // m)                 # (V,)

        def _pool(group_batch):
            h_mean = global_mean_pool(h_v, group_batch)
            if self.use_max_pool:
                h_max = global_max_pool(h_v, group_batch)
                return torch.cat([h_mean, h_max], dim=-1)
            return h_mean

        h_item = _pool(item_batch);             h_item_exp = h_item[item_batch]
        h_scen = _pool(scen_batch);             h_scen_exp = h_scen[scen_batch]
        h_cons = _pool(constraint_batch);       h_cons_exp = h_cons[constraint_batch]

        g_pool = global_mean_pool(h_v, batch)                            # (B, 2h)
        g_exp  = g_pool[batch]                                            # (V, 2h)
        ctx_exp = ctx_norm[batch]                                          # (V, 9)

        cat   = torch.cat([h_v, h_item_exp, h_scen_exp, h_cons_exp, g_exp, ctx_exp], dim=-1)
        h_out = self.head(cat).squeeze(-1)

        # Rescale : le réseau prédit une quantité normalisée d'ordre 1, on la
        # ramène à l'échelle réelle de λ[i,j,k] (dont l'ordre de grandeur
        # naturel est c_scale * p_ij = c_scale/(N*m), cf. lagrangian_v1cld).
        lam = h_out * data.c_scale / (float(data.N) * float(data.m))

        if return_hidden:
            extra = {
                "x": x, "h_embed": h, "h1": h1, "h2": h2, "h_v": h_v,
                "h_item": h_item, "item_batch": item_batch,
                "h_scen": h_scen, "scen_batch": scen_batch,
                "h_cons": h_cons, "constraint_batch": constraint_batch,
                "g_pool": g_pool, "h_out": h_out,
            }
            return lam, extra
        return lam


# ─────────────────────────────────────────────────────────────────────────────
# Borne L1(λ) V1CLD — repris quasi verbatim de gnn/train_gnn_v1cld.py
# (math déjà validée, PAS remise en cause ici — seule la construction du
# graphe change dans ce fichier)
# ─────────────────────────────────────────────────────────────────────────────

def _solve_zeta_bb(cost: np.ndarray, a_row: np.ndarray, b_j: float):
    """
    0/1 knapsack exact par B&B avec borne de Dantzig (fallback Python si le
    binaire C++ solve_zeta_batch est indisponible).
    Même algorithme que solve_zeta_i_exact_bb dans mkp_subproblems.hpp.
    Retourne (valeur optimale, x* ∈ {0,1}^n).
    """
    n = len(cost)
    x = np.zeros(n, dtype=np.float32)
    val_free = 0.0

    items = []
    for k in range(n):
        if a_row[k] < 1e-9:
            if cost[k] > 0.0:
                x[k] = 1.0
                val_free += cost[k]
        elif cost[k] > 0.0:
            items.append(k)

    if not items:
        return val_free, x

    items.sort(key=lambda k: cost[k] / a_row[k], reverse=True)
    best_val = [val_free]
    best_x   = [x.copy()]

    def bound(idx, cap, val):
        b = val
        c = cap
        for kk in items[idx:]:
            if a_row[kk] <= c:
                c -= a_row[kk]; b += cost[kk]
            else:
                b += cost[kk] * c / a_row[kk]
                break
        return b

    def rec(idx, cap, val, chosen):
        if val > best_val[0]:
            best_val[0] = val
            xx = x.copy()
            for kk in chosen:
                xx[kk] = 1.0
            best_x[0] = xx
        if idx == len(items):
            return
        if bound(idx, cap, val) <= best_val[0]:
            return
        k = items[idx]
        if a_row[k] <= cap:
            chosen.append(k)
            rec(idx + 1, cap - a_row[k], val + cost[k], chosen)
            chosen.pop()
        rec(idx + 1, cap, val, chosen)

    rec(0, b_j, val_free, [])
    return best_val[0], best_x[0]


def lagrangian_v1cld(lam: torch.Tensor, data: pyg_data.Data) -> torch.Tensor:
    """
    Calcule L1(λ) sur les scénarios libres, différentiable en λ.

    lam  : (N_z*m*n_g,) — sortie brute du GNN (ou tout λ candidat)
    data : graphe avec A_g (N_z, m, n_g), b_res_g (N_z, m), c_g (n_g,)

    Probabilités : p_i = 1/N (N original), p_ij = 1/(N*m).
    Seuil de sélection : Σ p_i y_i ≥ (1−ε) − n_fixed_1_z/N.

    Algorithme (théorème de l'enveloppe) :
      1. Résoudre θ et ζ → x*_θ, x*_ζ  [detach]
      2. Sélection gloutonne y*        [detach]
      3. L1 comme fonction linéaire de λ à x*, y* fixés → autograd
    """
    N_z = data.N_z
    N   = data.N
    m   = data.m
    n_g = data.n_g
    n_fixed_1_z = data.n_fixed_1_z
    dev = lam.device

    lam_3d = lam.view(N_z, m, n_g)

    A_g     = data.A_g.to(dev)
    b_res_g = data.b_res_g.to(dev)
    c_g     = data.c_g.to(dev)
    epsilon = data.epsilon.item()

    p_unif = 1.0 / N
    inv_m  = 1.0 / m
    p_ij   = p_unif * inv_m

    # Λ_sum_i = Σ_j λ[i,j] — PAR SCÉNARIO (cf. PREUVE_v1CLD_centrage_par_scenario.md)
    Lambda_sum_i = lam_3d.sum(dim=1)                       # (N_z, n_g)

    c_brd = c_g.unsqueeze(0).unsqueeze(0)                  # (1, 1, n_g)
    L_brd = (Lambda_sum_i / p_unif).unsqueeze(1)           # (N_z, 1, n_g)
    cost  = c_brd + lam_3d / p_ij - L_brd                 # (N_z, m, n_g)

    cost_np = cost.detach().cpu().numpy()
    A_np    = A_g.cpu().numpy()
    bres_np = b_res_g.cpu().numpy()

    x_theta_np = (cost_np > 0).astype(np.float32)

    if _USE_CPP_BNB:
        x_zeta_np = _cpp_solve_zeta_batch(
            cost_np.reshape(N_z * m, n_g),
            A_np.reshape(N_z * m, 1, n_g),
            bres_np.reshape(N_z * m, 1),
        ).reshape(N_z, m, n_g)
    else:
        x_zeta_np = np.zeros_like(x_theta_np)
        for i in range(N_z):
            for j in range(m):
                _, x_z = _solve_zeta_bb(cost_np[i, j], A_np[i, j], bres_np[i, j])
                x_zeta_np[i, j] = x_z

    theta_np = np.sum(np.maximum(cost_np, 0.0), axis=2)   # (N_z, m)
    zeta_np  = np.sum(cost_np * x_zeta_np,      axis=2)   # (N_z, m)
    Delta_np = np.maximum(0.0, theta_np - zeta_np).mean(axis=1)  # (N_z,)

    required_prob = max(0.0, (1.0 - epsilon) - n_fixed_1_z * p_unif)
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

    x_theta_t = torch.tensor(x_theta_np, device=dev)
    x_zeta_t  = torch.tensor(x_zeta_np,  device=dev)
    y_star_t  = torch.tensor(y_star,     device=dev)      # (N_z,)

    theta_lam = (cost * x_theta_t).sum(dim=2)                       # (N_z, m)
    delta_lam = (cost * (x_theta_t - x_zeta_t)).sum(dim=2)         # (N_z, m)

    L1_theta = p_ij * theta_lam.sum()
    val_lam  = (p_unif * inv_m * y_star_t.unsqueeze(1) * delta_lam).sum()
    return L1_theta - val_lam


def verify_v1cld(
    data: pyg_data.Data,
    T: int = 500,
    precision: float | None = None,
    patience: int | None = None,
    lam_init: np.ndarray | None = None,
    return_n_iters: bool = False,
    return_lam: bool = False,
) -> float | tuple:
    """
    Référence indépendante : reproduit l'algorithme du solveur V1CLD
    (subgradient 1/√(t+1)) en Python pur et retourne la meilleure L1 vue.
    Sert à vérifier que le GNN converge vers une borne au moins aussi bonne,
    et (return_lam=True) à générer des cibles λ* pour le warmup supervisé
    (cf. train_scaled_v1CLD_sd_edit.py::warmup_on_roots, même usage que
    verify_v1ld dans train_v1LD_sd_edit.py).

    lam_init : (N_z, m, n_g) float64 ; None = cold start λ=0.
    """
    N_z = data.N_z
    N   = data.N
    m   = data.m
    n_g = data.n_g
    n_fixed_1_z = data.n_fixed_1_z
    epsilon   = data.epsilon.item()
    A_np      = data.A_g.numpy()     # (N_z, m, n_g)
    b_np      = data.b_res_g.numpy() # (N_z, m)
    c_np      = data.c_g.numpy()     # (n_g,)

    p_unif = 1.0 / N
    inv_m  = 1.0 / m
    p_ij   = p_unif * inv_m
    required_prob = max(0.0, (1.0 - epsilon) - n_fixed_1_z * p_unif)

    lam = lam_init.copy() if lam_init is not None else np.zeros((N_z, m, n_g), dtype=np.float64)
    best     = np.inf
    best_lam = lam.copy()
    no_improve = 0
    n_iters = T

    for t in range(T):
        Lsum_i = lam.sum(axis=1)                                # (N_z, n_g)
        cost = c_np + lam / p_ij - (Lsum_i / p_unif)[:, None, :]  # (N_z, m, n_g)

        x_theta = (cost > 0).astype(np.float64)

        # Batché via le solveur C++ quand disponible (cf. lagrangian_v1cld) —
        # sans ce chemin, T itérations x N_z*m sous-problèmes en B&B Python
        # pur devient prohibitif dès que N_z ou m grandit un peu (c'est ce
        # qui manquait dans gnn/train_gnn_v1cld.py::verify_v1cld, jamais
        # corrigé faute d'avoir été mis à l'échelle — cf. docstring de module).
        if _USE_CPP_BNB:
            x_zeta = _cpp_solve_zeta_batch(
                cost.reshape(N_z * m, n_g).astype(np.float64),
                A_np.reshape(N_z * m, 1, n_g).astype(np.float64),
                b_np.reshape(N_z * m, 1).astype(np.float64),
            ).reshape(N_z, m, n_g).astype(np.float64)
        else:
            x_zeta = np.zeros((N_z, m, n_g), dtype=np.float64)
            for i in range(N_z):
                for j in range(m):
                    _, xz = _solve_zeta_bb(cost[i, j].astype(np.float32),
                                           A_np[i, j].astype(np.float32),
                                           float(b_np[i, j]))
                    x_zeta[i, j] = xz

        theta_vals = np.sum(np.maximum(cost, 0.0), axis=2)    # (N_z, m)
        delta_ij   = np.maximum(0.0, theta_vals - np.sum(cost * x_zeta, axis=2))
        Delta      = delta_ij.mean(axis=1)                     # (N_z,)

        order  = np.argsort(Delta)
        y_star = np.zeros(N_z); cumprob = 0.0
        for idx in order:
            if required_prob <= 0.0:
                break
            y_star[idx] = 1.0; cumprob += p_unif
            if cumprob >= required_prob - 1e-9: break

        val = np.sum(p_unif * y_star * Delta)
        L1  = p_ij * theta_vals.sum() - val
        if L1 < best - (precision or 0.0):
            best = L1
            best_lam = lam.copy()
            no_improve = 0
        else:
            no_improve += 1
        if patience is not None and precision is not None and no_improve >= patience:
            n_iters = t + 1
            break

        active = y_star > 0.5
        x_bar  = np.zeros(n_g)
        for i in range(N_z):
            for j in range(m):
                x_bar += p_ij * (x_zeta[i, j] if active[i] else x_theta[i, j])

        step = 1.0 / np.sqrt(t + 1)
        for i in range(N_z):
            for j in range(m):
                xij = x_zeta[i, j] if active[i] else x_theta[i, j]
                lam[i, j] += step * (x_bar - xij)

    if return_n_iters and return_lam:
        return best, n_iters, best_lam
    if return_lam:
        return best, best_lam
    if return_n_iters:
        return best, n_iters
    return best


def overfit_one_graph(data: pyg_data.Data, n_epochs: int = 1000, lr: float = 1e-3):
    """
    Sanity-check d'architecture : entraîne le GNN sur un seul graphe avec la
    vraie borne V1CLD comme loss. Doit converger vers ≤ L1(λ=0), sinon
    l'architecture (graphe/pooling/head) a un problème avant même de passer
    à l'entraînement multi-instances.
    """
    device = torch.device("cpu")
    data   = data.to(device)

    model     = GNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)

    opt_val = data.opt.item()
    with torch.no_grad():
        lam0    = torch.zeros(data.N_z * data.m * data.n_g)
        l1_init = lagrangian_v1cld(lam0, data).item()

    print(f"\n{'='*60}")
    print(f"Overfit — n_g={data.n_g}  N_z={data.N_z}/{data.N}  m={data.m}")
    print(f"  opt={opt_val:.2f}   L1(λ=0)={l1_init:.2f}")
    print(f"  graphe : {data.x.shape[0]} nœuds, {data.edge_index.shape[1]} arêtes")
    print(f"{'='*60}")
    print(f"{'epoch':>6}  {'L1(λ_GNN)':>12}  {'gap_vs_opt%':>12}  {'gap_vs_L1_0%':>13}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        lam  = model(data)
        loss = lagrangian_v1cld(lam, data)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            b     = loss.item()
            g_opt = (b - opt_val) / opt_val * 100
            g_l0  = (b - l1_init) / l1_init * 100
            print(f"{epoch:6d}  {b:12.4f}  {g_opt:+11.2f}%  {g_l0:+12.2f}%")

    print(f"{'='*60}")
    print(f"Borne finale : {loss.item():.4f}  (opt={opt_val:.2f}, L1_0={l1_init:.2f})")


# ─────────────────────────────────────────────────────────────────────────────
# Test rapide / smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/train_v1CLD")
    n_val    = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    mode     = sys.argv[3] if len(sys.argv) > 3 else "inspect"

    inst_file  = data_dir / f"instances_n{n_val}.txt"
    nodes_file = data_dir / f"nodes_n{n_val}.txt"

    instances, records = load_raw_nodes(inst_file, nodes_file, max_records=5)
    lazy = LazyGraphDataset(instances, records)

    graphs = []
    for i in range(len(lazy)):
        try:
            graphs.append(lazy[i])
        except ValueError as e:
            print(f"  [skip] {e}")

    for i, g in enumerate(graphs):
        print(
            f"  graph {i:3d} | nodes={g.x.shape[0]:6d} | "
            f"edges={g.edge_index.shape[1]:8d} | "
            f"n_g={g.n_g} N_z={g.N_z}/{g.N} m={g.m} | "
            f"depth={int(g.context[0].item() * max(g.n_g,1))} | opt={g.opt.item():.2f}"
        )

    if mode in ("verify", "overfit") and graphs:
        g0 = graphs[0]
        lam0    = torch.zeros(g0.N_z * g0.m * g0.n_g)
        l1_init = lagrangian_v1cld(lam0, g0).item()
        print(f"\n── Vérification V1CLD (T=100 itérations subgradient) ──")
        print(f"  L1(λ=0)          : {l1_init:.4f}")
        py_best = verify_v1cld(g0, T=100)
        print(f"  subgradient T=100 : {py_best:.4f}")
        print(f"  opt               : {g0.opt.item():.4f}")
        print(f"  amélioration      : {(l1_init - py_best) / l1_init * 100:.1f}%")

    if mode == "overfit" and graphs:
        overfit_one_graph(graphs[0], n_epochs=500)
