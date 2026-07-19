"""
train_v1LD_ssp.py — Portage de gnn_sd_edit/train_v1LD_sd_edit.py (MKP) au SSP
chance-constrained (shift scheduling, cf. ssp_stochastic/). Dupliqué depuis le
fichier MKP puis adapté (cf. discussion de conception) : la structure globale
(grille scénario × "item", clique intra-scénario + étoile inter-scénario, GNN
avec pooling item/scénario/graphe, borne L1(λ) par théorème de l'enveloppe) est
strictement identique. Ce qui change :

  - "item" (k_idx du MKP) devient une paire libre (employé e, période t) —
    l'axe symbole σ∈{O,P,R,W} n'est PAS un axe de graphe (comme l'axe
    contrainte j du MKP ne l'était pas) : c'est un canal de SORTIE (λ a 4
    composantes par nœud au lieu d'1).
  - Les features par nœud sont des données brutes de l'instance (salaire,
    demande, comptage d'employés déjà fixés à W) — PAS de quantité dérivée
    d'une DP/solve (cf. discussion : delta[e,t] rejeté, on laisse le
    message-passing faire ce travail implicitement).
  - ζ_i (sous-problème V1LD) n'est plus un knapsack B&B maison mais un appel
    CP-SAT (SSPShiftProblem::solve_zeta, via gnn_ssp/bnb_solver_ssp.py) —
    BEAUCOUP plus cher (cf. micro-benchmark : ~11s pour un seul appel à
    E=8,T=10 avec un DFA jouet à 3 états, alors que le DFA réel a plus
    d'états). Assumé volontairement malgré le coût, pour valider l'architecture
    sur petites instances avant de reconsidérer la faisabilité en B&B réel.

Format des fichiers d'entrée (générés par bab_ssp_gecode_base_collect, cf.
solver_bab/ssp/collect_ssp_base.hpp) :

  instances_E{E}_T{T}_N{N}.txt — une ligne par instance résolue à optimalité :
    inst_id, E, T, N, epsilon, opt,
    initial, n_states, n_accepting, accepting[0..n_accepting-1],
    n_transitions, (src, sym, dst) × n_transitions,
    salary[0..T-1],
    demand[i][t]  (i=0..N-1, t=0..T-1, row-major)

  nodes_E{E}_T{T}_N{N}.txt — une ligne par nœud B&B collecté (Format C, sans
  borne Lagrangienne — la collecte tourne en CP pur) :
    inst_id, depth, curr_cost, x_mask[0..E*T-1], z_mask[0..N-1]
  x_mask : 0=O,1=P,2=R,3=W fixé ; 4=libre.  z_mask : 0/1 fixé, 2=libre.
  Indexation supposée x_mask[e*T+t] (e majeur, t mineur — cf. idx(e,t,σ) de
  ssp_problem.hpp) : À VÉRIFIER empiriquement si les résultats sont incohérents.

Structure du graphe produit (un graphe par nœud B&B) :
  - Scénarios libres  : free_z = {i : z_mask[i] == 2} ; N_z = |free_z|
                        n_fixed_z1 = |{i : z_mask[i] == 1}|
  - Paires (e,t) libres : free_et = {(e,t) : x_mask[e*T+t] == 4} ; G = |free_et|
  - Nœuds   : N_z * G paires (scénario libre i_idx, paire libre (e,t))
  - Features (6, données brutes uniquement) :
      [salary[t]/s_max, demand[i,t]/d_max, y_committed[t]/E,
       shortfall[i,t]/E, e/E, t/T]
      où y_committed[t] = nb d'employés (tous, pas seulement libres) déjà
      fixés à W à la période t ; shortfall[i,t] = max(0, demand[i,t] − y_committed[t]).
  - Arêtes  : item_varies (clique par scénario sur les paires (e,t), poids 1/G)
              scenario_varies (ÉTOILE ancrée au scénario libre 0, poids 1)
  - Sortie  : λ[i,(e,t)] ∈ R^4 (une composante par symbole σ) au lieu d'un scalaire
  - Contexte (9, même schéma que MKP) :
      [depth/(E*T), frac_fixé, tightest, curr_cost/c_scale, epsilon,
       n_fixed_z1/N, log(G), log(N_z), N_z/G]
  - Target  : minimiser L1(λ) — borne V1LD, ζ_i via CP-SAT (SSPShiftProblem)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# IMPORTANT (Compute Canada / nœuds partagés à beaucoup de cœurs) : le thread
# pool intra-op par défaut de torch (= nproc, ex. 192 sur ce cluster) crée une
# contention catastrophique dès qu'il tourne dans le même process qu'OR-Tools
# CP-SAT, MÊME si CP-SAT est configuré num_workers=1 côté C++ (cf. ssp_problem.cpp)
# — chaque epoch mesuré à 60+s au lieu de ~20ms une fois les threads torch
# plafonnés. Symptôme constaté : rien à voir avec la difficulté combinatoire
# de ζ (mesurée isolément à quelques ms) — un artefact d'environnement, pas
# un bug de cette implémentation.
torch.set_num_threads(1)

import importlib, sys
_pyg_data_mod = importlib.import_module("torch_geometric.data")
pyg_data = _pyg_data_mod
DataLoader = importlib.import_module("torch_geometric.loader").DataLoader
from torch_geometric.nn import GatedGraphConv
from torch_geometric.nn import global_mean_pool, global_max_pool

# ── Solveur ζ/θ C++ (SSPShiftProblem via CP-SAT, cf. bnb_solver_ssp.py) ──────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bnb_solver_ssp import SSPShiftProblemPy  # noqa: E402

SSP_SIGMA_SIZE = 4          # |Σ| = {O,P,R,W}, ordre alphabétique
SSP_SIGMA_W    = 3
SALARY_WORK_IDX = {0: 0, 1: 0, 2: 0, 3: 1}  # WORK_IND[σ], cf. cost.py


# ─────────────────────────────────────────────────────────────────────────────
# Chargement des instances
# ─────────────────────────────────────────────────────────────────────────────

def load_instances(instances_file: str | Path) -> dict[int, dict]:
    """
    Lit instances_E{E}_T{T}_N{N}.txt (cf. collect_ssp_base.hpp) et retourne
    un dict {inst_id: {E,T,N,epsilon,opt,transitions,accepting,initial,
    salary,demand}}.

    Format d'une ligne (vérifié contre data/train_ssp_base_s10/) :
      inst_id, E, T, N, epsilon, opt,
      initial, n_states, n_accepting, accepting[0..n_accepting-1],
      n_transitions, (src,sym,dst) × n_transitions,
      salary[0..T-1],
      demand[i][t]  (i=0..N-1, t=0..T-1, row-major)
    """
    instances: dict[int, dict] = {}
    with open(instances_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(x) for x in line.split(",")]

            inst_id = int(vals[0])
            E       = int(vals[1])
            T       = int(vals[2])
            N       = int(vals[3])
            epsilon = vals[4]
            opt     = vals[5]

            off = 6
            initial     = int(vals[off]); off += 1
            n_states    = int(vals[off]); off += 1
            n_accepting = int(vals[off]); off += 1
            accepting   = [int(v) for v in vals[off : off + n_accepting]]
            off += n_accepting

            n_transitions = int(vals[off]); off += 1
            trans_flat = vals[off : off + 3 * n_transitions]
            transitions = [
                (int(trans_flat[3*k]), int(trans_flat[3*k+1]), int(trans_flat[3*k+2]))
                for k in range(n_transitions)
            ]
            off += 3 * n_transitions

            salary = np.array(vals[off : off + T], dtype=np.float64)
            off += T

            demand = np.array(vals[off : off + N * T], dtype=np.int32).reshape(N, T)
            off += N * T

            instances[inst_id] = {
                "E": E, "T": T, "N": N,
                "epsilon": epsilon, "opt": opt,
                "initial": initial, "n_states": n_states,
                "accepting": accepting, "transitions": transitions,
                "salary": salary, "demand": demand,
                "ssp_problem": None,  # construit paresseusement, cf. get_ssp_problem()
            }
    return instances


def get_ssp_problem(inst: dict, free_z: np.ndarray, x_mask: np.ndarray) -> SSPShiftProblemPy:
    """
    Construit (ou reconstruit — pas de cache inter-nœud, cf. discussion :
    l'unrolling du DAG est O(T*Q), négligeable face au coût de solve_zeta lui-
    même) une instance SSPProblemBnB (variante masquée, cf. ssp_problem_bnb.hpp)
    restreinte aux scénarios LIBRES (indices locaux 0..N_z-1) et au masque de
    branchement du nœud courant — SANS le masque, solve_theta/solve_zeta
    ignoreraient les variables déjà fixées et donneraient une borne invalide
    pour ce nœud (cf. commentaire ssp_v1ld_wrapper.cpp).

    x_mask : (E*T,) int8, idx=e*T+t, valeurs 0..3=symbole fixé, 4=libre.
    """
    E, T = inst["E"], inst["T"]
    demand_free = inst["demand"][free_z, :]
    N_z = len(free_z)
    p_local = [1.0 / max(N_z, 1)] * N_z  # jamais utilisé par solve_theta/solve_zeta

    mask = x_mask.reshape(E, T).astype(np.int64)
    mask = np.where(mask == SSP_SIGMA_SIZE, -1, mask)   # 4 (libre) -> -1

    return SSPShiftProblemPy(
        inst["transitions"], inst["accepting"], inst["initial"],
        inst["salary"], demand_free, p_local, inst["epsilon"], inst["E"],
        mask=mask,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Construction du graphe pour un nœud B&B
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(inst: dict, node_rec: dict) -> Optional[pyg_data.Data]:
    """
    Construit un graphe torch_geometric à partir d'une instance SSP et d'un
    état de nœud B&B (Format C, cf. collect_ssp_base.hpp) :
      x_mask[e*T+t] ∈ {0,1,2,3,4} : 0..3 = symbole fixé, 4 = libre
      z_mask[i]     ∈ {0,1,2}     : 0/1 = fixé, 2 = libre

    Nœuds : N_z × G paires (scénario libre i_idx, paire libre (e,t)_idx),
    indexé i_idx*G + g_idx (MÊME convention que le MKP — le pooling de GNN
    en dépend, cf. GNN.forward).

    Features (6, données brutes — pas de DP/solve) :
      [salary[t]/s_max, demand[i,t]/d_max, y_committed[t]/E,
       shortfall[i,t]/E, e/E, t/T]

    Arêtes : item_varies (clique par scénario sur (e,t), poids 1/G) ;
             scenario_varies (étoile ancrée au scénario libre 0, poids 1).

    Retourne None si G == 0 ou N_z == 0.
    """
    E, T, N = inst["E"], inst["T"], inst["N"]
    salary  = inst["salary"]   # (T,) float64
    demand  = inst["demand"]   # (N, T) int32

    x_mask = node_rec["x_mask"]   # (E*T,) int8 — idx = e*T+t (à vérifier empiriquement)
    z_mask = node_rec["z_mask"]   # (N,)   int8

    free_et    = np.where(x_mask == SSP_SIGMA_SIZE)[0]   # (e,t) libres, idx global e*T+t
    free_z     = np.where(z_mask == 2)[0]
    n_fixed_z1 = int((z_mask == 1).sum())

    G   = len(free_et)
    N_z = len(free_z)
    if G == 0 or N_z == 0:
        return None

    e_idx = free_et // T
    t_idx = free_et % T

    s_max = max(float(salary.max()), 1e-9)
    d_max = max(float(demand.max()), 1.0)

    # y_committed[t] : nb d'employés (TOUS, y compris fixés) déjà fixés à W à t.
    x_mask_2d   = x_mask.reshape(E, T)
    y_committed = (x_mask_2d == SSP_SIGMA_W).sum(axis=0).astype(np.float32)  # (T,)

    demand_free = demand[free_z, :].astype(np.float32)   # (N_z, T)

    curr_cost = node_rec["curr_cost"]
    # c_scale : borne grossière du surcoût encore possible (analogue c_g.sum() du MKP) —
    # somme des salaires des périodes encore libres, sur-approximation car suppose
    # que tout employé libre pourrait travailler à chaque période libre.
    c_scale = max(float(curr_cost) + float(salary[t_idx].sum()), 1.0)

    salary_g      = (salary[t_idx].astype(np.float32) / s_max)            # (G,)
    e_norm_g      = (e_idx / max(E - 1, 1)).astype(np.float32)             # (G,)
    t_norm_g      = (t_idx / max(T - 1, 1)).astype(np.float32)             # (G,)
    y_committed_g = y_committed[t_idx] / max(E, 1)                         # (G,)

    d_ig         = demand_free[:, t_idx] / d_max                                          # (N_z, G)
    shortfall_ig = np.maximum(0.0, demand_free[:, t_idx] - y_committed[None, t_idx]) / max(E, 1)  # (N_z, G)

    f_salary  = np.broadcast_to(salary_g[None, :],      (N_z, G))
    f_ycommit = np.broadcast_to(y_committed_g[None, :], (N_z, G))
    f_e       = np.broadcast_to(e_norm_g[None, :],      (N_z, G))
    f_t       = np.broadcast_to(t_norm_g[None, :],      (N_z, G))

    X = np.stack(
        [f_salary, d_ig, f_ycommit, shortfall_ig, f_e, f_t], axis=-1
    ).reshape(N_z * G, 6).astype(np.float32)                              # (N_z*G, 6)

    # ── Arêtes (vectorisées — identique au MKP, n_g remplacé par G) ──────────
    i_all = np.arange(N_z, dtype=np.int64)
    k_all = np.arange(G, dtype=np.int64)

    # 1. (e,t) varie (même scénario i_idx) : clique, poids 1/G
    ki, kj   = np.triu_indices(G, k=1)
    i_rep    = np.repeat(i_all, len(ki))
    ki_rep   = np.tile(ki, N_z);  kj_rep = np.tile(kj, N_z)
    src1_a   = i_rep * G + ki_rep;  src1_b = i_rep * G + kj_rep
    src1     = np.concatenate([src1_a, src1_b])
    dst1     = np.concatenate([src1_b, src1_a])
    w1       = np.full(len(src1), 1.0 / G, dtype=np.float32)

    # 2. Scénario varie (même (e,t) k_idx) : ÉTOILE ancrée au scénario libre 0
    #    (même schéma que sd_edit MKP), poids 1, bidirectionnelle.
    ii       = np.arange(1, N_z, dtype=np.int64)
    k_rep    = np.repeat(k_all, len(ii))
    ii_rep   = np.tile(ii, G)
    src2_a   = 0 * G + k_rep;   src2_b = ii_rep * G + k_rep
    src2     = np.concatenate([src2_a, src2_b])
    dst2     = np.concatenate([src2_b, src2_a])
    w2       = np.full(len(src2), 1.0, dtype=np.float32)

    edge_index  = torch.tensor(
        np.stack([np.concatenate([src1, src2]), np.concatenate([dst1, dst2])]),
        dtype=torch.long)
    edge_weight = torch.tensor(np.concatenate([w1, w2]), dtype=torch.float32)

    # ── Contexte (features au niveau du graphe, même schéma que MKP) ─────────
    depth_norm = float(node_rec["depth"]) / max(E * T, 1)
    frac_fixed = 1.0 - G / max(E * T, 1)
    # "tightest" : slack moyen restant (1 - besoin moyen normalisé) du scénario
    # libre le plus proche de la saturation — analogue de min(cap_frac) du MKP
    # (plus petit = plus contraint), mais dérivé de shortfall (brut) plutôt que
    # d'une capacité résiduelle explicite.
    scen_slack     = 1.0 - shortfall_ig.mean(axis=1)
    tightest       = float(scen_slack.min())
    curr_cost_norm = float(curr_cost) / c_scale

    context = torch.tensor(
        [depth_norm, frac_fixed, tightest, curr_cost_norm,
         float(inst["epsilon"]), n_fixed_z1 / N,
         float(np.log(max(G, 1))), float(np.log(max(N_z, 1))), float(N_z) / float(G)],
        dtype=torch.float32)

    return pyg_data.Data(
        x             = torch.from_numpy(X),
        edge_index    = edge_index,
        edge_weight   = edge_weight,
        context       = context,
        opt           = torch.tensor(float(inst["opt"]), dtype=torch.float32),
        curr_cost_raw = torch.tensor(float(curr_cost), dtype=torch.float32),
        c_scale       = torch.tensor(c_scale, dtype=torch.float32),
        n_g           = G,          # nom conservé (= MKP) pour réutiliser le pooling tel quel
        N             = N,
        N_z           = N_z,
        E             = E,
        T             = T,
        n_fixed_z1    = n_fixed_z1,
        epsilon       = torch.tensor(float(inst["epsilon"]), dtype=torch.float32),
        # Nécessaire au calcul de L1(λ) (lagrangian_v1ld_ssp) — indices/valeurs brutes.
        free_et     = torch.from_numpy(free_et.copy()),                 # (G,) idx global e*T+t
        free_z      = torch.from_numpy(free_z.copy()),                  # (N_z,) idx global scénario
        x_mask_full = torch.from_numpy(x_mask.astype(np.int64).copy()),  # (E*T,) 0..3 fixé, 4=libre
        salary_full = torch.from_numpy(salary.astype(np.float32).copy()),  # (T,)
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

    Un seul format côté SSP (Format C, bab_ssp_gecode_base_collect,
    3+E*T+N champs — pas d'analogue "Format A/B" avec borne Lagrangienne
    puisque la collecte tourne en CP pur, cf. docstring du module) :
      inst_id, depth, curr_cost, x_mask[E*T]{0..3 fixé, 4=libre}, z_mask[N]{0/1 fixé, 2=libre}
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
            E, T, N = inst["E"], inst["T"], inst["N"]

            n_fmt_C = 3 + E * T + N
            if len(vals) != n_fmt_C:
                skipped += 1
                continue

            x_mask = np.array(vals[3 : 3 + E * T], dtype=np.int8)
            z_mask = np.array(vals[3 + E * T : 3 + E * T + N], dtype=np.int8)
            node_rec = {
                "depth":     depth,
                "curr_cost": curr_cost,
                "x_mask":    x_mask,
                "z_mask":    z_mask,
            }

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
    Parse instances + nœuds sans construire les graphes PyG (Format C
    uniquement, 3+E*T+N champs — cf. load_dataset).
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
            E, T, N = inst["E"], inst["T"], inst["N"]

            n_fmt_C = 3 + E * T + N
            if len(vals) != n_fmt_C:
                skipped += 1
                continue

            x_mask = np.array(vals[3 : 3 + E * T], dtype=np.int8)
            z_mask = np.array(vals[3 + E * T : 3 + E * T + N], dtype=np.int8)

            if (x_mask == SSP_SIGMA_SIZE).sum() == 0 or (z_mask == 2).sum() == 0:
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

        node_rec = {
            "depth":     rec["depth"],
            "curr_cost": rec["curr_cost"],
            "x_mask":    rec["x_mask"],
            "z_mask":    rec["z_mask"],
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
    Prédit λ[i,(e,t)] ∈ R^4 (une composante par symbole σ∈{O,P,R,W}) par
    nœud (scénario libre i, paire libre (e,t)) — portage SSP de la classe
    GNN du MKP (gnn_sd_edit/train_v1LD_sd_edit.py) : seuls NODE_DIM (6 fixe,
    au lieu de 7m+5 — pas d'axe "contrainte" côté SSP) et la largeur de sortie
    de la tête (4 au lieu de 1) changent ; le reste (embed, convs, pooling
    item/scénario/graphe) est identique.

    Architecture (6 node features brutes, context 9-dim, max_pool optionnel) :
      embed      : X(6) → hidden → ReLU → hidden → ReLU → LayerNorm
      conv1      : GatedGraphConv(hidden, 2 it.) + LayerNorm
      conv2      : GatedGraphConv(hidden, 2 it.) + LayerNorm + skip(h1)
      proj       : hidden → 4h → ReLU → 2h                    → h_v
      pool item  : nœud → item k [mean] (+ [max] si use_max_pool) → h_item_k
      pool scén  : nœud → scénario i [mean] (+ [max] si use_max_pool) → h_scen_i
      pool graph : scénario → graphe [mean] → g_pool (2h)
      norm_ctx   : LayerNorm sur les 9 features de contexte (calculé une seule fois)
      head       : [h_v | h_item_k | h_scen_i | g_pool | ctx] → h → ReLU → h → ReLU → 4

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
      hidden              : dimension des couches cachées (défaut 64)
      use_max_pool (bool) : si True, pool item/scén = [mean | max]
    """
    CONTEXT_DIM = 9  # [depth, frac_fixed, tightest, curr_cost, epsilon, n_fixed_z1/N, log(n_g), log(N_z), N_z/n_g]
    NODE_DIM    = 6  # [salary/s_max, demand/d_max, y_committed/E, shortfall/E, e/E, t/T]
    N_SIGMA     = 4  # |Σ| = {O,P,R,W} — dimension de sortie (λ par symbole)

    def __init__(self, hidden: int = 64, use_max_pool: bool = False):
        super().__init__()
        self.hidden       = hidden
        self.use_max_pool = use_max_pool
        node_dim    = self.NODE_DIM

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
            nn.Linear(hidden, self.N_SIGMA),
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
        h_out   = self.head(cat)                                   # (V, 4) — un λ par symbole σ

        lam = h_out * data.c_scale / float(data.N)                  # broadcast (V,4) * scalaire
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
# Borne L1(λ) — implémentation différentiable de V1LD (SSP)
# ─────────────────────────────────────────────────────────────────────────────

def _base_cost_full(salary_full: torch.Tensor, E: int, T: int, dev) -> torch.Tensor:
    """c[e,t,σ] : -salary[t] si σ=W, 0 sinon — partagé par tous les employés."""
    c = torch.zeros(E * T * SSP_SIGMA_SIZE, dtype=torch.float32, device=dev)
    salary_rep = salary_full.to(dev).repeat(E)                              # (E*T,)
    w_idx = torch.arange(E * T, device=dev) * SSP_SIGMA_SIZE + SSP_SIGMA_W
    c[w_idx] = -salary_rep
    return c


def lagrangian_v1ld_ssp(lam: torch.Tensor, data: pyg_data.Data, inst: dict) -> torch.Tensor:
    """
    Calcule L1(λ) sur les scénarios libres, différentiable en λ. Portage SSP
    de lagrangian_v1ld (MKP) — même schéma (enveloppe : θ rapide+différentiable
    via l'argmax figé, ζ exact détaché, Δ et y* pour la correction), mais :

      - le "cost" est le vecteur COMPLET (E*T*4,), pas juste les items actifs
        (le masquage SSP ne rétrécit pas l'espace de décision, cf. discussion
        et ssp_problem_bnb.hpp) — λ/Λ_sum ne corrigent QUE les positions
        libres (free_et×σ), le reste garde le coût de base (pas de λ là où
        il n'y a plus de décision à arbitrer, comme les items fixés du MKP
        n'apparaissent pas dans Λ_sum).
      - θ_i et ζ_i viennent de SSPProblemBnB::solve_theta/solve_zeta (DP par
        employé masqué / CP-SAT masqué), pas d'un calcul vectorisé numpy —
        appelés en boucle Python sur N_z scénarios (pas de batching C++ ici,
        contrairement à _cpp_solve_zeta_batch du MKP ; à ajouter plus tard
        si ça devient le goulot, cf. discussion sur le coût de ζ CP-SAT).
      - PAS d'ajout séparé de curr_cost : le masque impose déjà les symboles
        fixés dans la DP, donc leur coût de base est intrinsèquement inclus
        dans la valeur retournée par solve_theta/solve_zeta (contrairement au
        MKP, où le rétrécissement exclut les items fixés de theta/zeta et
        curr_cost doit être ajouté séparément pour comparer à `opt`).

    inst : dict instance (transitions/accepting/salary/demand/E) — nécessaire
           pour reconstruire SSPProblemBnB (pas stocké sur `data`, cf. discussion :
           objet ctypes non batchable proprement par PyG).
    """
    N_z        = data.N_z
    G          = data.n_g
    N          = data.N
    E, T       = int(data.E), int(data.T)
    n_fixed_z1 = data.n_fixed_z1
    epsilon    = data.epsilon.item()
    dev        = lam.device

    lam_3d = lam.view(N_z, G, SSP_SIGMA_SIZE)                     # (N_z, G, 4)
    p_unif = 1.0 / N

    free_et = data.free_et.to(dev)                                # (G,) idx global e*T+t
    free_z_np = data.free_z.cpu().numpy()

    c_full = _base_cost_full(data.salary_full, E, T, dev)         # (n_vars,)

    # (A) Λ_sum = Σ_i λ_i  (sur N_z scénarios libres, par (e,t,σ) libre)
    Lambda_sum = lam_3d.sum(dim=0)                                # (G, 4)

    # (B) cost_full[i] = c_full partout, corrigé de λ_i/p − Λ_sum sur (free_et, ·)
    lam_correction = (lam_3d / p_unif - Lambda_sum.unsqueeze(0))  # (N_z, G, 4)
    flat_free_idx = (
        free_et[:, None] * SSP_SIGMA_SIZE
        + torch.arange(SSP_SIGMA_SIZE, device=dev)[None, :]
    ).reshape(-1)                                                 # (G*4,)

    cost_full = c_full.unsqueeze(0).repeat(N_z, 1)                # (N_z, n_vars)
    cost_full = cost_full.index_add(1, flat_free_idx, lam_correction.reshape(N_z, -1))

    # ── Sous-problèmes (hors graphe de calcul) — un solveur par nœud, réutilisé
    #    pour les N_z appels theta/zeta de CE nœud ────────────────────────────
    prob = get_ssp_problem(inst, free_z_np, data.x_mask_full.cpu().numpy())
    cost_np = cost_full.detach().cpu().numpy().astype(np.float64)   # (N_z, n_vars)

    theta_np   = np.zeros(N_z)
    zeta_np    = np.zeros(N_z)
    x_theta_np = np.zeros((N_z, prob.n_vars), dtype=np.float64)
    x_zeta_np  = np.zeros((N_z, prob.n_vars), dtype=np.float64)
    for i_local in range(N_z):
        v_t, x_t = prob.solve_theta(cost_np[i_local])
        v_z, x_z = prob.solve_zeta(i_local, cost_np[i_local])
        theta_np[i_local], x_theta_np[i_local] = v_t, x_t
        zeta_np[i_local],  x_zeta_np[i_local]  = v_z, x_z

    Delta_np = np.maximum(0.0, theta_np - zeta_np)

    # (C) y* greedy — identique au MKP
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

    # (D) L1 linéaire en λ à x*, y* fixés (théorème de l'enveloppe)
    x_theta_t = torch.tensor(x_theta_np, dtype=torch.float32, device=dev)
    x_zeta_t  = torch.tensor(x_zeta_np,  dtype=torch.float32, device=dev)
    y_star_t  = torch.tensor(y_star,     device=dev)

    theta_lam = (cost_full * x_theta_t).sum(dim=1)
    delta_lam = (cost_full * (x_theta_t - x_zeta_t)).sum(dim=1)

    L1_theta = (p_unif * theta_lam).sum()
    val_lam  = (p_unif * y_star_t * delta_lam).sum()
    return L1_theta - val_lam


# ─────────────────────────────────────────────────────────────────────────────
# Vérification par sous-gradient Python pur — port de v1ld_solver.cpp
# ─────────────────────────────────────────────────────────────────────────────

def verify_v1ld_ssp(
    data: pyg_data.Data,
    inst: dict,
    T: int = 200,
    v_lower: float | None = None,
    precision: float | None = None,
    patience:  int | None   = None,
    lam_init:  np.ndarray | None = None,
    return_n_iters: bool = False,
    return_lam: bool = False,
    history_out: list | None = None,
) -> float | tuple:
    """
    Port SSP de verify_v1ld (MKP) — reproduit l'algorithme de sous-gradient
    v1ld_solver.cpp en Python pur, via SSPProblemBnB (DP/CP-SAT masqués) au
    lieu du calcul numpy vectorisé du MKP : θ n'est plus une simple somme de
    coûts positifs, donc chaque itération appelle solve_theta/solve_zeta N_z
    fois (coûteux — cf. discussion sur le prix de ζ CP-SAT ; ce vérificateur
    n'est praticable que sur de très petites instances).

    v_lower : borne inférieure connue sur v1LD* (ex. −opt via Gecode/B&B).
        Si fournie, active le pas de Polyak (gap/‖g‖²) — cf. v1ld_solver.cpp.
        Sinon (None), retombe sur le pas 1/√(t+1), documenté comme dangereux
        pour N grand dans bench_ssp.cpp (amplification ×N via λ_i/p_i).
    lam_init : (N_z, G, 4) float64 ; None = cold start λ=0.
    """
    N_z        = data.N_z
    G          = data.n_g
    N          = data.N
    E, T_hor   = int(data.E), int(data.T)
    n_fixed_z1 = data.n_fixed_z1
    epsilon    = data.epsilon.item()

    free_et_np = data.free_et.numpy()
    free_z_np  = data.free_z.numpy()

    c_full_np = _base_cost_full(data.salary_full, E, T_hor, "cpu").numpy().astype(np.float64)

    # Ordre (g, s) pour matcher lam[i].reshape(-1) — même convention que
    # flat_free_idx dans lagrangian_v1ld_ssp.
    flat_free_idx = (free_et_np[:, None] * SSP_SIGMA_SIZE
                     + np.arange(SSP_SIGMA_SIZE)[None, :]).reshape(-1)     # (G*4,)

    prob = get_ssp_problem(inst, free_z_np, data.x_mask_full.numpy())

    p_unif        = 1.0 / N
    required_prob = max(0.0, (1.0 - epsilon) - n_fixed_z1 * p_unif)

    lam = (lam_init.copy() if lam_init is not None
           else np.zeros((N_z, G, SSP_SIGMA_SIZE), dtype=np.float64))

    best       = np.inf
    best_lam   = lam.copy()
    no_improve = 0
    n_iters    = T

    for t in range(T):
        Lsum           = lam.sum(axis=0)                              # (G,4)
        lam_correction = lam / p_unif - Lsum[None, :, :]              # (N_z, G, 4)

        cost_full = np.broadcast_to(c_full_np, (N_z, c_full_np.shape[0])).copy()
        cost_full[:, flat_free_idx] += lam_correction.reshape(N_z, -1)

        theta_vals   = np.zeros(N_z)
        zeta_vals    = np.zeros(N_z)
        x_theta_full = np.zeros((N_z, prob.n_vars))
        x_zeta_full  = np.zeros((N_z, prob.n_vars))
        for i in range(N_z):
            v_t, x_t = prob.solve_theta(cost_full[i])
            v_z, x_z = prob.solve_zeta(i, cost_full[i])
            theta_vals[i], x_theta_full[i] = v_t, x_t
            zeta_vals[i],  x_zeta_full[i]  = v_z, x_z

        Delta = np.maximum(0.0, theta_vals - zeta_vals)

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

        # Restriction aux positions libres (G,4) pour la mise à jour du sous-gradient
        x_theta_g = x_theta_full[:, flat_free_idx].reshape(N_z, G, SSP_SIGMA_SIZE)
        x_zeta_g  = x_zeta_full[:,  flat_free_idx].reshape(N_z, G, SSP_SIGMA_SIZE)

        x_bar = np.zeros((G, SSP_SIGMA_SIZE))
        for i in range(N_z):
            xi = x_zeta_g[i] if y_star[i] > 0.5 else x_theta_g[i]
            x_bar += p_unif * xi

        # ── Norme du sous-gradient ‖g‖² = Σ_i ‖x_i* − x̄‖² ──────────────────
        gnorm_sq = 0.0
        for i in range(N_z):
            xi = x_zeta_g[i] if y_star[i] > 0.5 else x_theta_g[i]
            gnorm_sq += float(((xi - x_bar) ** 2).sum())

        # Pas de Polyak si v_lower est fourni, sinon 1/√(t+1) — cf. v1ld_solver.cpp
        if v_lower is not None and v_lower > -1e17 and gnorm_sq > 1e-14:
            gap  = L1 - v_lower
            step = (gap / gnorm_sq) if gap > 0.0 else 1.0 / np.sqrt(t + 1)
        else:
            step = 1.0 / np.sqrt(t + 1)

        for i in range(N_z):
            xi = x_zeta_g[i] if y_star[i] > 0.5 else x_theta_g[i]
            lam[i] += step * (x_bar - xi)

    if return_n_iters and return_lam:
        return best, n_iters, best_lam
    if return_lam:
        return best, best_lam
    if return_n_iters:
        return best, n_iters
    return best


def _apply_center_ssp(lam: torch.Tensor, N_z: int, G: int) -> torch.Tensor:
    """
    λ_i,(e,t,σ) ← λ_i,(e,t,σ) − (1/N_z) Σ_i λ_i,(e,t,σ) (centrage sur les
    scénarios, différentiable). Port direct de _apply_center (MKP,
    gnn_sd_edit/train_scaled_sd_edit.py:671) — élimine la dérive de jauge
    de λ le long de l'axe scénario (constatée empiriquement : sans ce
    centrage, la loss diverge sur plusieurs centaines d'epochs même sur un
    petit graphe réel, cf. discussion — symptôme identique à celui qui a
    motivé --center_lam côté MKP).
    """
    lam_3d = lam.view(N_z, G, SSP_SIGMA_SIZE)
    return (lam_3d - lam_3d.mean(dim=0, keepdim=True)).reshape(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Overfit sur un graphe — validation de l'architecture
# ─────────────────────────────────────────────────────────────────────────────

def overfit_one_graph(data: pyg_data.Data, inst: dict, n_epochs: int = 1000, lr: float = 1e-3,
                       center_lam: bool = True):
    """
    Entraîne le GNN sur un seul graphe avec la vraie borne V1LD (SSP) comme loss.

    IMPORTANT (sign convention) : lagrangian_v1ld_ssp retourne L1(λ) dans la
    convention interne max(−coût) (c[e,t,W] = −salary[t], cf. ssp_problem.hpp)
    — L1(λ) ≤ 0 toujours. La borne réellement comparable à `opt` (coût salarial,
    positif) est cost_lower = −L1(λ), exactement comme bench_ssp.cpp:266-268
    ("Le solveur est max-natif... donc v1LD_star ≤ 0. La borne sur le coût min
    original est −v1LD_star ≥ 0."). Affichée ici directement (pas L1 brut) pour
    éviter l'erreur de comparaison faite plus tôt dans la session.
    """
    device = torch.device("cpu")
    data   = data.to(device)

    model     = GNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)

    opt_val = data.opt.item()
    with torch.no_grad():
        lam0    = torch.zeros(data.N_z * data.n_g * SSP_SIGMA_SIZE)
        cl_init = -lagrangian_v1ld_ssp(lam0, data, inst).item()   # cost_lower(λ=0)

    print(f"\n{'='*60}")
    print(f"Overfit SSP — G={data.n_g}  N_z={data.N_z}  N={data.N}  E={data.E}  T={data.T}")
    print(f"  opt={opt_val:.2f}   cost_lower(λ=0)={cl_init:.2f}")
    print(f"  graphe : {data.x.shape[0]} nœuds, {data.edge_index.shape[1]} arêtes")
    print(f"{'='*60}")
    print(f"{'epoch':>6}  {'cost_lower':>12}  {'gap_vs_opt%':>12}  {'gain_vs_l0%':>13}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        lam  = model(data)
        if center_lam:
            lam = _apply_center_ssp(lam, int(data.N_z), int(data.n_g))
        loss = lagrangian_v1ld_ssp(lam, data, inst)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 50 == 0 or epoch == 1:
            cl    = -loss.item()                                          # cost_lower(λ_GNN)
            g_opt = (cl - opt_val) / opt_val * 100 if opt_val > 0 else float("nan")
            g_l0  = (cl - cl_init) / abs(cl_init) * 100 if cl_init != 0 else float("nan")
            print(f"{epoch:6d}  {cl:12.4f}  {g_opt:+11.2f}%  {g_l0:+12.2f}%")

    cl_final = -loss.item()
    print(f"{'='*60}")
    print(f"Borne finale (cost_lower) : {cl_final:.4f}  (opt={opt_val:.2f}, cost_lower(λ=0)={cl_init:.2f})")


# ─────────────────────────────────────────────────────────────────────────────
# Test rapide
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/train_ssp_base_s10")
    tag      = sys.argv[2] if len(sys.argv) > 2 else "E3_T12_N20"
    mode     = sys.argv[3] if len(sys.argv) > 3 else "inspect"

    inst_file  = data_dir / f"instances_{tag}.txt"
    nodes_file = data_dir / f"nodes_{tag}.txt"

    instances = load_instances(inst_file)
    graphs    = load_dataset(inst_file, nodes_file, max_graphs=5)

    for i, g in enumerate(graphs):
        print(
            f"  graph {i:3d} | nodes={g.x.shape[0]:6d} | "
            f"edges={g.edge_index.shape[1]:8d} | "
            f"G={g.n_g} N_z={g.N_z}/{g.N} E={g.E} T={g.T} | "
            f"depth={int(g.context[0].item())} | opt={g.opt.item():.2f}"
        )

    if mode in ("verify", "overfit") and graphs:
        g0   = graphs[0]
        inst0 = instances[int(g0.inst_id.item())]
        lam0    = torch.zeros(g0.N_z * g0.n_g * SSP_SIGMA_SIZE)
        cl_init = -lagrangian_v1ld_ssp(lam0, g0, inst0).item()   # cost_lower(λ=0), cf. bench_ssp.cpp:266-268
        print(f"\n── Vérification V1LD SSP (T=20 itérations subgradient, pas Polyak) ──")
        print(f"  cost_lower(λ=0)  : {cl_init:.4f}")
        py_best    = verify_v1ld_ssp(g0, inst0, T=20, v_lower=-g0.opt.item())
        cl_subgrad = -py_best
        print(f"  subgradient T=20  : {cl_subgrad:.4f}")
        print(f"  opt               : {g0.opt.item():.4f}")
        print(f"  amélioration      : {(cl_subgrad - cl_init) / abs(cl_init) * 100:.1f}%")

    if mode == "overfit" and graphs:
        g0    = graphs[0]
        inst0 = instances[int(g0.inst_id.item())]
        overfit_one_graph(g0, inst0)
