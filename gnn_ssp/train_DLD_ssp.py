"""
train_DLD_ssp.py — Portage DLD, dupliqué depuis train_SLD_ssp.py (SLD).

Différence mathématique clé (vérifiée contre solver_generic/dld_solver.cpp,
la seule source faisant foi) : DLD ne dualise pas un scalaire par scénario
mais une copie COMPLÈTE du planning (E*T*4 variables) pour CHAQUE paire
(scénario i, période j) — Λ_sum se somme sur (i,j) ensemble, pas seulement sur
i. λ_dld a donc la forme (N, T, E*T*4), un facteur T plus gros que λ_sld
(N, E*T*4) — cf. gecode_bab_ssp.hpp:249.

Conséquence sur le graphe : la grille scénario×(e,t) de SLD devient
(scénario libre × période j) × (e,t) libre — on aplatit (i,j) en un seul axe
"scénario-like" de taille N_z*T (même ordre i-majeur/j-mineur que MKP,
group = i_local*T + j), et on RÉUTILISE tel quel le pooling/les arêtes de
SLD (qui ne dépendent que d'un nombre de groupes et d'une taille de groupe
G, jamais de leur sens) — seul le nombre de groupes change (N_z*T au lieu de
N_z). Deux features de nœud ajoutées par rapport à SLD : `j_norm` (période
dont la couverture est dualisée dans CE groupe) et `same_period` (1 si le
(e,t) de ce nœud correspond à j, 0 sinon) — sans ça, rien ne distingue les
N_z*T groupes entre eux au niveau des features (salaire/demande à l'instant
t restent identiques quel que soit j).

ζ_{i,j} est un GLOUTON O(E·T·Q) (SSPDLDProblemBnB::solve_zeta, pas CP-SAT) —
beaucoup moins cher par appel que SLD, mais il y a N_z*T appels par
évaluation de la borne au lieu de N_z (theta ET zeta à chaque (i,j)).

⚠️ La formule d'agrégation Δ_i = (1/T)Σ_j δ_{i,j} (utilisée ici, fidèle à
dld_solver.cpp:146-150) a une validité mathématique comme majorant de
θ_i − ζ_i qui n'est PAS démontrée ici — piste ouverte à l'origine de
l'investigation de juillet 2026 (instance de debug E=4,T=8,N=20, historique
supprimé lors du nettoyage du dépôt) ; cf. PREUVE_DLD_centrage_par_scenario.md
(racine du dépôt) pour la preuve établie depuis pour la variante centrée par
scénario. Les bornes produites ici sont donc exploratoires (utiles pour comparer
GNN vs sous-gradient sur la MÊME loss), pas certifiées valides.

Format des fichiers d'entrée : identique à train_SLD_ssp.py (mêmes
instances_E{E}_T{T}_N{N}.txt / nodes_E{E}_T{T}_N{N}.txt).

Structure du graphe produit (un graphe par nœud B&B) :
  - Scénarios libres  : free_z = {i : z_mask[i] == 2} ; N_z = |free_z|
  - Paires (e,t) libres : free_et = {(e,t) : x_mask[e*T+t] == 4} ; G = |free_et|
  - Groupes : N_z * T (un par (scénario libre, période j∈0..T-1))
  - Nœuds   : (N_z*T) * G
  - Features (8) : [salary[t]/s_max, demand[i,t]/d_max, y_committed[t]/E,
                    shortfall[i,t]/E, e/E, t/T, j/T, same_period(t==j)]
  - Arêtes  : identiques à SLD (clique intra-groupe poids 1/G,
              étoile inter-groupes ancrée au groupe 0, poids 1)
  - Sortie  : λ[i,j,(e,t)] ∈ R^4
  - Target  : minimiser L1(λ) — borne DLD (cf. dld_solver.cpp)
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

# ── Solveur ζ/θ C++ (SSPDLDProblemBnB, glouton — cf. bnb_solver_ssp.py) ──────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bnb_solver_ssp import SSPDLDProblemPy  # noqa: E402

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


def get_ssp_dld_problem(inst: dict, free_z: np.ndarray, x_mask: np.ndarray) -> SSPDLDProblemPy:
    """Miroir de get_ssp_problem (SLD), ciblant SSPDLDProblemPy (glouton, pas CP-SAT)."""
    E, T = inst["E"], inst["T"]
    demand_free = inst["demand"][free_z, :]
    N_z = len(free_z)
    p_local = [1.0 / max(N_z, 1)] * N_z

    mask = x_mask.reshape(E, T).astype(np.int64)
    mask = np.where(mask == SSP_SIGMA_SIZE, -1, mask)

    return SSPDLDProblemPy(
        inst["transitions"], inst["accepting"], inst["initial"],
        inst["salary"], demand_free, p_local, inst["epsilon"], inst["E"],
        mask=mask,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Construction du graphe pour un nœud B&B
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(inst: dict, node_rec: dict) -> Optional[pyg_data.Data]:
    """
    Construit un graphe torch_geometric DLD à partir d'une instance SSP et
    d'un état de nœud B&B (même Format C que SLD, cf. collect_ssp_base.hpp).

    Groupes : N_z * T (un par (scénario libre i_local, période j)), ordonnés
    i_local*T + j (i-majeur, j-mineur). Chaque groupe contient les MÊMES G
    paires (e,t) libres — le pooling de GNN.forward (item/scénario) réutilise
    donc tel quel la même logique que SLD, avec "n_scénarios" = N_z*T au
    lieu de N_z.

    Features (8, données brutes) :
      [salary[t]/s_max, demand[i,t]/d_max, y_committed[t]/E, shortfall[i,t]/E,
       e/E, t/T, j/T, same_period=1{t==j}]
    Les deux premières features par (e,t) sont identiques à SLD (ne dépendent
    pas de j) ; j/T et same_period sont les seules à distinguer les N_z*T
    groupes entre eux.

    (Testé et écarté 2026-07-14 : ajout de y_committed[j]/demand[i,j]/d_eff[i,j]
    en broadcast groupe — aucun gain mesuré en overfit A/B seedé, cf. le
    graphe est déjà complètement connecté intra-groupe [item_varies], donc
    cette info est reconstructible en 1 hop sans l'expliciter.)

    Retourne None si G == 0 ou N_z == 0.
    """
    E, T, N = inst["E"], inst["T"], inst["N"]
    salary  = inst["salary"]   # (T,) float64
    demand  = inst["demand"]   # (N, T) int32

    x_mask = node_rec["x_mask"]   # (E*T,) int8 — idx = e*T+t
    z_mask = node_rec["z_mask"]   # (N,)   int8

    free_et    = np.where(x_mask == SSP_SIGMA_SIZE)[0]
    free_z     = np.where(z_mask == 2)[0]
    n_fixed_z1 = int((z_mask == 1).sum())

    G   = len(free_et)
    N_z = len(free_z)
    if G == 0 or N_z == 0:
        return None

    n_groups = N_z * T

    e_idx = free_et // T
    t_idx = free_et % T

    s_max = max(float(salary.max()), 1e-9)
    d_max = max(float(demand.max()), 1.0)

    x_mask_2d   = x_mask.reshape(E, T)
    y_committed = (x_mask_2d == SSP_SIGMA_W).sum(axis=0).astype(np.float32)  # (T,)

    demand_free = demand[free_z, :].astype(np.float32)   # (N_z, T)

    curr_cost = node_rec["curr_cost"]
    c_scale = max(float(curr_cost) + float(salary[t_idx].sum()), 1.0)

    salary_g      = (salary[t_idx].astype(np.float32) / s_max)            # (G,)
    e_norm_g      = (e_idx / max(E - 1, 1)).astype(np.float32)             # (G,)
    t_norm_g      = (t_idx / max(T - 1, 1)).astype(np.float32)             # (G,)
    y_committed_g = y_committed[t_idx] / max(E, 1)                         # (G,)

    d_ig         = demand_free[:, t_idx] / d_max                                          # (N_z, G)
    shortfall_ig = np.maximum(0.0, demand_free[:, t_idx] - y_committed[None, t_idx]) / max(E, 1)  # (N_z, G)

    # Expansion sur l'axe période j (T) — seules j_norm/same_period en dépendent.
    j_norm_j     = (np.arange(T) / max(T - 1, 1)).astype(np.float32)             # (T,)
    same_period  = (t_idx[None, :] == np.arange(T)[:, None]).astype(np.float32)  # (T, G)

    f_salary    = np.broadcast_to(salary_g[None, None, :],      (N_z, T, G))
    f_d         = np.broadcast_to(d_ig[:, None, :],              (N_z, T, G))
    f_ycommit   = np.broadcast_to(y_committed_g[None, None, :], (N_z, T, G))
    f_shortfall = np.broadcast_to(shortfall_ig[:, None, :],      (N_z, T, G))
    f_e         = np.broadcast_to(e_norm_g[None, None, :],      (N_z, T, G))
    f_t         = np.broadcast_to(t_norm_g[None, None, :],      (N_z, T, G))
    f_j         = np.broadcast_to(j_norm_j[None, :, None],      (N_z, T, G))
    f_sameper   = np.broadcast_to(same_period[None, :, :],      (N_z, T, G))

    X = np.stack(
        [f_salary, f_d, f_ycommit, f_shortfall, f_e, f_t, f_j, f_sameper], axis=-1
    ).reshape(n_groups * G, 8).astype(np.float32)   # (N_z*T*G, 8), groupe = i_local*T+j

    # ── Arêtes (identiques à SLD, n_groups remplace N_z) ────────────────────
    i_all = np.arange(n_groups, dtype=np.int64)
    k_all = np.arange(G, dtype=np.int64)

    ki, kj   = np.triu_indices(G, k=1)
    i_rep    = np.repeat(i_all, len(ki))
    ki_rep   = np.tile(ki, n_groups);  kj_rep = np.tile(kj, n_groups)
    src1_a   = i_rep * G + ki_rep;  src1_b = i_rep * G + kj_rep
    src1     = np.concatenate([src1_a, src1_b])
    dst1     = np.concatenate([src1_b, src1_a])
    w1       = np.full(len(src1), 1.0 / G, dtype=np.float32)

    ii       = np.arange(1, n_groups, dtype=np.int64)
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

    # ── Contexte (features au niveau du graphe) ──────────────────────────────
    depth_norm = float(node_rec["depth"]) / max(E * T, 1)
    frac_fixed = 1.0 - G / max(E * T, 1)
    scen_slack     = 1.0 - shortfall_ig.mean(axis=1)
    tightest       = float(scen_slack.min())
    curr_cost_norm = float(curr_cost) / c_scale

    context = torch.tensor(
        [depth_norm, frac_fixed, tightest, curr_cost_norm,
         float(inst["epsilon"]), n_fixed_z1 / N,
         float(np.log(max(G, 1))), float(np.log(max(n_groups, 1))), float(n_groups) / float(G)],
        dtype=torch.float32)

    return pyg_data.Data(
        x             = torch.from_numpy(X),
        edge_index    = edge_index,
        edge_weight   = edge_weight,
        context       = context,
        opt           = torch.tensor(float(inst["opt"]), dtype=torch.float32),
        curr_cost_raw = torch.tensor(float(curr_cost), dtype=torch.float32),
        c_scale       = torch.tensor(c_scale, dtype=torch.float32),
        n_g           = G,          # nom conservé (= MKP/SLD) pour réutiliser le pooling tel quel
        N             = N,
        N_z           = N_z,
        E             = E,
        T             = T,
        n_fixed_z1    = n_fixed_z1,
        epsilon       = torch.tensor(float(inst["epsilon"]), dtype=torch.float32),
        free_et     = torch.from_numpy(free_et.copy()),
        free_z      = torch.from_numpy(free_z.copy()),
        x_mask_full = torch.from_numpy(x_mask.astype(np.int64).copy()),
        salary_full = torch.from_numpy(salary.astype(np.float32).copy()),
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
    GNN du MKP (gnn_sd_edit/train_SLD_sd_edit.py) : seuls NODE_DIM (6 fixe,
    au lieu de 7m+5 — pas d'axe "contrainte" côté SSP) et la largeur de sortie
    de la tête (4 au lieu de 1) changent ; le reste (embed, convs, pooling
    item/scénario/graphe) est identique.

    Architecture (8 node features brutes, context 9-dim, max_pool optionnel) :
      embed      : X(8) → hidden → ReLU → hidden → ReLU → LayerNorm
      conv1      : GatedGraphConv(hidden, 2 it.) + LayerNorm
      conv2      : GatedGraphConv(hidden, 2 it.) + LayerNorm + skip(h1)
      proj       : hidden → 4h → ReLU → 2h                    → h_v
      pool item     : nœud → item k        [mean] (+[max] si use_max_pool) → h_item_k
      pool scénario : nœud → scénario i    [mean] (+[max] si use_max_pool) → h_scen_i
      pool période  : nœud → période j     [mean] (+[max] si use_max_pool) → h_period_j
      pool graph    : nœud → graphe        [mean] → g_pool (2h)
      norm_ctx      : LayerNorm sur les 9 features de contexte (calculé une seule fois)
      head          : [h_v | h_item_k | h_scen_i | h_period_j | g_pool | ctx]
                      → h → ReLU → h → ReLU → 4

    `h_item_k` (ajouté 2026-07-07, `diag_scenario_variance.py`) = moyenne
    (+max) de `h_v` sur les N_z scénarios, à item k fixé — taille n_g,
    indépendante de N, analogue direct du consensus `x̄_k` de la mise à jour
    sous-gradient (`verify_sld` : `lam[i] += step*(x_bar - xi)`), jamais
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

    `h_scen_i` / `h_period_j` (2026-08-17, remplace l'ancien pooling "par
    groupe (i,j) flatten" qui n'était ni un vrai marginal scénario ni période
    -- mirror exact de h_item/h_scen/h_cons de gnn_sd_edit/
    train_DLD_sd_edit.py (MKP), "période" jouant le rôle de "contrainte".
    Validé par A/B test à 300 graphes/100 epochs (discussion 2026-08-17) :
    gain net et cohérent sur 5/6 combos (gap global -8.72%→-8.02%,
    pct_vs_sg 59.0%→62.3%) -- adopté comme architecture standard, plus de
    branche alternative dans le code.

    Paramètres :
      hidden              : dimension des couches cachées (défaut 64)
      use_max_pool (bool) : si True, pool item/scén/période = [mean | max]
    """
    CONTEXT_DIM = 9  # [depth, frac_fixed, tightest, curr_cost, epsilon, n_fixed_z1/N, log(n_g), log(n_groups), n_groups/n_g]
    NODE_DIM    = 8  # [salary/s_max, demand/d_max, y_committed/E, shortfall/E, e/E, t/T, j/T, same_period]
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
        # pool item / scén / période : 2h chacun (mean seul) ou 4h (mean+max)
        item_pool_dim   = 4 * hidden if use_max_pool else 2 * hidden
        scen_pool_dim   = 4 * hidden if use_max_pool else 2 * hidden
        period_pool_dim = 4 * hidden if use_max_pool else 2 * hidden

        # h_v(2h) + h_item_k(item_pool_dim) + h_scen_i(scen_pool_dim)
        # + h_period_j(period_pool_dim) + g_pool(2h) + ctx(9)
        head_in = (2 * hidden + item_pool_dim + scen_pool_dim + period_pool_dim
                   + 2 * hidden + self.CONTEXT_DIM)
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

        # ── Pooling item (h_item_k), scénario pur (h_scen_i), période pure
        # (h_period_j) ────────────────────────────────────────────────────────
        # Nœuds ordonnés i*(T*n_g) + j*n_g + k (scénario majeur, puis période,
        # puis item), donc à local_idx dans son graphe :
        #   k   = local_idx % n_g
        #   grp = local_idx // n_g  =  i*T + j
        #   j   = grp % T   (période, marginalise i et k)
        #   i   = grp // T  (scénario, marginalise j et k)
        # NB : suppose n_g ET T uniformes sur le batch (n_g varie par nœud B&B
        # mais est supposé constant au sein d'UN batch, même hypothèse que
        # SLD ; T est fixe par instance donc toujours uniforme en pratique).
        n_g = data.n_g if isinstance(data.n_g, int) else int(data.n_g.item())
        T   = data.T   if isinstance(data.T,   int) else int(data.T.item())

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

        grp = local_idx // n_g                                      # i*T + j, 0..N_z*T-1

        n_scen_per = counts // (n_g * T)                            # (B,) = N_z
        scen_off   = torch.zeros(n_graphs, dtype=torch.long, device=dev)
        scen_off[1:] = n_scen_per[:-1].cumsum(0)
        scen_batch = scen_off[batch] + (grp // T)                   # (V,) i pur

        h_scen_mean = global_mean_pool(h_v, scen_batch)              # (Σ N_z, 2h)
        if self.use_max_pool:
            h_scen_max = global_max_pool(h_v, scen_batch)         # (Σ N_z, 2h)
            h_scen = torch.cat([h_scen_mean, h_scen_max], dim=-1) # (Σ N_z, 4h)
        else:
            h_scen = h_scen_mean                                   # (Σ N_z, 2h)
        h_scen_exp = h_scen[scen_batch]                            # (V, 2h or 4h)

        period_off   = torch.arange(n_graphs, device=dev) * T
        period_batch = period_off[batch] + (grp % T)                # (V,) j pur

        h_period_mean = global_mean_pool(h_v, period_batch)           # (Σ T, 2h)
        if self.use_max_pool:
            h_period_max = global_max_pool(h_v, period_batch)
            h_period = torch.cat([h_period_mean, h_period_max], dim=-1)
        else:
            h_period = h_period_mean
        h_period_exp = h_period[period_batch]                      # (V, 2h or 4h)

        ctx_exp = ctx_norm[batch]                                          # (V, 9)

        g_pool  = global_mean_pool(h_v, batch)                        # (B, 2h)
        g_exp   = g_pool[batch]                                            # (V, 2h)

        cat     = torch.cat(
            [h_v, h_item_exp, h_scen_exp, h_period_exp, g_exp, ctx_exp], dim=-1)
        h_out   = self.head(cat)                                   # (V, 4) — un λ par symbole σ

        # FIX (2026-08-17) : héritage SLD, ne divisait que par N. La correction
        # DLD (Λ_sum_i/p_i, p_i=1/N ; lam/p_ij, p_ij=p_i/T) fait s'annuler le N
        # mais laisse un facteur T non compensé dans la perturbation de coût
        # effective (cf. review du pipeline, 2026-08-16-17) -- confirmé
        # empiriquement (diag_overfit_single_dld_ssp.py --t_compensate,
        # job 19155118 vs baseline T=12 : gnorm max ep1 40.3 -> 1.6, ÷25).
        lam = h_out * data.c_scale / (float(data.N) * float(data.T))  # broadcast (V,4) * scalaire
        if return_hidden:
            # Diagnostic uniquement (bottleneck de pooling) : h_v (par nœud),
            # h_item_mean (par item), h_scen_mean (par scénario), h_period_mean
            # (par période), avant concat max.
            extra = {"x": x, "h_embed": h, "h1": h1, "h2": h2,
                     "h_v": h_v, "h_item_mean": h_item_mean, "item_batch": item_batch,
                     "h_scen_mean": h_scen_mean, "scen_batch": scen_batch,
                     "h_period_mean": h_period_mean, "period_batch": period_batch,
                     "g_pool": g_pool, "h_out": h_out}
            return lam, extra
        return lam


# ─────────────────────────────────────────────────────────────────────────────
# Borne L1(λ) — implémentation différentiable de SLD (SSP)
# ─────────────────────────────────────────────────────────────────────────────

def _base_cost_full(salary_full: torch.Tensor, E: int, T: int, dev) -> torch.Tensor:
    """c[e,t,σ] : -salary[t] si σ=W, 0 sinon — partagé par tous les employés."""
    c = torch.zeros(E * T * SSP_SIGMA_SIZE, dtype=torch.float32, device=dev)
    salary_rep = salary_full.to(dev).repeat(E)                              # (E*T,)
    w_idx = torch.arange(E * T, device=dev) * SSP_SIGMA_SIZE + SSP_SIGMA_W
    c[w_idx] = -salary_rep
    return c


def lagrangian_dld_ssp(lam: torch.Tensor, data: pyg_data.Data, inst: dict) -> torch.Tensor:
    """
    Calcule L1(λ) DLD, différentiable en λ — port fidèle de la formule
    CORRIGÉE (2026-08-13) de solver_generic/dld_solver.cpp (la seule source
    faisant foi, cf. docstring du module) :

      Λ_sum_i     = Σ_j λ_{i,j}                           (PAR SCÉNARIO — pas sur tous les (i,j) ensemble)
      cost_{i,j}  = c + λ_{i,j}/p_{i,j} − Λ_sum_i/p_i,  p_{i,j} = p_i/T
      θ_{i,j}     = solve_theta(j, cost_{i,j})            (DP libre, différentiable via argmax figé)
      ζ_{i,j}     = solve_zeta(i, j, cost_{i,j})           (glouton, détaché)
      Δ_i         = (1/T) Σ_j max(0, θ_{i,j} − ζ_{i,j})   (le clamp est un no-op prouvé : ζ_{i,j}≤θ_{i,j} toujours, cf. preuve)
      y*, val     = glouton sur Δ (mêmes required_prob/n_fixed_z1 que SLD)
      L1          = Σ_{i,j} p_{i,j} θ_{i,j} − Σ_i p_i y*_i Δ_i(λ)

    Validité démontrée pour tout λ (centrage par scénario, condition
    nécessaire et suffisante ici) : cf. PREUVE_DLD_centrage_par_scenario.md
    (racine du dépôt) — preuve + vérification empirique sur 8 instances SSP
    (recherche B&B complète comparée à la référence exacte, dont l'instance
    de debug E=4,T=8,N=20 qui a motivé toute l'investigation en juillet 2026 ;
    historique supprimé lors du nettoyage du dépôt). Le fix miroir côté C++
    est déjà en production (solver_generic/dld_solver.cpp). Ce fichier
    Python N'ÉTAIT PAS encore corrigé avant ce commit — l'entraînement
    effectué avec l'ancienne formule (checkpoints existants) reste à
    ré-évaluer/réentraîner.

    inst : dict instance, nécessaire pour reconstruire SSPDLDProblemBnB (cf.
           get_ssp_dld_problem — mêmes raisons qu'en SLD, objet ctypes non
           stocké sur `data`).
    """
    N_z        = data.N_z
    G          = data.n_g
    N          = data.N
    E, T       = int(data.E), int(data.T)
    n_fixed_z1 = data.n_fixed_z1
    epsilon    = data.epsilon.item()
    dev        = lam.device

    n_groups = N_z * T
    lam_4d = lam.view(N_z, T, G, SSP_SIGMA_SIZE)                  # (N_z, T, G, 4)
    p_i  = 1.0 / N
    p_ij = p_i / T

    free_et   = data.free_et.to(dev)                              # (G,) idx global e*T+t
    free_z_np = data.free_z.cpu().numpy()

    c_full = _base_cost_full(data.salary_full, E, T, dev)         # (n_vars,)

    # (A) Λ_sum_i = Σ_j λ_{i,j}  — PAR SCÉNARIO (axe période seulement)
    # FIX (2026-08-13) : centrage global (sur les deux axes) remplacé par un
    # centrage par scénario -- cf. preuve PREUVE_DLD_centrage_par_scenario.md
    # et le fix miroir dans solver_generic/dld_solver.cpp. Le centrage
    # global n'annule le terme libre en X̄ que globalement, pas scénario par
    # scénario -- condition dont la dualité faible a besoin pour garantir
    # theta_i(0) <= mean_j theta_ij(λ) (et l'analogue pour zeta_i).
    Lambda_sum_i = lam_4d.sum(dim=1)                              # (N_z, G, 4)

    # (B) cost_{i,j} = c + λ_{i,j}/p_{i,j} − Λ_sum_i/p_i, appliqué aux positions libres
    lam_correction = lam_4d / p_ij - (Lambda_sum_i / p_i).unsqueeze(1)  # (N_z,T,G,4)
    flat_free_idx = (
        free_et[:, None] * SSP_SIGMA_SIZE
        + torch.arange(SSP_SIGMA_SIZE, device=dev)[None, :]
    ).reshape(-1)                                                  # (G*4,)

    cost_full = c_full.unsqueeze(0).repeat(n_groups, 1)            # (N_z*T, n_vars)
    cost_full = cost_full.index_add(1, flat_free_idx, lam_correction.reshape(n_groups, -1))

    # ── Sous-problèmes : N_z*T appels theta+zeta (glouton, pas CP-SAT) ───────
    prob = get_ssp_dld_problem(inst, free_z_np, data.x_mask_full.cpu().numpy())
    cost_np = cost_full.detach().cpu().numpy().astype(np.float64)  # (n_groups, n_vars)

    theta_np   = np.zeros((N_z, T))
    zeta_np    = np.zeros((N_z, T))
    x_theta_np = np.zeros((n_groups, prob.n_vars), dtype=np.float64)
    x_zeta_np  = np.zeros((n_groups, prob.n_vars), dtype=np.float64)
    for i_local in range(N_z):
        for j in range(T):
            g_idx = i_local * T + j
            v_t, x_t = prob.solve_theta(j, cost_np[g_idx])
            v_z, x_z = prob.solve_zeta(i_local, j, cost_np[g_idx])
            theta_np[i_local, j] = v_t
            zeta_np[i_local, j]  = v_z
            x_theta_np[g_idx] = x_t
            x_zeta_np[g_idx]  = x_z

    delta_ij = np.maximum(0.0, theta_np - zeta_np)                 # (N_z, T)
    Delta_i  = delta_ij.mean(axis=1)                                # (N_z,) — (1/T) Σ_j

    # (C) y* greedy sur Δ_i (au niveau scénario, PAS (i,j)) — même mécanisme que SLD
    required_prob = max(0.0, (1.0 - epsilon) - n_fixed_z1 * p_i)
    order   = np.argsort(Delta_i)
    y_star  = np.zeros(N_z, dtype=np.float32)
    cumprob = 0.0
    for idx in order:
        if required_prob <= 0.0:
            break
        y_star[idx] = 1.0
        cumprob += p_i
        if cumprob >= required_prob - 1e-9:
            break

    # (D) L1 linéaire en λ à x*, y* fixés (théorème de l'enveloppe)
    x_theta_t = torch.tensor(x_theta_np, dtype=torch.float32, device=dev)
    x_zeta_t  = torch.tensor(x_zeta_np,  dtype=torch.float32, device=dev)
    y_star_t  = torch.tensor(y_star,     dtype=torch.float32, device=dev)

    theta_lam_ij = (cost_full * x_theta_t).sum(dim=1)                        # (n_groups,)
    delta_lam_ij = (cost_full * (x_theta_t - x_zeta_t)).sum(dim=1)           # (n_groups,)
    delta_lam_i  = delta_lam_ij.view(N_z, T).mean(dim=1)                     # (N_z,) différentiable

    L1_theta = (p_ij * theta_lam_ij).sum()
    val_lam  = (p_i * y_star_t * delta_lam_i).sum()
    return L1_theta - val_lam


# ─────────────────────────────────────────────────────────────────────────────
# Vérification par sous-gradient Python pur — port de sld_solver.cpp
# ─────────────────────────────────────────────────────────────────────────────

def verify_dld_ssp(
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
    Port SSP de verify_dld (MKP-style) — reproduit l'algorithme de sous-
    gradient dld_solver.cpp en Python pur, via SSPDLDProblemBnB (glouton
    masqué). N_z*T appels theta+zeta par itération (T ici = nb itérations,
    à ne pas confondre avec l'horizon — cf. paramètre `T_hor`).

    v_lower : borne inférieure connue sur DLD* (ex. −opt via Gecode/B&B).
        Si fournie, active le pas de Polyak (gap/‖g‖²) — cf. dld_solver.cpp.
        Sinon (None), retombe sur le pas 1/√(t+1) — DANGEREUX ici : le facteur
        d'amplification est λ_ij/p_ij avec p_ij = p_i/T, donc ×(N·T) au lieu
        de ×N (SLD) — c'est ce qui produisait les bornes dégénérées
        (cost_lower de l'ordre de -100 à -500) observées avec le pas fixe.
    lam_init : (N_z, T_hor, G, 4) float64 ; None = cold start λ=0.
    """
    N_z        = data.N_z
    G          = data.n_g
    N          = data.N
    E, T_hor   = int(data.E), int(data.T)
    n_fixed_z1 = data.n_fixed_z1
    epsilon    = data.epsilon.item()
    n_groups   = N_z * T_hor

    free_et_np = data.free_et.numpy()
    free_z_np  = data.free_z.numpy()

    c_full_np = _base_cost_full(data.salary_full, E, T_hor, "cpu").numpy().astype(np.float64)

    flat_free_idx = (free_et_np[:, None] * SSP_SIGMA_SIZE
                     + np.arange(SSP_SIGMA_SIZE)[None, :]).reshape(-1)     # (G*4,)

    prob = get_ssp_dld_problem(inst, free_z_np, data.x_mask_full.numpy())

    p_i  = 1.0 / N
    p_ij = p_i / T_hor
    required_prob = max(0.0, (1.0 - epsilon) - n_fixed_z1 * p_i)

    lam = (lam_init.copy() if lam_init is not None
           else np.zeros((N_z, T_hor, G, SSP_SIGMA_SIZE), dtype=np.float64))

    best       = np.inf
    best_lam   = lam.copy()
    no_improve = 0
    n_iters    = T

    for t in range(T):
        # FIX (2026-08-13) : centrage PAR SCÉNARIO (axe période seulement),
        # miroir du fix dans lagrangian_dld_ssp et solver_generic/dld_solver.cpp
        # -- cf. PREUVE_DLD_centrage_par_scenario.md. L'ancien centrage
        # global (axis=(0,1)) est la cause des "bornes dégénérées" mentionnées
        # ci-dessus, pas seulement le facteur d'amplification N·T.
        Lsum_i         = lam.sum(axis=1)                                # (N_z, G, 4)
        lam_correction = lam / p_ij - (Lsum_i / p_i)[:, None, :, :]     # (N_z,T_hor,G,4)

        cost_full = np.broadcast_to(c_full_np, (n_groups, c_full_np.shape[0])).copy()
        cost_full[:, flat_free_idx] += lam_correction.reshape(n_groups, -1)

        theta_vals   = np.zeros((N_z, T_hor))
        zeta_vals    = np.zeros((N_z, T_hor))
        x_theta_full = np.zeros((n_groups, prob.n_vars))
        x_zeta_full  = np.zeros((n_groups, prob.n_vars))
        for i in range(N_z):
            for j in range(T_hor):
                g_idx = i * T_hor + j
                v_t, x_t = prob.solve_theta(j, cost_full[g_idx])
                v_z, x_z = prob.solve_zeta(i, j, cost_full[g_idx])
                theta_vals[i, j] = v_t
                zeta_vals[i, j]  = v_z
                x_theta_full[g_idx] = x_t
                x_zeta_full[g_idx]  = x_z

        delta_ij = np.maximum(0.0, theta_vals - zeta_vals)             # (N_z, T_hor)
        Delta_i  = delta_ij.mean(axis=1)                                # (N_z,)

        order   = np.argsort(Delta_i)
        y_star  = np.zeros(N_z)
        cumprob = 0.0
        for idx in order:
            if required_prob <= 0.0:
                break
            y_star[idx] = 1.0
            cumprob += p_i
            if cumprob >= required_prob - 1e-9:
                break

        L1 = p_ij * theta_vals.sum() - p_i * (y_star * Delta_i).sum()

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
        x_theta_g = x_theta_full[:, flat_free_idx].reshape(N_z, T_hor, G, SSP_SIGMA_SIZE)
        x_zeta_g  = x_zeta_full[:,  flat_free_idx].reshape(N_z, T_hor, G, SSP_SIGMA_SIZE)

        x_bar = np.zeros((G, SSP_SIGMA_SIZE))
        for i in range(N_z):
            active = y_star[i] > 0.5
            for j in range(T_hor):
                xij = x_zeta_g[i, j] if active else x_theta_g[i, j]
                x_bar += p_ij * xij

        # ── Norme du sous-gradient ‖g‖² = Σ_{i,j} ‖x_{i,j}* − x̄‖² ──────────
        gnorm_sq = 0.0
        for i in range(N_z):
            active = y_star[i] > 0.5
            for j in range(T_hor):
                xij = x_zeta_g[i, j] if active else x_theta_g[i, j]
                gnorm_sq += float(((xij - x_bar) ** 2).sum())

        # Pas de Polyak si v_lower est fourni, sinon 1/√(t+1) — cf. dld_solver.cpp
        if v_lower is not None and v_lower > -1e17 and gnorm_sq > 1e-14:
            gap  = L1 - v_lower
            step = (gap / gnorm_sq) if gap > 0.0 else 1.0 / np.sqrt(t + 1)
        else:
            step = 1.0 / np.sqrt(t + 1)

        for i in range(N_z):
            active = y_star[i] > 0.5
            for j in range(T_hor):
                xij = x_zeta_g[i, j] if active else x_theta_g[i, j]
                lam[i, j] += step * (x_bar - xij)

    if return_n_iters and return_lam:
        return best, n_iters, best_lam
    if return_lam:
        return best, best_lam
    if return_n_iters:
        return best, n_iters
    return best


def _apply_center_ssp(lam: torch.Tensor, N_z: int, T: int, G: int) -> torch.Tensor:
    """
    λ_{i,j,(e,t,σ)} ← λ_{i,j,(e,t,σ)} − moyenne sur (i,j) (centrage joint
    scénario+période, différentiable) — généralisation de _apply_center (MKP)
    et _apply_center_ssp (SLD) : Λ_sum se somme ici sur (i,j) ensemble
    (cf. lagrangian_dld_ssp), donc la direction de jauge à éliminer est la
    moyenne jointe, pas seulement sur i.
    """
    lam_4d = lam.view(N_z, T, G, SSP_SIGMA_SIZE)
    return (lam_4d - lam_4d.mean(dim=(0, 1), keepdim=True)).reshape(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Overfit sur un graphe — validation de l'architecture
# ─────────────────────────────────────────────────────────────────────────────

def overfit_one_graph(data: pyg_data.Data, inst: dict, n_epochs: int = 1000, lr: float = 1e-3,
                       center_lam: bool = True):
    """Entraîne le GNN sur un seul graphe avec la vraie borne DLD (SSP) comme loss."""
    device = torch.device("cpu")
    data   = data.to(device)

    model     = GNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)

    # cf. bench_ssp.cpp:266-268 : convention max(−coût) → L1(λ) ≤ 0 toujours ;
    # cost_lower = −L1(λ) est la quantité réellement comparable à `opt` (positif).
    opt_val = data.opt.item()
    N_z, T, G = int(data.N_z), int(data.T), int(data.n_g)
    with torch.no_grad():
        lam0    = torch.zeros(N_z * T * G * SSP_SIGMA_SIZE)
        cl_init = -lagrangian_dld_ssp(lam0, data, inst).item()

    print(f"\n{'='*60}")
    print(f"Overfit SSP DLD — G={G}  N_z={N_z}  T={T}  N={data.N}  E={data.E}  "
          f"(groupes={N_z*T})")
    print(f"  opt={opt_val:.2f}   cost_lower(λ=0)={cl_init:.2f}")
    print(f"  graphe : {data.x.shape[0]} nœuds, {data.edge_index.shape[1]} arêtes")
    print(f"{'='*60}")
    print(f"{'epoch':>6}  {'cost_lower':>12}  {'gap_vs_opt%':>12}  {'gain_vs_l0%':>13}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        lam  = model(data)
        if center_lam:
            lam = _apply_center_ssp(lam, N_z, T, G)
        loss = lagrangian_dld_ssp(lam, data, inst)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 50 == 0 or epoch == 1:
            cl    = -loss.item()
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
        lam0    = torch.zeros(int(g0.N_z) * int(g0.T) * int(g0.n_g) * SSP_SIGMA_SIZE)
        cl_init = -lagrangian_dld_ssp(lam0, g0, inst0).item()   # cost_lower(λ=0), cf. bench_ssp.cpp:266-268
        print(f"\n── Vérification DLD SSP (T=20 itérations subgradient, pas Polyak) ──")
        print(f"  cost_lower(λ=0)  : {cl_init:.4f}")
        py_best    = verify_dld_ssp(g0, inst0, T=20, v_lower=-g0.opt.item())
        cl_subgrad = -py_best
        print(f"  subgradient T=20  : {cl_subgrad:.4f}")
        print(f"  opt               : {g0.opt.item():.4f}")
        print(f"  amélioration      : {(cl_subgrad - cl_init) / abs(cl_init) * 100:.1f}%")

    if mode == "overfit" and graphs:
        g0    = graphs[0]
        inst0 = instances[int(g0.inst_id.item())]
        overfit_one_graph(g0, inst0)
