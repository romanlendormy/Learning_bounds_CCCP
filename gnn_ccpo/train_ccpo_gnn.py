"""
train_ccpo_gnn.py — Boucle d'entraînement multi-instances du GNN CCPO, calquée le plus
fidèlement possible sur gnn_sd_edit/train_scaled_sd_edit.py (MKP).

Correspondances explicites avec MKP (mêmes noms de concepts, adaptés) :
  - discover_pairs/load_instances          -> discover_cells/load_all
  - LazyGraphDataset + stream_reservoir     -> nos fichiers nodes_*.txt sont
    déjà échantillonnés à la collecte (racine + 300/instance, collect_ccpo_samples.py) --
    pas besoin d'un second reservoir sampling, mais ON GARDE la sélection
    STRATIFIÉE par cellule (n,N) + cap sur les racines (root_frac), pour ne
    pas biaiser vers les cellules qui ont le plus de lignes brutes.
  - select_stratified                       -> stratified_select (même
    politique : budget égal par cellule, racines plafonnées, round-robin sur
    les instances pas encore touchées pour le reste).
  - split_by_instance (80/20, seed)         -> split_by_instance (identique).
  - DataLoader(batch_size=1) + accumulation -> même mécanisme (pas de vrai
    torch_geometric.Batch : nos graphes ont des tailles hétérogènes -- n_free
    ET n_scenarios_eligible varient tous les deux d'un nœud à l'autre, un
    problème strictement PLUS sévère que le n_g uniforme supposé par le
    pooling MKP). batch_size=32 (accumulation logique).
  - Adam lr=3e-4, weight_decay=0            -> identique (valeurs de
    gnn_sd_edit/run_train_sld.sh, pas le lr=3e-2 d'un réglage sur une seule
    instance -- régime d'overfit différent de l'entraînement réel).
  - LinearWarmup(10)+CosineAnnealingLR(eta_min=1e-5) -> identique.
  - clip_grad_norm_(max_norm=0.5) PAR EXEMPLE avant accumulation -> identique
    (choix delibéré de MKP : clip-puis-moyenne, pas moyenne-puis-clip).
  - warmup_on_roots (régression L2 vers λ* par sous-gradient, T=50,
    n_warmup=10, warmup_lr=1e-3) -> identique, via solve_ccpo_sld_raw
    (nouveau binding ccpo_lagrangian_bind.cpp::solve_ccpo_sld_c, ajouté
    spécifiquement pour cette étape).
  - save_checkpoint (tous les epochs, keep_every=10, keep_ckpts=3)
    -> identique.
  - PAS de régularisation ||λ||² (ALPHA_L2=0 en production MKP), PAS de
    pondération par profondeur (DEPTH_W0=1.0 = désactivé en production)
    -> on garde les MÊMES défauts (0 partout), conforme à MKP.
  - PAS de "best model" -- MKP n'en garde pas non plus, seulement le
    checkpointing périodique + les métriques JSON à inspecter après coup.

Différence assumée (pas dans MKP) : garde-fou explicite sur les évaluations
Lagrangiennes NaN (sous-problème résiduel prouvé infaisable : noeud profond
structurellement infaisable meme a lambda=0) -- un exemple
NaN est ignoré (ni backward ni comptage dans le batch logique), jamais un
loss NaN silencieusement propagé dans optimizer.step(). MKP n'a pas ce
problème sous la même forme (filtré en amont au chargement) mais CCPO peut légitimement produire un residual sous-pb
infaisable indépendant de lambda à n'importe quelle profondeur.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import random
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# CORRECTIF : ulimit -t (3600s de temps CPU CUMULE, pas de temps
# mural) sur ce cluster partage -- deja rencontre pour la generation de
# donnees (BLAS/OpenMP). Les variables d'environnement
# OMP_NUM_THREADS=1 etc. passees au lancement ne suffisent apparemment pas a
# contraindre tout le pool de threads interne de torch -- fixe explicitement
# ici, AVANT tout usage de torch, pour eviter que le process ne soit tue
# silencieusement vers ~30 min (temps CPU cumule sur plusieurs threads
# atteignant 3600s bien avant 3600s de temps mural).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from ccpo_dataset import CCPOInstanceRecord, CCPONodeRecord, load_instances_ccpo, load_nodes_ccpo, \
    reconstruct_residual_instance
from ccpo_graph import build_graph
from ccpo_gnn_model import CCPOGNN
from ccpo_lagrangian_autograd import (ccpo_lagrangian, evaluate_ccpo_lagrangian_raw, solve_ccpo_sld_raw,
                                      solve_ccpo_sld_projected)


# ═════════════════════════════════════════════════════════════════════════
# Découverte + chargement des cellules (n,N)
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class CellData:
    n: int
    N: int
    instances: dict[int, CCPOInstanceRecord]
    nodes_by_instance: dict[int, list[CCPONodeRecord]]


def discover_cells(data_dir: Path) -> list[tuple[int, int]]:
    cells = []
    for f in sorted(data_dir.glob("instances_n*_N*.txt")):
        # instances_n{n}_N{N}.txt
        stem = f.stem  # instances_n8_N20
        parts = stem.replace("instances_n", "").split("_N")
        n, N = int(parts[0]), int(parts[1])
        cells.append((n, N))
    return cells


def load_all(data_dir: Path, node_suffix: str = "") -> dict[tuple[int, int], CellData]:
    out = {}
    for n, N in discover_cells(data_dir):
        inst_path = data_dir / f"instances_n{n}_N{N}.txt"
        nodes_path = data_dir / f"nodes_n{n}_N{N}{node_suffix}.txt"
        if not nodes_path.exists():
            nodes_path = data_dir / f"nodes_n{n}_N{N}.txt"
        instances = load_instances_ccpo(str(inst_path))
        nodes = load_nodes_ccpo(str(nodes_path))
        by_inst: dict[int, list[CCPONodeRecord]] = {}
        for rec in nodes:
            by_inst.setdefault(rec.instance_id, []).append(rec)
        out[(n, N)] = CellData(n=n, N=N, instances=instances, nodes_by_instance=by_inst)
        print(f"[data] cellule n={n} N={N} : {len(instances)} instances, {len(nodes)} nœuds", flush=True)
    return out


# ═════════════════════════════════════════════════════════════════════════
# Sélection stratifiée (budget égal par cellule, racines plafonnées) --
# même politique que select_stratified() côté MKP.
# ═════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Example:
    n: int
    N: int
    instance_id: int
    depth: int
    node_idx: int  # index dans nodes_by_instance[instance_id]


def stratified_select(cells: dict[tuple[int, int], CellData], target_graphs: int,
                       root_frac: float, rng: random.Random) -> list[Example]:
    n_cells = len(cells)
    budget_per_cell = max(1, target_graphs // n_cells)
    selected: list[Example] = []

    for (n, N), cell in cells.items():
        root_budget = max(1, round(budget_per_cell * root_frac))
        roots: list[Example] = []
        non_roots_by_inst: dict[int, list[Example]] = {}
        for inst_id, recs in cell.nodes_by_instance.items():
            for idx, rec in enumerate(recs):
                ex = Example(n, N, inst_id, rec.depth, idx)
                if rec.depth == 0:
                    roots.append(ex)
                else:
                    non_roots_by_inst.setdefault(inst_id, []).append(ex)

        rng.shuffle(roots)
        kept_roots = roots[:min(root_budget, len(roots))]
        selected.extend(kept_roots)

        remaining = budget_per_cell - len(kept_roots)
        touched_by_root = {ex.instance_id for ex in kept_roots}
        # Round-robin : priorise les instances pas encore touchées par une racine.
        inst_order = sorted(non_roots_by_inst.keys(),
                             key=lambda i: (i in touched_by_root, rng.random()))
        pools = {i: non_roots_by_inst[i][:] for i in inst_order}
        for pool in pools.values():
            rng.shuffle(pool)
        kept_non_root = []
        while remaining > 0 and any(pools.values()):
            for i in inst_order:
                if remaining <= 0:
                    break
                if pools[i]:
                    kept_non_root.append(pools[i].pop())
                    remaining -= 1
        selected.extend(kept_non_root)
        print(f"[stratify] n={n} N={N} : {len(kept_roots)} racines + "
              f"{len(kept_non_root)} non-racines = {len(kept_roots) + len(kept_non_root)}", flush=True)

    return selected


def split_by_instance(examples: list[Example], train_frac: float,
                       seed: int) -> tuple[list[Example], list[Example]]:
    keys = sorted({(e.n, e.N, e.instance_id) for e in examples})
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_train = int(round(len(keys) * train_frac))
    train_keys = set(keys[:n_train])
    train = [e for e in examples if (e.n, e.N, e.instance_id) in train_keys]
    val = [e for e in examples if (e.n, e.N, e.instance_id) not in train_keys]
    return train, val


# ═════════════════════════════════════════════════════════════════════════
# Construction d'un exemple (graphe + tout ce qu'il faut pour évaluer L(λ))
# ═════════════════════════════════════════════════════════════════════════

def instance_scales(inst: CCPOInstanceRecord) -> tuple[float, float]:
    c_scale = float(np.max(np.abs(inst.c_int))) if inst.c_int.size else 1.0
    c_scale = max(c_scale, 1.0)
    eligible = np.where(~inst.infeasible)[0]
    if len(eligible) == 0:
        sigma_scale = 1.0
    else:
        diags = [np.diagonal(inst.Sigma_dense[j]).mean() for j in eligible]
        sigma_scale = max(float(np.mean(diags)), 1e-6)
    return c_scale, sigma_scale


# Configuration du graphe : version des features et mode de poids d'arêtes.
# Défauts = comportement d'origine (v1, poids Σ stockés mais ignorés si use_edge_weight=False).
# main() la fixe d'après --feat_version/--edge_mode ; les scripts de diagnostic qui
# chargent un checkpoint v2 doivent appeler configure_graph_from_args(ckpt["args"]).
GRAPH_CFG = {"feat_version": 1, "edge_mode": "sigma"}
# Device du MODÈLE (--device) : le calcul du Lagrangien (C++/ctypes)
# reste toujours sur CPU ; seuls le forward/backward du GNN vont sur DEVICE.
DEVICE = torch.device("cpu")
# Threads OpenMP PAR évaluation du Lagrangien (--omp_threads) : 0 = séquentiel.
EVAL_OMP = 0


def _to_dev(graph):
    return graph if DEVICE.type == "cpu" else graph.to(DEVICE)


def resolve_edge_mode(args: dict) -> str:
    em = args.get("edge_mode")
    if em:
        return em
    return "sigma" if args.get("edge_weight_sigma", False) else "none"


def configure_graph_from_args(args: dict) -> None:
    GRAPH_CFG["feat_version"] = int(args.get("feat_version", 1))
    em = resolve_edge_mode(args)
    GRAPH_CFG["edge_mode"] = "sigma" if em == "none" else em  # "none" : poids ignorés par le modèle


def model_from_args(args: dict) -> CCPOGNN:
    """CCPOGNN correspondant à ckpt["args"] (anciens checkpoints -> configuration d'origine)."""
    from ccpo_graph import node_dim_for
    em = resolve_edge_mode(args)
    return CCPOGNN(node_dim=node_dim_for(int(args.get("feat_version", 1))), hidden=int(args["hidden"]),
                   use_edge_weight=(em != "none"), center_lam=bool(args.get("center_lam", False)),
                   relu_h1=bool(args.get("relu_h1", True)))


def build_from_example(cells: dict[tuple[int, int], CellData], ex: Example):
    """Retourne (graph, ri, order, infeasible_u8) ou None si le nœud est une
    feuille (aucune variable libre) -- rien à prédire, cf. reconstruct_residual_instance."""
    cell = cells[(ex.n, ex.N)]
    inst = cell.instances[ex.instance_id]
    node = cell.nodes_by_instance[ex.instance_id][ex.node_idx]
    c_scale, sigma_scale = instance_scales(inst)
    ri = reconstruct_residual_instance(inst, node, c_scale, sigma_scale)
    if ri is None:
        return None
    graph = build_graph(ri, GRAPH_CFG["feat_version"], GRAPH_CFG["edge_mode"])
    order = np.argsort(-ri.cost_free).astype(np.int32)
    infeasible_u8 = inst.infeasible.astype(np.uint8)
    return graph, ri, order, infeasible_u8, inst


# ═════════════════════════════════════════════════════════════════════════
# Checkpointing -- identique en esprit à save_checkpoint/find_latest_checkpoint (MKP)
# ═════════════════════════════════════════════════════════════════════════

def ckpt_path(out_dir: Path, run_name: str, epoch: int) -> Path:
    return out_dir / f"ckpt_{run_name}_ep{epoch:04d}.pt"


def _atomic_torch_save(obj, path: Path) -> None:
    """Ecriture atomique : fichier temporaire dans le MEME dossier puis
    os.replace. Un job tue pendant l'ecriture laisse au pire un .tmp orphelin,
    jamais un checkpoint tronque sous le nom final ."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_checkpoint(out_dir: Path, run_name: str, epoch: int, model, optimizer, scheduler,
                     history: dict, args: argparse.Namespace, saved: list[Path],
                     keep: int, keep_every: int = 10):
    path = ckpt_path(out_dir, run_name, epoch)
    _atomic_torch_save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "history": history, "args": vars(args)}, path)
    if epoch % keep_every == 0:
        return
    saved.append(path)
    if len(saved) > keep:
        saved.pop(0).unlink(missing_ok=True)


def list_checkpoints(out_dir: Path, run_name: str) -> list[Path]:
    return sorted(out_dir.glob(f"ckpt_{run_name}_ep*.pt"))


def find_latest_checkpoint(out_dir: Path, run_name: str) -> Path | None:
    ckpts = list_checkpoints(out_dir, run_name)
    return ckpts[-1] if ckpts else None


def load_latest_valid_checkpoint(out_dir: Path, run_name: str):
    """Charge le checkpoint le plus recent LISIBLE ; si le dernier est corrompu
    (job tue pendant une ecriture, ancienne version non atomique), retombe sur
    le precedent avec un avertissement. Retourne (path, ckpt) ou (None, None)."""
    for path in reversed(list_checkpoints(out_dir, run_name)):
        try:
            return path, torch.load(path, map_location="cpu")
        except Exception as e:  # noqa: BLE001 -- fichier tronque/corrompu : on essaie le precedent
            print(f"[resume] ATTENTION : {path.name} illisible ({type(e).__name__}: {e}) -- "
                  f"repli sur le checkpoint precedent", flush=True)
    return None, None


# ── Verrou anti-double-job sur un meme out_dir . ──────
# Detection de peremption par HEARTBEAT (mtime du fichier de verrou, rafraichi par
# un thread toutes les LOCK_BEAT_S) et NON par squeue : squeue echoue depuis un noeud
# de calcul (constate sur Trillium, test_resume 2487466), alors que le cas normal d'un
# verrou orphelin est un job tue par TIMEOUT/SIGKILL.
_LOCK_RELEASE = None   # rempli par acquire_run_lock ; appele avant os._exit (qui saute atexit)
LOCK_BEAT_S = 30
LOCK_STALE_S = 120
LOCK_OBSERVE_S = 45


def acquire_run_lock(out_dir: Path, run_name: str) -> None:
    import threading
    lock = out_dir / f".train_{run_name}.lock"
    me = {"job": os.environ.get("SLURM_JOB_ID", ""), "host": socket.gethostname(),
          "pid": os.getpid(), "t": time.time()}

    def _try_create() -> bool:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as f:
            json.dump(me, f)
        return True

    for _attempt in range(3):
        if _try_create():
            break
        try:
            other = json.load(open(lock))
        except Exception:  # noqa: BLE001
            other = {}
        try:
            age = time.time() - lock.stat().st_mtime
        except FileNotFoundError:
            continue
        same_job = bool(me["job"]) and other.get("job") == me["job"]   # requeue du MEME job
        stale = same_job or age > LOCK_STALE_S
        if not stale:
            # heartbeat recent : verifier qu'il progresse (un detenteur vivant rafraichit
            # le fichier toutes les LOCK_BEAT_S ; un job tout juste tue ne le fait plus)
            m0 = lock.stat().st_mtime
            print(f"[lock] verrou recent ({other}, age {age:.0f}s) -- observation {LOCK_OBSERVE_S}s ...", flush=True)
            time.sleep(LOCK_OBSERVE_S)
            try:
                alive = lock.stat().st_mtime > m0
            except FileNotFoundError:
                alive = False
            if alive:
                print(f"[lock] REFUS : out_dir deja utilise par un job actif {other}. "
                      f"Si c'est une erreur, supprimer {lock}", flush=True)
                sys.exit(3)
        print(f"[lock] verrou perime ({other}) -- reprise en main", flush=True)
        lock.unlink(missing_ok=True)
    else:
        print("[lock] impossible d'acquerir le verrou", flush=True)
        sys.exit(3)

    stop = threading.Event()

    def _beat():
        while not stop.wait(LOCK_BEAT_S):
            try:
                os.utime(lock, None)
            except OSError:
                pass
    threading.Thread(target=_beat, daemon=True).start()

    def _release():
        stop.set()
        try:
            if json.load(open(lock)).get("pid") == os.getpid():
                lock.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
    atexit.register(_release)
    global _LOCK_RELEASE
    _LOCK_RELEASE = _release


def _intra_epoch_ckpt_path(out_dir: Path, run_name: str) -> Path:
    return out_dir / f"intra_epoch_{run_name}.pt"


def save_intra_epoch_checkpoint(out_dir: Path, run_name: str, epoch: int, example_idx: int,
                                 accum_loss: float, accum_count: int, n_skipped: int,
                                 model, optimizer, extra: dict | None = None) -> None:
    """CORRECTIF : une epoch complete (6723 exemples) peut
    depasser le ulimit -t=3600-5400s du cluster partage avant meme de
    terminer -- contrairement au warmup (deja resumable via cache), la
    boucle principale ne sauvegardait qu'A LA FIN de chaque epoch complete,
    donc une coupure en cours d'epoch perdait tout son travail, a chaque
    fois, sans jamais pouvoir progresser (). Sauvegarde
    intermediaire, ecrasee a chaque appel (pas d'historique, juste le
    dernier point de reprise) -- retiree une fois l'epoch reellement finie."""
    tmp = _intra_epoch_ckpt_path(out_dir, run_name)
    tmp_new = tmp.with_suffix(".pt.tmp")
    torch.save({
        "epoch": epoch, "example_idx": example_idx,
        "accum_loss": accum_loss, "accum_count": accum_count, "n_skipped": n_skipped,
        "extra": extra or {},
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
    }, tmp_new)
    tmp_new.replace(tmp)  # ecriture atomique -- jamais de fichier a moitie ecrit si coupure ici


def load_intra_epoch_checkpoint(out_dir: Path, run_name: str) -> dict | None:
    path = _intra_epoch_ckpt_path(out_dir, run_name)
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


# ═════════════════════════════════════════════════════════════════════════
# Évaluation de L(λ_GNN) pour un exemple -- coeur de la perte (auto-supervisé)
# ═════════════════════════════════════════════════════════════════════════

def eval_example_loss(model: CCPOGNN, cells, ex: Example) -> torch.Tensor | None:
    built = build_from_example(cells, ex)
    if built is None:
        return None
    graph, ri, order, infeasible_u8, inst = built
    graph = _to_dev(graph)
    lam_full = model.build_lambda_tensor(graph, N=ri.N, n_free=ri.n_free, c_scale=ri.c_scale)
    Lv = ccpo_lagrangian(lam_full, ri.n_free, ri.N, ri.k_res, ri.Q_res, ri.L, ri.U, inst.min_feas,
                          ri.cost_free, ri.Sigma_free, ri.B_res, infeasible_u8, order, ri.R_ext)
    if torch.isnan(Lv):
        return None
    return Lv / ri.c_scale


# ═════════════════════════════════════════════════════════════════════════
# Traitement PARALLELE d'un micro-batch (optimisation threading).
#
# evaluate_ccpo_lagrangian_c (ccpo_lagrangian_bind.cpp) est un appel ctypes
# vers une fonction C++ pure -- ctypes relache le GIL pendant l'appel (CPython
# standard), donc plusieurs threads Python peuvent executer ce calcul
# (zeta_solve, EXACT, sans budget de temps -- le vrai goulot d'etranglement
# de tout ce projet) EN PARALLELE, sur des coeurs distincts. C'est
# spécifiquement la partie couteuse qu'on parallelise -- le forward
# (build_lambda_tensor) et le backward (populent model.grad, etat PARTAGE)
# restent SEQUENTIELS pour eviter toute race condition sur les gradients.
#
# Sur le noeud de connexion, torch.set_num_threads(1) etait necessaire pour
# eviter le ulimit -t=3600s (). Sur un
# noeud SLURM dedie (192 coeurs, aucune contention), ce garde-fou reste en
# place pour les operations TENSORIELLES internes de torch (BLAS/OMP), mais
# n'empeche PAS ce parallelisme manuel au niveau Python/ctypes -- ce sont
# deux mecanismes de threading distincts.
# ═════════════════════════════════════════════════════════════════════════

def _prepare_example(model: CCPOGNN, cells, ex: Example):
    """Phase 1 (sequentielle, rapide) : construit le graphe et calcule
    lambda_GNN. Touche le modele partage -- doit rester hors thread."""
    built = build_from_example(cells, ex)
    if built is None:
        return None
    graph, ri, order, infeasible_u8, inst = built
    graph = _to_dev(graph)
    lam_full = model.build_lambda_tensor(graph, N=ri.N, n_free=ri.n_free, c_scale=ri.c_scale)
    return lam_full, ri, order, infeasible_u8, inst


def _cpp_eval_only(parts):
    """Phase 2 (parallelisable) : le vrai calcul couteux, sans dependance a
    l'etat partage du modele -- chaque appel travaille sur des tableaux
    numpy locaux a cet exemple."""
    lam_full, ri, order, infeasible_u8, inst = parts
    lam_np = lam_full.detach().cpu().numpy().astype(np.float64)
    return evaluate_ccpo_lagrangian_raw(
        ri.n_free, ri.N, ri.k_res, ri.Q_res, ri.L, ri.U, inst.min_feas,
        ri.cost_free, ri.Sigma_free, ri.B_res, infeasible_u8, lam_np, order, ri.R_ext,
        omp_threads=EVAL_OMP)


def _ex_key(ex: Example) -> tuple:
    return (ex.n, ex.N, ex.instance_id, ex.node_idx)


def compute_L0_map(cells, examples: list[Example], executor: ThreadPoolExecutor,
                    cache_path: Path) -> dict:
    """L(lambda=0) par exemple (cache JSONL, resumable) -- la reference pour
    mesurer l'amelioration RELATIVE du GNN (imp = (L0 - L)/|L0|) pendant
    l'entrainement, et pour la perte relative (--loss_mode rel)."""
    cache: dict = {}
    if cache_path.exists():
        for line in open(cache_path):
            r = json.loads(line)
            cache[(r["n"], r["N"], r["inst"], r["idx"])] = r["L0"]
    todo = [ex for ex in examples if _ex_key(ex) not in cache]
    print(f"[L0] {len(cache)} en cache, {len(todo)} a calculer", flush=True)
    if todo:
        with open(cache_path, "a") as f:
            pending = []
            def flush_pending():
                for ex, fut in pending:
                    L0, _, feas = fut.result()
                    if feas and not math.isnan(L0):
                        cache[_ex_key(ex)] = L0
                        f.write(json.dumps({"n": ex.n, "N": ex.N, "inst": ex.instance_id,
                                            "idx": ex.node_idx, "L0": L0}) + "\n")
                f.flush()
                pending.clear()
            for ex in todo:
                built = build_from_example(cells, ex)
                if built is None:
                    continue
                graph, ri, order, infeasible_u8, inst = built
                lam0 = np.zeros((ri.N, ri.n_free), dtype=np.float64)
                fut = executor.submit(
                    evaluate_ccpo_lagrangian_raw, ri.n_free, ri.N, ri.k_res, ri.Q_res, ri.L, ri.U,
                    inst.min_feas, ri.cost_free, ri.Sigma_free, ri.B_res, infeasible_u8, lam0,
                    order, ri.R_ext, EVAL_OMP)
                pending.append((ex, fut))
                if len(pending) >= 256:
                    flush_pending()
            flush_pending()
    return cache


def process_chunk_parallel(model: CCPOGNN, cells, chunk: list[Example],
                            executor: ThreadPoolExecutor, max_norm: float,
                            clip_mode: str = "per_example", l0_map: dict | None = None,
                            loss_mode: str = "abs", rel_scale: float = 8.5,
                            hinge_beta: float = 0.0):
    """Traite un micro-batch : forward sequentiel -> evaluations C++ EN
    PARALLELE (GIL relache pendant chaque appel ctypes) -> backward sequentiel
    (theoreme de l'enveloppe : dL/dlam = subgrad deja calcule cote C++).

    clip_mode  :
      "per_example" (defaut, comme MKP train_scaled_sd_edit.py:924-948) : le
          gradient de CHAQUE exemple est ecrete individuellement puis somme ;
          l'appelant divise par count.
      "accumulated" (ancien comportement 23/09 -> 02/10, CONSERVE pour
          l'ablation) : clip_grad_norm_ sur le CUMUL apres chaque exemple.
    loss_mode (ablation) :
      "abs" : poids 1/c_scale (comportement d'origine : perte L/c_scale).
      "rel" : perte relative rel_scale*(L-L0)/|L0| (L0 = L(lambda=0)), avec
          garde-fou : poids x(1+hinge_beta) quand L > L0 (borne pire que ne rien
          faire). rel_scale ~ 8.5 = L0/c_scale moyen, pour garder une echelle de
          gradient comparable a "abs" vis-a-vis de max_norm.

    Retourne un dict : loss_sum (L/c_scale), count, skipped, gn_sum, gn_max,
    n_clipped, imp_sum (somme des (L0-L)/|L0|), n_worse (L > L0), n_imp."""
    params = [p for p in model.parameters()]
    prepared = []
    n_skipped = 0
    t0 = time.perf_counter()
    for ex in chunk:
        parts = _prepare_example(model, cells, ex)
        if parts is None:
            n_skipped += 1
            continue
        prepared.append((ex, parts, executor.submit(_cpp_eval_only, parts)))

    st = dict(loss_sum=0.0, count=0, skipped=0, gn_sum=0.0, gn_max=0.0, n_clipped=0,
              imp_sum=0.0, n_worse=0, n_imp=0, t_fwd=time.perf_counter() - t0, t_wait=0.0, t_bwd=0.0)
    acc = None
    for ex, parts, fut in prepared:
        lam_full, ri, order, infeasible_u8, inst = parts
        tw = time.perf_counter()
        L_val, subgrad_np, feasible = fut.result()
        tb = time.perf_counter()
        st["t_wait"] += tb - tw
        if not feasible or math.isnan(L_val):
            n_skipped += 1
            continue
        L0 = l0_map.get(_ex_key(ex)) if l0_map is not None else None
        if L0 is not None:
            st["imp_sum"] += (L0 - L_val) / abs(L0)
            st["n_worse"] += int(L_val > L0)
            st["n_imp"] += 1
        if loss_mode == "rel" and L0 is not None:
            scale = rel_scale / abs(L0) * (1.0 + hinge_beta if L_val > L0 else 1.0)
        else:
            scale = 1.0 / ri.c_scale
        subgrad_t = torch.from_numpy(subgrad_np).to(dtype=lam_full.dtype, device=lam_full.device) * scale
        if clip_mode == "per_example":
            for p in params:
                p.grad = None
            lam_full.backward(subgrad_t)
            gn = float(torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm))
            if acc is None:
                acc = [p.grad.detach().clone() if p.grad is not None else None for p in params]
            else:
                for j, p in enumerate(params):
                    if p.grad is not None and acc[j] is not None:
                        acc[j] += p.grad.detach()
        else:  # "accumulated" (legacy)
            lam_full.backward(subgrad_t)
            gn = float(torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm))
        st["t_bwd"] += time.perf_counter() - tb
        st["gn_sum"] += gn
        st["gn_max"] = max(st["gn_max"], gn)
        st["n_clipped"] += int(gn > max_norm)
        st["loss_sum"] += L_val / ri.c_scale
        st["count"] += 1

    if clip_mode == "per_example":
        for p, a_ in zip(params, acc if acc is not None else [None] * len(params)):
            p.grad = a_
    st["skipped"] = n_skipped
    return st


def eval_metrics(model: CCPOGNN, cells, examples: list[Example], l0_map: dict,
                 executor: ThreadPoolExecutor, infeasible_keys: set | None = None) -> dict:
    """Amelioration relative de la borne vs L(0) sur `examples` (sans gradient),
    par tranche de n_free : la metrique qui compte (la loss brute masque des
    ecarts de 1-3 %)."""
    model.eval()
    rows = []
    with torch.no_grad():
        futs = []
        for ex in examples:
            parts = _prepare_example(model, cells, ex)
            if parts is None:
                continue
            futs.append((ex, parts, executor.submit(_cpp_eval_only, parts)))
        for ex, parts, fut in futs:
            L_val, _, feas = fut.result()
            L0 = l0_map.get(_ex_key(ex))
            if not feas or math.isnan(L_val) or L0 is None:
                continue
            rows.append((parts[1].n_free, (L0 - L_val) / abs(L0), _ex_key(ex)))
    model.train()
    out = {"n": len(rows)}
    if rows:
        imp = np.array([r[1] for r in rows]); nf = np.array([r[0] for r in rows])
        out["imp_mean"] = float(imp.mean()); out["imp_med"] = float(np.median(imp))
        out["frac_worse"] = float(np.mean(imp < -1e-9))
        for lo, hi in [(1, 4), (5, 7), (8, 12)]:
            m = (nf >= lo) & (nf <= hi)
            if m.any():
                out[f"imp_{lo}-{hi}"] = (float(imp[m].mean()), float(np.mean(imp[m] < -1e-9)), int(m.sum()))
        if infeasible_keys is not None:
            # métrique ROBUSTE = nœuds VAL dont le sous-arbre n'est PAS prouvé infaisable (force brute,
            # --exclude_nodes) ; sur un nœud infaisable L(λ) n'est pas borné et (L0-L)/|L0| non plus.
            keep = np.array([r[2] not in infeasible_keys for r in rows])
            if keep.any():
                out["imp_mean_feas"] = float(imp[keep].mean()); out["imp_med_feas"] = float(np.median(imp[keep]))
                out["n_feas"] = int(keep.sum())
    return out


# ═════════════════════════════════════════════════════════════════════════
# Préchauffage supervisé (régression L2 vers λ* par sous-gradient, T=warmup_T)
# -- sur le patron de warmup_on_roots (MKP).
# ═════════════════════════════════════════════════════════════════════════

def resolve_warmup_target_mode(args) -> str:
    """Famille de λ des cibles de préchauffage.
      legacy : sous-gradient complet sur les N lignes (solve_ccpo_sld C++),
               tronqué aux lignes éligibles -- historique, INCOHÉRENT avec ce
               que le GNN peut produire (lignes infaisables à 0) ;
      E0     : sous-gradient projeté, lignes infaisables gelées à 0 ;
      E0c    : E0 + centrage sur E (cohérent avec --center_lam) ;
      auto   : E0c si center_lam, sinon E0."""
    m = getattr(args, "warmup_target_mode", "legacy")
    if m == "auto":
        m = "E0c" if getattr(args, "center_lam", False) else "E0"
    if getattr(args, "center_lam", False) and m != "E0c":
        print(f"[warmup] ATTENTION : center_lam avec des cibles '{m}' non centrées -- "
              "la régression MSE ne peut pas atteindre ses cibles.", flush=True)
    return m


def _warmup_cache_path(out_dir: Path, run_name: str, mode: str = "legacy") -> Path:
    # legacy : nom historique (caches existants relus tels quels) ; autres modes :
    # suffixe -> jamais de mélange de cibles de familles différentes.
    suffix = "" if mode == "legacy" else f"_{mode}"
    return out_dir / f"warmup_targets_{run_name}{suffix}.jsonl"


def _load_warmup_cache(cache_path: Path) -> dict[tuple, list]:
    """Relit les cibles lambda* deja calculees (reprise apres coupure d'un job,
    comme curate_feasible_examples.py)."""
    cache: dict[tuple, list] = {}
    if not cache_path.exists():
        return cache
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec["n"], rec["N"], rec["instance_id"], rec["node_idx"])
            cache[key] = rec["lam_star"]
    return cache


def warmup_on_roots(model: CCPOGNN, cells, train_examples: list[Example], args,
                     out_dir: Path, run_name: str) -> None:
    eligible = [e for e in train_examples if e.n <= args.warmup_max_n]
    if not eligible:
        print("[warmup] aucun exemple eligible (warmup_max_n trop bas) -- warmup ignore.")
        return
    if args.warmup_n_examples is not None and len(eligible) > args.warmup_n_examples:
        rng_wu = random.Random(args.seed + 7)
        rng_wu.shuffle(eligible)
        eligible = eligible[:args.warmup_n_examples]
    print(f"[warmup] {len(eligible)} exemples eligibles (n<={args.warmup_max_n}), "
          f"{args.n_warmup} epochs, T={args.warmup_t}", flush=True)

    mode = resolve_warmup_target_mode(args)
    print(f"[warmup] cibles : mode {mode}", flush=True)
    cache_path = _warmup_cache_path(out_dir, run_name, mode)
    cache = _load_warmup_cache(cache_path)
    if cache:
        print(f"[warmup] reprise -- {len(cache)} cibles deja en cache ({cache_path.name})", flush=True)

    targets = []
    n_from_cache = n_computed = n_skipped = 0
    cache_file = open(cache_path, "a")
    # les sous-gradients T=warmup_t des cibles lambda* sont
    # independants et ctypes relache le GIL -> calcules en PARALLELE (n_workers
    # threads) au lieu de sequentiellement (sur N=100 certains noeuds coutent des
    # minutes : 300 cibles en serie = plusieurs heures). Ordre des cibles conserve.
    pending = []   # (ex, graph, ri, key, lam_star_list|None, future|None)
    with ThreadPoolExecutor(max_workers=max(1, args.n_workers)) as pool:
        for ex in eligible:
            built = build_from_example(cells, ex)
            if built is None:
                n_skipped += 1
                continue
            graph, ri, order, infeasible_u8, inst = built
            key = (ex.n, ex.N, ex.instance_id, ex.node_idx)
            if key in cache:
                pending.append((ex, graph, ri, key, infeasible_u8, cache[key], None))
            elif mode == "legacy":
                fut = pool.submit(
                    solve_ccpo_sld_raw, ri.n_free, ri.N, ri.k_res, ri.Q_res, ri.L, ri.U, inst.min_feas,
                    ri.cost_free, ri.Sigma_free, ri.B_res, infeasible_u8, ri.R_ext,
                    args.warmup_t, 1e-3, 15)
                pending.append((ex, graph, ri, key, infeasible_u8, None, fut))
            else:
                fut = pool.submit(
                    solve_ccpo_sld_projected, ri.n_free, ri.N, ri.k_res, ri.Q_res, ri.L, ri.U,
                    inst.min_feas, ri.cost_free, ri.Sigma_free, ri.B_res, infeasible_u8, order,
                    ri.R_ext, mode, args.warmup_t, 1e-3, 15, EVAL_OMP)
                pending.append((ex, graph, ri, key, infeasible_u8, None, fut))
        for i, (ex, graph, ri, key, infeasible_u8, lam_star_list, fut) in enumerate(pending):
            if fut is None:
                n_from_cache += 1
            else:
                lam_star, L_star, feasible = fut.result()[:3]
                if not feasible:
                    n_skipped += 1
                    continue
                eligible_idx_np = np.where(~infeasible_u8.astype(bool))[0]
                lam_star_list = (lam_star[eligible_idx_np] * ri.N / ri.c_scale).tolist()
                cache_file.write(json.dumps({
                    "n": ex.n, "N": ex.N, "instance_id": ex.instance_id, "node_idx": ex.node_idx,
                    "lam_star": lam_star_list, "mode": mode, "L_star": L_star,
                }) + "\n")
                cache_file.flush()
                n_computed += 1
            targets.append((_to_dev(graph), ri, torch.tensor(lam_star_list, dtype=torch.float32, device=DEVICE)))
            if (i + 1) % 50 == 0:
                print(f"[warmup] cibles : {i+1}/{len(pending)} traites "
                      f"({n_from_cache} cache, {n_computed} calcules, {n_skipped} ignores)", flush=True)
    cache_file.close()

    print(f"[warmup] {len(targets)} cibles lambda* pretes "
          f"({n_from_cache} depuis le cache, {n_computed} calculees dans ce job).", flush=True)
    if not targets:
        return

    opt = torch.optim.Adam(model.parameters(), lr=args.warmup_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.n_warmup, eta_min=1e-5)
    for epoch in range(1, args.n_warmup + 1):
        random.shuffle(targets)
        total_loss = 0.0
        for graph, ri, lam_star_scaled in targets:
            opt.zero_grad()
            h_out = model(graph)
            h_out = h_out.view(graph.n_scenarios_eligible, ri.n_free).float()
            loss = torch.nn.functional.mse_loss(h_out, lam_star_scaled)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
            opt.step()
            total_loss += loss.item()
        sched.step()
        print(f"[warmup] epoch {epoch}/{args.n_warmup} loss={total_loss/len(targets):.4f}", flush=True)


# ═════════════════════════════════════════════════════════════════════════
# Boucle principale
# ═════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--node_suffix", default="_feasible",
                     help="Suffixe des fichiers nœuds à charger (défaut : '_feasible', "
                          "cf. curate_feasible_examples.py -- retombe sur nodes_n{n}_N{N}.txt "
                          "brut si le fichier suffixé n'existe pas)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--run_name", default="ccpo_gnn")
    ap.add_argument("--target_graphs", type=int, default=10000)
    ap.add_argument("--root_frac", type=float, default=0.05)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--n_epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_workers", type=int, default=1,
                     help="Threads paralleles pour les appels C++ evaluate_ccpo_lagrangian "
                          "(ctypes relache le GIL). "
                          "1 = sequentiel (comportement d'origine). Utile seulement sur un "
                          "noeud avec plusieurs coeurs REELLEMENT disponibles (SLURM dedie, "
                          "pas le noeud de connexion).")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eta_min", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--lr_warmup_ep", type=int, default=10)
    ap.add_argument("--max_norm", type=float, default=0.5)
    ap.add_argument("--clip_mode", choices=["per_example", "accumulated"], default="per_example",
                    help="per_example (defaut, = MKP) : ecretage du gradient de chaque exemple ; "
                         "accumulated : ancien comportement (ecretage du cumul apres chaque exemple), "
                         "conserve pour l'ablation (cf. diag_clip_effect.py)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--omp_threads", type=int, default=0,
                    help="threads OpenMP par évaluation du Lagrangien (boucle scénario) ; 0 = séquentiel. "
                         "Résultats identiques au bit près, cf. bench_train_device/MANIFEST.md")
    ap.add_argument("--device", default="cpu",
                    help="device du GNN (cpu|cuda) ; le Lagrangien C++ reste sur CPU (n_workers threads)")
    # ── Ablation ──
    ap.add_argument("--max_n", type=int, default=99, help="ne garde que les cellules n <= max_n")
    ap.add_argument("--init_ckpt", default=None,
                     help="poids initiaux communs (sortie de --warmup_only) -- saute le warmup")
    ap.add_argument("--warmup_only", action="store_true",
                     help="fait le warmup, sauve <out_dir>/warmup_init.pt puis s'arrete")
    ap.add_argument("--loss_mode", choices=["abs", "rel"], default="abs")
    ap.add_argument("--rel_scale", type=float, default=8.5)
    ap.add_argument("--hinge_beta", type=float, default=0.0)
    ap.add_argument("--feat_version", type=int, choices=[1, 2, 3], default=1,
                    help="1 = 10 features (modeles existants) ; 2 = 26 features (agregats facon CCMKP)")
    ap.add_argument("--edge_mode", choices=["none", "sigma", "mkp"], default=None,
                    help="poids d'aretes ; defaut : sigma si --edge_weight_sigma, sinon none")
    ap.add_argument("--center_lam", action="store_true",
                    help="centre la sortie sur les scenarios eligibles (option A, cf. centering_A)")
    ap.add_argument("--no_relu_h1", dest="relu_h1", action="store_false",
                    help="supprime la ReLU apres norm1 (comme CCMKP)")
    ap.add_argument("--edge_weight_sigma", action="store_true",
                     help="poids d'arete = Sigma_j[a][b]/sigma_scale (cf. ccpo_graph.build_edge_weights)")
    ap.add_argument("--exclude_nodes", default=None,
                    help="JSONL de find_infeasible_nodes.py : retire de la perte les noeuds au sous-arbre prouvé infaisable")
    ap.add_argument("--exclude_metric_only", action="store_true",
                    help="avec --exclude_nodes : ne retire rien de la perte, ajoute seulement la métrique VAL robuste (contrôle)")
    ap.add_argument("--extra_nodes_dir", default=None,
                    help="dossier de nodes_n*_N*_extra.txt (build_extra_feasible_nodes.py) ajoutés au TRAIN "
                         "après la sélection/split (VAL inchangée) ; indices placés après les nœuds de base de l'instance")
    ap.add_argument("--train_use_frac", type=float, default=1.0,
                     help="n'utilise que cette fraction des exemples d'ENTRAINEMENT (validation inchangee) : "
                          "courbe d'apprentissage vs volume de donnees (bras BH de l'ablation)")
    ap.add_argument("--stop_after_epoch", type=int, default=0,
                     help="s'arrete proprement apres cette epoch (checkpoint sauvegarde) ; 0 = jusqu'a n_epochs. "
                          "Permet un test court avec le MEME planning de lr qu'un run plus long")
    ap.add_argument("--metrics_every", type=int, default=1)
    ap.add_argument("--metrics_sample", type=int, default=300)

    ap.add_argument("--n_warmup", type=int, default=10)
    ap.add_argument("--warmup_lr", type=float, default=1e-3)
    ap.add_argument("--warmup_t", type=int, default=50)
    ap.add_argument("--warmup_max_n", type=int, default=20)
    ap.add_argument("--warmup_n_examples", type=int, default=None,
                     help="Sous-echantillonne le warmup a ce nombre d'exemples max "
                          "(defaut : tous les exemples eligibles --'6723 exemples x T=50 = plusieurs jours, trop lent')")
    ap.add_argument("--warmup_target_mode", choices=["legacy", "E0", "E0c", "auto"], default="legacy",
                     help="famille des cibles lambda* du warmup (cf. resolve_warmup_target_mode) ; "
                          "auto = E0c si --center_lam, sinon E0")
    ap.add_argument("--skip_warmup", action="store_true")

    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--eval_sample", type=int, default=400)
    ap.add_argument("--keep_ckpts", type=int, default=3)
    ap.add_argument("--keep_every", type=int, default=10)
    ap.add_argument("--intra_ckpt_every", type=int, default=5,
                     help="Sauvegarde intermediaire toutes les N accumulations logiques "
                          "(N*batch_size exemples) -- reprise en cours d'epoch'epoch trop longue pour ulimit -t du cluster'")
    ap.add_argument("--model_seed", type=int, default=None,
                    help="graine de l'init des poids (torch) et de l'ordre des exemples par epoch ; "
                         "la sélection, le split et la VAL restent fixés par --seed. Défaut = --seed (bit-identique).")
    args = ap.parse_args()
    if args.model_seed is None:
        args.model_seed = args.seed

    torch.manual_seed(args.model_seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    acquire_run_lock(out_dir, args.run_name)   # refuse de demarrer si un autre job actif utilise ce out_dir
    for stale in out_dir.glob("*.pt.tmp"):      # ecritures interrompues d'un job precedent
        stale.unlink(missing_ok=True)

    cells = load_all(Path(args.data_dir), node_suffix=args.node_suffix)
    cells = {k: v for k, v in cells.items() if k[0] <= args.max_n}
    print(f"[cells] {sorted(cells)}", flush=True)
    rng = random.Random(args.seed)
    selected = stratified_select(cells, args.target_graphs, args.root_frac, rng)
    train_examples, val_examples = split_by_instance(selected, args.train_frac, args.seed)
    print(f"[split] train={len(train_examples)} val={len(val_examples)}", flush=True)
    if args.train_use_frac < 1.0:
        _r = random.Random(args.seed + 99)
        _r.shuffle(train_examples)
        train_examples = train_examples[:max(1, int(len(train_examples) * args.train_use_frac))]
        print(f"[split] --train_use_frac {args.train_use_frac} : train reduit a {len(train_examples)} "
              f"(validation inchangee : {len(val_examples)})", flush=True)

    # --exclude_nodes : retire de la PERTE (train seulement) les nœuds dont le sous-arbre est prouvé
    # infaisable (statut "infeasible" de find_infeasible_nodes.py). La VAL reste IDENTIQUE (comparabilité entre runs) ;
    # la métrique robuste imp_mean_feas l'évalue sans ces nœuds.
    INFEAS_KEYS = None
    if args.exclude_nodes:
        INFEAS_KEYS = set()
        for line in open(args.exclude_nodes):
            r = json.loads(line)
            if r["status"] == "infeasible":
                INFEAS_KEYS.add((r["n"], r["N"], r["inst"], r["idx"]))
        n0 = len(train_examples)
        if not args.exclude_metric_only:
            train_examples = [e for e in train_examples if _ex_key(e) not in INFEAS_KEYS]
        n_val_inf = sum(1 for e in val_examples if _ex_key(e) in INFEAS_KEYS)
        print(f"[exclude] {len(INFEAS_KEYS)} noeuds prouvés infaisables ({args.exclude_nodes}) : train {n0} -> "
              f"{len(train_examples)}{' (métrique seulement, perte inchangée)' if args.exclude_metric_only else ''} ; "
              f"VAL inchangée ({n_val_inf} infaisables sur {len(val_examples)})", flush=True)

    # nœuds faisables supplémentaires (train seulement). Ajoutés APRÈS stratified_select/split, en fin de
    # liste par instance : sélection, VAL et clés (n, N, inst, idx) des nœuds de base inchangées.
    if args.extra_nodes_dir:
        val_inst = {(e.n, e.N, e.instance_id) for e in val_examples}
        n_extra = 0
        for (n, N), cell in cells.items():
            p = Path(args.extra_nodes_dir) / f"nodes_n{n}_N{N}_extra.txt"
            if not p.exists():
                continue
            for rec in load_nodes_ccpo(str(p)):
                if (n, N, rec.instance_id) in val_inst:
                    raise SystemExit(f"--extra_nodes_dir : nœud d'une instance VAL ({n}, {N}, {rec.instance_id})")
                if rec.instance_id not in cell.instances:
                    raise SystemExit(f"--extra_nodes_dir : instance inconnue ({n}, {N}, {rec.instance_id})")
                lst = cell.nodes_by_instance.setdefault(rec.instance_id, [])
                lst.append(rec)
                train_examples.append(Example(n, N, rec.instance_id, rec.depth, len(lst) - 1))
                n_extra += 1
        print(f"[extra] {n_extra} nœuds faisables ajoutés au train ({args.extra_nodes_dir}) : train -> "
              f"{len(train_examples)} ; VAL inchangée ({len(val_examples)})", flush=True)

    eval_rng = random.Random(args.seed + 1)
    eval_rng.shuffle(val_examples)
    eval_examples = val_examples[:args.eval_sample]

    args.edge_mode = resolve_edge_mode(vars(args))
    configure_graph_from_args(vars(args))
    global DEVICE, EVAL_OMP
    DEVICE = torch.device(args.device)
    EVAL_OMP = max(0, args.omp_threads)
    print(f"[eval] Lagrangien : {'OpenMP ' + str(EVAL_OMP) + ' threads/évaluation' if EVAL_OMP else 'séquentiel'}, "
          f"{args.n_workers} évaluations en parallèle", flush=True)
    if DEVICE.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda demandé mais aucun GPU visible")
    model = model_from_args(vars(args)).to(DEVICE)
    print(f"[device] {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""), flush=True)
    print(f"[model] feat_version={args.feat_version} edge_mode={args.edge_mode} "
          f"center_lam={args.center_lam} relu_h1={args.relu_h1} graph_cfg={GRAPH_CFG}", flush=True)

    start_epoch = 1
    history: dict[str, list] = {"loss_train": [], "loss_eval": [], "n_skipped_train": []}
    saved: list[Path] = []

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_warmup_ep > 0:
        n_decay = max(args.n_epochs - args.lr_warmup_ep, 1)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=args.eta_min / args.lr, end_factor=1.0,
                    total_iters=args.lr_warmup_ep),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=n_decay, eta_min=args.eta_min),
            ],
            milestones=[args.lr_warmup_ep],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.n_epochs, eta_min=args.eta_min)

    latest, ckpt = load_latest_valid_checkpoint(out_dir, args.run_name)
    intra = load_intra_epoch_checkpoint(out_dir, args.run_name)
    # la rotation (keep_ckpts) doit continuer a purger les anciens checkpoints apres reprise
    saved = [pth for pth in list_checkpoints(out_dir, args.run_name)
             if int(pth.stem.split("_ep")[-1]) % args.keep_every != 0]
    if latest is not None:
        print(f"[resume] reprise depuis {latest}", flush=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        history = ckpt["history"]
        start_epoch = ckpt["epoch"] + 1
        if "scheduler" in ckpt:
            # restauration EXACTE de l'etat du scheduler. Avant,
            # on le re-parcourait a la main (scheduler.step() x (start_epoch-1)) APRES
            # avoir charge l'optimiseur : avec LinearLR/SequentialLR (formules recursives
            # sur group['lr']) le lr de l'epoch rejouee pouvait etre faux.
            scheduler.load_state_dict(ckpt["scheduler"])
        else:
            for _ in range(start_epoch - 1):
                scheduler.step()
        print(f"[resume] lr de l'epoch {start_epoch} : {optimizer.param_groups[0]['lr']:.3e}", flush=True)
    elif args.init_ckpt:
        print(f"[init] poids initiaux depuis {args.init_ckpt} (warmup saute)", flush=True)
        model.load_state_dict(torch.load(args.init_ckpt, map_location="cpu"))
    elif not args.skip_warmup:
        warmup_on_roots(model, cells, train_examples, args, out_dir, args.run_name)
        if args.warmup_only:
            torch.save(model.state_dict(), out_dir / "warmup_init.pt")
            print(f"[warmup_only] poids sauvegardes dans {out_dir / 'warmup_init.pt'}", flush=True)
            return

    # Reprise EN COURS D'EPOCH (cf. save_intra_epoch_checkpoint) : ne
    # s'applique que si elle correspond a l'epoch qu'on est sur le point de
    # commencer (sinon obsolete -- ex. checkpoint complet d'epoch plus
    # recent deja charge ci-dessus, on l'ignore).
    if intra is not None and intra["epoch"] == start_epoch:
        print(f"[resume] reprise en cours d'epoch {start_epoch} a l'exemple "
              f"{intra['example_idx']}/{len(train_examples)}", flush=True)
        model.load_state_dict(intra["model"])
        optimizer.load_state_dict(intra["optimizer"])
    else:
        intra = None

    # L(lambda=0) de reference (cache JSONL) pour les exemples d'entrainement et
    # d'evaluation -- sert a la metrique d'amelioration relative ET a --loss_mode rel.
    # tire depuis val_examples (deja melange) et NON depuis eval_examples,
    # plafonne par --eval_sample (60 dans l'ablation -> metrique sur 60 noeuds au lieu de 300)
    metrics_examples = val_examples[:args.metrics_sample]
    with ThreadPoolExecutor(max_workers=max(1, args.n_workers)) as ex0:
        l0_map = compute_L0_map(cells, train_examples + metrics_examples, ex0,
                                out_dir / f"L0_cache_{args.run_name}.jsonl")

    for epoch in range(start_epoch, args.n_epochs + 1):
        model.train()
        # Ordre DETERMINISTE (seed derivee de epoch) -- indispensable pour
        # qu'une reprise en cours d'epoch puisse sauter exactement le meme
        # prefixe d'exemples deja traites ().
        epoch_examples = list(train_examples)
        random.Random(args.model_seed * 1000 + epoch).shuffle(epoch_examples)

        ex_restore = {}
        if intra is not None and intra["epoch"] == epoch:
            start_idx = intra["example_idx"]
            accum_loss = intra["accum_loss"]
            accum_count = intra["accum_count"]
            n_skipped = intra["n_skipped"]
            ex_restore = intra.get("extra", {})
        else:
            start_idx = 0
            accum_loss = 0.0
            accum_count = 0
            n_skipped = 0
        intra = None  # ne s'applique qu'a la toute premiere epoch de cette execution
        # statistiques d'epoch (normes de gradient avant ecretage, amelioration vs L0) :
        # restaurees apres une reprise en milieu d'epoch pour que le log reste exact
        gn_sum_ep = ex_restore.get("gn_sum_ep", 0.0); gn_max_ep = ex_restore.get("gn_max_ep", 0.0)
        n_clip_ep = ex_restore.get("n_clip_ep", 0)
        imp_sum_ep = ex_restore.get("imp_sum_ep", 0.0); n_worse_ep = ex_restore.get("n_worse_ep", 0)
        n_imp_ep = ex_restore.get("n_imp_ep", 0)
        optimizer.zero_grad()
        t_ep0 = time.perf_counter()
        t_fwd_ep = t_wait_ep = t_bwd_ep = 0.0

        # Micro-batchs de taille batch_size : forward+backward sequentiels,
        # evaluations C++ (le vrai cout) EN PARALLELE sur n_workers threads
        # (GIL relache pendant l'appel ctypes). n_workers=1 -> comportement sequentiel d'origine.
        with ThreadPoolExecutor(max_workers=max(1, args.n_workers)) as executor:
            n_chunks_done = 0
            for idx in range(start_idx, len(epoch_examples), args.batch_size):
                chunk = epoch_examples[idx: idx + args.batch_size]
                st = process_chunk_parallel(
                    model, cells, chunk, executor, args.max_norm, args.clip_mode,
                    l0_map, args.loss_mode, args.rel_scale, args.hinge_beta)
                count = st["count"]
                accum_loss += st["loss_sum"]
                accum_count += count
                n_skipped += st["skipped"]
                gn_sum_ep += st["gn_sum"]; gn_max_ep = max(gn_max_ep, st["gn_max"]); n_clip_ep += st["n_clipped"]
                imp_sum_ep += st["imp_sum"]; n_worse_ep += st["n_worse"]; n_imp_ep += st["n_imp"]
                t_fwd_ep += st["t_fwd"]; t_wait_ep += st["t_wait"]; t_bwd_ep += st["t_bwd"]
                if count > 0:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad /= count
                    optimizer.step()
                optimizer.zero_grad()

                n_chunks_done += 1
                if n_chunks_done % args.intra_ckpt_every == 0:
                    chunk_end = idx + len(chunk)
                    save_intra_epoch_checkpoint(
                        out_dir, args.run_name, epoch, chunk_end, accum_loss, accum_count, n_skipped,
                        model, optimizer,
                        extra=dict(gn_sum_ep=gn_sum_ep, gn_max_ep=gn_max_ep, n_clip_ep=n_clip_ep,
                                   imp_sum_ep=imp_sum_ep, n_worse_ep=n_worse_ep, n_imp_ep=n_imp_ep))
                    print(f"  [epoch {epoch}] checkpoint intermediaire a l'exemple "
                          f"{chunk_end}/{len(epoch_examples)}", flush=True)

        # (plus de "reliquat de batch" a gerer ici : chaque chunk ci-dessus
        # fait deja son propre optimizer.step(), cf. boucle parallele)

        # Epoch reellement terminee : le checkpoint intra-epoch n'a plus lieu
        # d'etre (le checkpoint complet ci-dessous prend le relais).
        intra_path = _intra_epoch_ckpt_path(out_dir, args.run_name)
        intra_path.unlink(missing_ok=True)

        print(f"[epoch {epoch}] temps train={time.perf_counter() - t_ep0:.1f}s "
              f"(gnn_fwd={t_fwd_ep:.1f}s attente_C++={t_wait_ep:.1f}s gnn_bwd+clip={t_bwd_ep:.1f}s, "
              f"device={DEVICE}, n_workers={args.n_workers})", flush=True)
        scheduler.step()
        mean_loss = accum_loss / max(1, accum_count)
        history["loss_train"].append(mean_loss)
        history["n_skipped_train"].append(n_skipped)
        print(f"[epoch {epoch}/{args.n_epochs}] loss_train={mean_loss:.4f} "
              f"skipped={n_skipped} lr={scheduler.get_last_lr()[0]:.2e} "
              f"gn_mean={gn_sum_ep / max(1, accum_count):.3g} gn_max={gn_max_ep:.3g} "
              f"frac_clipped={n_clip_ep / max(1, accum_count):.2f} clip_mode={args.clip_mode} "
              f"loss_mode={args.loss_mode} | TRAIN imp_vs_L0={100 * imp_sum_ep / max(1, n_imp_ep):.3f}% "
              f"frac_worse={n_worse_ep / max(1, n_imp_ep):.2f}", flush=True)

        if args.metrics_every > 0 and (epoch % args.metrics_every == 0 or epoch == args.n_epochs):
            with ThreadPoolExecutor(max_workers=max(1, args.n_workers)) as exm:
                mt = eval_metrics(model, cells, metrics_examples, l0_map, exm, INFEAS_KEYS)
            if mt.get("n"):
                parts_ = " ".join(f"[nf {k.split('_')[1]}: {100 * v[0]:.2f}% worse={v[1]:.2f} n={v[2]}]"
                                  for k, v in mt.items() if k.startswith("imp_") and isinstance(v, tuple))
                print(f"[epoch {epoch}] VAL imp_vs_L0 mean={100 * mt['imp_mean']:.3f}% med={100 * mt['imp_med']:.3f}% "
                      f"frac_worse={mt['frac_worse']:.2f} (n={mt['n']}) {parts_}"
                      + (f" | FAISABLES mean={100 * mt['imp_mean_feas']:.3f}% med={100 * mt['imp_med_feas']:.3f}% "
                         f"(n={mt['n_feas']})" if "imp_mean_feas" in mt else ""), flush=True)
                history.setdefault("val_metrics", []).append({"epoch": epoch, **{k: v for k, v in mt.items()}})

        if epoch % args.eval_every == 0 or epoch == args.n_epochs:
            model.eval()
            with torch.no_grad():
                pass  # build_lambda_tensor a besoin d'autograd pour la reconstruction
            eval_losses = []
            for ex in eval_examples:
                loss = eval_example_loss(model, cells, ex)
                if loss is not None:
                    eval_losses.append(loss.item())
            mean_eval = float(np.mean(eval_losses)) if eval_losses else float("nan")
            history["loss_eval"].append({"epoch": epoch, "loss": mean_eval, "n": len(eval_losses)})
            print(f"[epoch {epoch}] loss_eval={mean_eval:.4f} (n={len(eval_losses)})", flush=True)
            with open(out_dir / f"metrics_{args.run_name}.json", "w") as f:
                json.dump(history, f, indent=2)

        save_checkpoint(out_dir, args.run_name, epoch, model, optimizer, scheduler,
                         history, args, saved, args.keep_ckpts, args.keep_every)

        if args.stop_after_epoch and epoch >= args.stop_after_epoch:
            print(f"Arret demande apres l'epoch {epoch} (--stop_after_epoch {args.stop_after_epoch}).", flush=True)
            print("Entraînement termine.", flush=True)
            if _LOCK_RELEASE is not None:
                _LOCK_RELEASE()
            os._exit(0)

    print("Entraînement termine.", flush=True)
    if _LOCK_RELEASE is not None:
        _LOCK_RELEASE()   # os._exit saute atexit : liberer le verrou explicitement
    os._exit(0)


if __name__ == "__main__":
    main()
