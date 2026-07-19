"""
train_scaled_v1cld.py — Entraînement GNN V1CLD (SSP) sur plusieurs instances,
plusieurs tailles (E,T,N), stratifié par depth. Port SSP de
gnn/train_scaled_v1ld.py (MKP) — mêmes principes de sélection de données
(reservoir sampling par instance, budget égal par combo, plafond racine),
adaptés à 3 axes de taille (E,T,N) au lieu de 2 (n,N).

Différence clé avec MKP : E, T, N sont des propriétés de l'INSTANCE, fixes à
tous les nœuds du B&B pour cette instance — le B&B ne réduit ni E ni T, il
réduit G (paires (e,t) libres) et N_z (scénarios libres) au sein du même
E*T/N nominal. La combinaison (E,T,N) identifie donc le "combo" de taille
au niveau instance, pas au niveau nœud.

Warmup sur racines : pas de plafond warmup_max_n (contrairement à MKP) car
ζ_v1CLD est un glouton O(E·T·Q) — générer λ* via sous-gradient est bon marché
même sur les plus grosses instances testées ici.
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib
DataLoader = importlib.import_module("torch_geometric.loader").DataLoader

from train_v1CLD_ssp import (
    GNN, lagrangian_v1cld_ssp, verify_v1cld_ssp, load_instances, build_graph,
    LazyGraphDataset, EagerGraphDataset, _apply_center_ssp,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Entraîne GNN V1CLD (SSP) sur plusieurs combos (E,T,N), stratifié par depth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Données ───────────────────────────────────────────────────────────────
    p.add_argument("--data_dir", default=None,
                   help="Racine contenant train_ssp_base_s10/. Défaut: <repo>/data/")
    p.add_argument("--out_dir",  default=None,
                   help="Dossier de sortie. Défaut: <repo>/gnn_ssp/runs/")
    p.add_argument("--combos", type=str, nargs="+",
                   default=["E3_T12_N20", "E3_T12_N50", "E4_T12_N20",
                            "E4_T12_N50", "E4_T8_N20", "E4_T8_N50"],
                   help="Combos E{E}_T{T}_N{N} à charger depuis train_ssp_base_s10/.")
    p.add_argument("--target_graphs", type=int, default=10_000,
                   help="Nombre total de graphes dans le dataset sélectionné.")
    p.add_argument("--root_frac", type=float, default=0.05,
                   help="Fraction du budget par combo réservée aux racines (depth=0).")
    p.add_argument("--reservoir_k", type=int, default=50,
                   help="Nb max de nœuds non-racine gardés PAR INSTANCE (reservoir sampling).")
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)

    # ── Modèle ────────────────────────────────────────────────────────────────
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--use_max_pool", action="store_true")

    # ── Entraînement ──────────────────────────────────────────────────────────
    p.add_argument("--n_epochs",   type=int,   default=100)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--eta_min",    type=float, default=1e-5)
    p.add_argument("--no_scheduler", action="store_true")
    p.add_argument("--max_norm",   type=float, default=0.5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--center_lam", action="store_true",
                   help="Centrage des λ sur (scénario,période) après la sortie du GNN.")

    # ── Warmup supervisé (racines train) ──────────────────────────────────────
    p.add_argument("--n_warmup",  type=int,   default=10,
                   help="Epochs de warmup supervisé L2 sur λ* sous-gradient. 0 = désactivé.")
    p.add_argument("--warmup_lr", type=float, default=1e-3)
    p.add_argument("--warmup_T",  type=int,   default=50,
                   help="Itérations sous-gradient (Polyak) pour générer λ* cibles.")

    # ── Évaluation & infra ─────────────────────────────────────────────────────
    p.add_argument("--eval_sample", type=int, default=200,
                   help="Nombre max de graphes test pour l'éval.")
    p.add_argument("--eval_mixed", action="store_true",
                   help="Eval stratifié par depth (même méthode que train) plutôt que racines seules.")
    p.add_argument("--eval_sg_T", type=int, default=50,
                   help="Itérations sous-gradient (Polyak) pour la baseline eval.")
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--keep_ckpts", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--run_tag", type=str, default="")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Découverte + chargement (reservoir sampling par instance)
# ─────────────────────────────────────────────────────────────────────────────

def discover_pairs(data_root: Path, combos: list[str]) -> list[tuple[Path, Path, int, int, int]]:
    """Retourne [(inst_file, nodes_file, E, T, N)] pour chaque combo demandé,
    trouvé dans train_ssp_base_s10/."""
    pairs = []
    d = data_root / "train_ssp_base_s10"
    for combo in combos:
        inst_f  = d / f"instances_{combo}.txt"
        nodes_f = d / f"nodes_{combo}.txt"
        if inst_f.exists() and nodes_f.exists():
            # combo = "E{E}_T{T}_N{N}"
            parts = combo.split("_")
            E = int(parts[0][1:]); T = int(parts[1][1:]); N = int(parts[2][1:])
            pairs.append((inst_f, nodes_f, E, T, N))
        else:
            print(f"  WARN: absent — {combo} (cherché dans {d})")
    return pairs


def _parse_full_node_record(line: str, inst: dict) -> dict | None:
    """Parse complet d'une ligne nodes_*.txt (Format C : 3+E*T+N champs),
    logique dupliquée de load_raw_nodes (train_v1CLD_ssp.py) pour ne pas
    toucher ce module partagé."""
    vals = [float(x) for x in line.split(",")]
    inst_id   = int(vals[0])
    depth     = int(vals[1])
    curr_cost = vals[2]
    E, T, N = inst["E"], inst["T"], inst["N"]
    n_fmt_C = 3 + E * T + N
    if len(vals) != n_fmt_C:
        return None

    x_mask = np.array(vals[3 : 3 + E * T], dtype=np.int8)
    z_mask = np.array(vals[3 + E * T : 3 + E * T + N], dtype=np.int8)

    if (x_mask == 4).sum() == 0 or (z_mask == 2).sum() == 0:
        return None

    return {"inst_id": inst_id, "depth": depth, "curr_cost": curr_cost,
            "x_mask": x_mask, "z_mask": z_mask}


def stream_reservoir_by_instance(
    nodes_file: Path, instances: dict, reservoir_k: int, rng: random.Random,
) -> tuple[list, list]:
    """Un seul passage sur nodes_file : racines (depth=0) toutes gardées,
    non-racines plafonnées à reservoir_k par instance (reservoir sampling
    classique, algo R) — cf. gnn/train_scaled_v1ld.py::stream_reservoir_by_instance
    pour la justification détaillée (chaque instance a une chance égale
    d'être représentée, indépendamment de sa position dans le fichier)."""
    roots: list = []
    reservoirs: dict[int, list] = {}
    seen_non_root: dict[int, int] = defaultdict(int)
    skipped = 0

    with open(nodes_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c1 = line.index(",")
            c2 = line.index(",", c1 + 1)
            inst_id = int(line[:c1])
            depth   = int(line[c1 + 1 : c2])

            inst = instances.get(inst_id)
            if inst is None:
                skipped += 1
                continue

            if depth == 0:
                rec = _parse_full_node_record(line, inst)
                if rec is not None:
                    roots.append(rec)
                else:
                    skipped += 1
                continue

            seen_non_root[inst_id] += 1
            c   = seen_non_root[inst_id]
            cur = reservoirs.get(inst_id)
            if cur is None or len(cur) < reservoir_k:
                rec = _parse_full_node_record(line, inst)
                if rec is not None:
                    reservoirs.setdefault(inst_id, []).append(rec)
            else:
                j = rng.randint(1, c)
                if j <= reservoir_k:
                    rec = _parse_full_node_record(line, inst)
                    if rec is not None:
                        reservoirs[inst_id][j - 1] = rec

    non_root = [r for lst in reservoirs.values() for r in lst]
    return roots, non_root


def load_and_merge(
    pairs: list[tuple[Path, Path, int, int, int]], reservoir_k: int, seed: int,
) -> tuple[dict, list]:
    """Charge tous les (inst_file, nodes_file) par reservoir sampling,
    renumérote les inst_ids pour éviter les collisions entre combos, annote
    chaque record avec E_combo/T_combo/N_combo (métadonnées d'instance —
    cf. docstring module : E,T,N ne varient PAS avec la profondeur B&B)."""
    rng = random.Random(seed)
    merged_instances: dict = {}
    merged_records:   list = []

    for inst_f, nodes_f, E_val, T_val, N_val in pairs:
        print(f"  E={E_val} T={T_val} N={N_val}: chargement (reservoir k={reservoir_k}) ...",
              end=" ", flush=True)
        instances = load_instances(inst_f)
        roots, non_root = stream_reservoir_by_instance(nodes_f, instances, reservoir_k, rng)
        records = roots + non_root

        offset = max(merged_instances.keys(), default=-1) + 1
        id_map = {old: old + offset for old in instances}
        for old_id, inst in instances.items():
            merged_instances[id_map[old_id]] = inst
        for rec in records:
            nr = dict(rec)
            nr["inst_id"] = id_map[rec["inst_id"]]
            nr["E_combo"] = E_val
            nr["T_combo"] = T_val
            nr["N_combo"] = N_val
            merged_records.append(nr)

        depth_cnt = defaultdict(int)
        for r in records:
            depth_cnt[r["depth"]] += 1
        n_inst_seen = len({r["inst_id"] for r in records})
        print(f"{len(roots)} racines + {len(non_root)} non-racines "
              f"({n_inst_seen} instances vues sur {len(instances)} dispo)  "
              f"depths:{dict(sorted(depth_cnt.items()))}")

    return merged_instances, merged_records


# ─────────────────────────────────────────────────────────────────────────────
# Sélection stratifiée (budget égal par combo, plafond racine, round-robin par instance)
# ─────────────────────────────────────────────────────────────────────────────

def select_stratified(
    all_records: list, target_graphs: int, root_frac: float, seed: int,
) -> list:
    """Port direct de gnn/train_scaled_v1ld.py::select_stratified, clé de
    combo = (E_combo, T_combo, N_combo) au lieu de (n_orig, N_orig)."""
    rng = random.Random(seed)

    by_combo: dict = defaultdict(list)
    for r in all_records:
        by_combo[(r.get("E_combo", 0), r.get("T_combo", 0), r.get("N_combo", 0))].append(r)

    combos = sorted(by_combo.keys())
    n_combos = len(combos) or 1
    per_combo_budget = target_graphs // n_combos

    selected: list = []
    for combo in combos:
        recs      = by_combo[combo]
        roots     = [r for r in recs if r["depth"] == 0]
        non_roots = [r for r in recs if r["depth"] >  0]
        rng.shuffle(roots)

        root_budget    = min(len(roots), round(per_combo_budget * root_frac))
        nonroot_budget = max(0, per_combo_budget - root_budget)

        chosen_roots  = roots[:root_budget]
        used_inst_ids = {r["inst_id"] for r in chosen_roots}

        by_inst: dict = defaultdict(list)
        for r in non_roots:
            by_inst[r["inst_id"]].append(r)
        for lst in by_inst.values():
            rng.shuffle(lst)

        inst_ids = list(by_inst.keys())
        rng.shuffle(inst_ids)
        inst_ids.sort(key=lambda i: i in used_inst_ids)

        idx_per_inst: dict = {i: 0 for i in inst_ids}
        chosen_nonroot: list = []
        while len(chosen_nonroot) < nonroot_budget:
            progress = False
            for i in inst_ids:
                lst = by_inst[i]
                pos = idx_per_inst[i]
                if pos < len(lst) and len(chosen_nonroot) < nonroot_budget:
                    chosen_nonroot.append(lst[pos])
                    idx_per_inst[i] = pos + 1
                    progress = True
            if not progress:
                break

        selected.extend(chosen_roots)
        selected.extend(chosen_nonroot)

        combo_inst_ids = used_inst_ids | {r["inst_id"] for r in chosen_nonroot}
        print(f"  combo E={combo[0]:<2} T={combo[1]:<3} N={combo[2]:<4} budget={per_combo_budget:6d}  "
              f"racines={len(chosen_roots):5d}/{len(roots):<6d}  "
              f"non-racines={len(chosen_nonroot):6d}/{len(non_roots):<8d}  "
              f"instances_distinctes={len(combo_inst_ids)}")

    by_depth: dict = defaultdict(int)
    by_combo_cnt: dict = defaultdict(int)
    all_inst_ids: set = set()
    for r in selected:
        by_depth[r["depth"]] += 1
        by_combo_cnt[(r.get("E_combo", 0), r.get("T_combo", 0), r.get("N_combo", 0))] += 1
        all_inst_ids.add(r["inst_id"])
    print(f"[select] {len(selected)} graphes au total, {len(all_inst_ids)} instances distinctes")
    print(f"  par depth : {dict(sorted(by_depth.items()))}")
    print(f"  par combo : {dict(sorted(by_combo_cnt.items()))}")
    return selected


def split_by_instance(
    instances: dict, all_ds: LazyGraphDataset, train_frac: float, seed: int,
) -> tuple[LazyGraphDataset, LazyGraphDataset]:
    inst_ids = np.array(sorted(instances.keys()))
    rng = np.random.default_rng(seed)
    rng.shuffle(inst_ids)
    n_tr = int(len(inst_ids) * train_frac)
    train_ids = set(inst_ids[:n_tr].tolist())
    test_ids  = set(inst_ids[n_tr:].tolist())
    return all_ds.filter_by_inst_ids(train_ids), all_ds.filter_by_inst_ids(test_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Warmup supervisé (racines train, λ* sous-gradient Polyak)
# ─────────────────────────────────────────────────────────────────────────────

def warmup_on_roots(
    model: GNN, train_ds: LazyGraphDataset, instances: dict,
    args: argparse.Namespace, device: torch.device,
) -> None:
    """Warmup L2 sur λ* sous-gradient (Polyak, v_lower=-opt), sur TOUTES les
    racines train — pas de plafond de taille (contrairement à MKP) : ζ_v1CLD
    est un glouton bon marché, générer λ* reste rapide même sur les plus
    grosses instances testées ici."""
    root_recs = [r for r in train_ds.raw_records if r["depth"] == 0]
    if not root_recs:
        print("[warmup] Aucun graphe éligible — skip")
        return

    root_ds = LazyGraphDataset(train_ds.instances, root_recs)
    print(f"[warmup] Construction EagerGraphDataset ({len(root_ds)} graphes) ...", flush=True)
    eager = EagerGraphDataset(root_ds)
    graphs = [eager[i] for i in range(len(eager))]
    n_wm = len(graphs)

    print(f"[warmup] Génération cibles sous-gradient (T={args.warmup_T}, Polyak) ...", flush=True)
    t0 = time.perf_counter()
    lam_targets = []
    for g in graphs:
        inst = instances[int(g.inst_id.item())]
        _, lam_star = verify_v1cld_ssp(g, inst, T=args.warmup_T,
                                        v_lower=-float(inst["opt"]), return_lam=True)
        lam_targets.append(
            torch.tensor(lam_star.astype(np.float32).flatten(),
                         dtype=torch.float32, device=device))
    for g in graphs:
        g.to(device)
    print(f"[warmup] Cibles prêtes en {time.perf_counter()-t0:.1f}s  "
          f"|λ*|_mean={np.mean([t.abs().mean().item() for t in lam_targets]):.4f}")

    opt_wm   = torch.optim.Adam(model.parameters(), lr=args.warmup_lr)
    sched_wm = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_wm, T_max=max(args.n_warmup, 1), eta_min=1e-5)
    idx_list = list(range(n_wm))

    print(f"[warmup] {args.n_warmup} epochs (lr={args.warmup_lr}) ...", flush=True)
    t0 = time.perf_counter()
    for ep in range(1, args.n_warmup + 1):
        random.shuffle(idx_list)
        mse_sum = 0.0
        for gi in idx_list:
            g = graphs[gi]
            opt_wm.zero_grad()
            lam_pred = model(g)
            loss_wm  = ((lam_pred.flatten() - lam_targets[gi]) ** 2).mean()
            loss_wm.backward()
            if args.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
            opt_wm.step()
            mse_sum += loss_wm.item()
        sched_wm.step()
        if ep % max(1, args.n_warmup // 10) == 0 or ep == args.n_warmup:
            model.eval()
            with torch.no_grad():
                l1_vals = []
                for g in graphs:
                    inst = instances[int(g.inst_id.item())]
                    c_scale = max(float(g.c_scale), 1.0)
                    l1_vals.append(lagrangian_v1cld_ssp(model(g), g, inst).item() / c_scale)
            model.train()
            print(f"[warmup] ep {ep:4d}/{args.n_warmup}  mse={mse_sum/n_wm:.6f}  "
                  f"L1_norm={float(np.mean(l1_vals)):.4f}  "
                  f"t={time.perf_counter()-t0:.0f}s", flush=True)
    print(f"[warmup] Terminé en {time.perf_counter()-t0:.0f}s")


# ─────────────────────────────────────────────────────────────────────────────
# Évaluation
# ─────────────────────────────────────────────────────────────────────────────

def _zeros_lam(g) -> torch.Tensor:
    return torch.zeros(g.x.shape[0], 4, dtype=torch.float32, device=g.x.device)


def compute_l1_zero(graphs: list, instances: dict) -> np.ndarray:
    with torch.no_grad():
        return np.array([
            lagrangian_v1cld_ssp(_zeros_lam(g), g, instances[int(g.inst_id.item())]).item()
            for g in graphs
        ])


def relative_gap_pct(l1_val: float, opt_val: float) -> float:
    """Écart relatif à l'optimum connu, en % — même convention de signe que
    diag_overfit_single_ssp.py::gap_opt (cf. bench_ssp.cpp:266-268) :
    cost_lower = -L1(λ), gap = (cost_lower - opt)/opt*100 ; ≤0 = borne valide,
    0% = optimum atteint."""
    if not opt_val:
        return float("nan")
    cost_lower = -l1_val
    return (cost_lower - opt_val) / opt_val * 100.0


def eval_epoch(graphs, l1_baseline, instances, model, center_lam=False):
    """Retourne (l1_mean, gap_mean, pct_better) sur les graphes eval.
    gap_mean = écart relatif moyen à l'optimum (%, cf. relative_gap_pct) ;
    pct_better = fraction où le GNN bat la baseline sous-gradient (l1_baseline)."""
    model.eval()
    l1_vals, gap_vals, n_better = [], [], 0
    with torch.no_grad():
        for g, l0 in zip(graphs, l1_baseline):
            inst = instances[int(g.inst_id.item())]
            lam  = model(g)
            if center_lam:
                lam = _apply_center_ssp(lam, int(g.N_z), int(g.T), int(g.n_g))
            l1_gnn = lagrangian_v1cld_ssp(lam, g, inst).item()
            l1_vals.append(l1_gnn)
            gap_vals.append(relative_gap_pct(l1_gnn, float(g.opt.item())))
            if l1_gnn < l0:
                n_better += 1
    model.train()
    return float(np.mean(l1_vals)), float(np.mean(gap_vals)), 100.0 * n_better / max(len(graphs), 1)


def eval_loss_epoch(graphs, instances, model, center_lam=False):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for g in graphs:
            inst = instances[int(g.inst_id.item())]
            lam  = model(g)
            if center_lam:
                lam = _apply_center_ssp(lam, int(g.N_z), int(g.T), int(g.n_g))
            c_scale = max(float(g.c_scale), 1.0)
            total += lagrangian_v1cld_ssp(lam, g, inst).item() / c_scale
    model.train()
    return total / max(len(graphs), 1)


def eval_by_combo(graphs, l1_baseline, instances, model, center_lam=False):
    """L1, gap_vs_opt et pct_better par combo (E,T,N) — E/T/N sont des
    attributs directs du graphe (fixes pour toute l'instance, cf. docstring
    module)."""
    model.eval()
    by_combo: dict = {}
    with torch.no_grad():
        for g, l0 in zip(graphs, l1_baseline):
            combo_key = (int(g.E), int(g.T), int(g.N))
            inst = instances[int(g.inst_id.item())]
            lam  = model(g)
            if center_lam:
                lam = _apply_center_ssp(lam, int(g.N_z), int(g.T), int(g.n_g))
            l1_gnn = lagrangian_v1cld_ssp(lam, g, inst).item()
            by_combo.setdefault(combo_key, {"l1": [], "gap": [], "better": []})
            by_combo[combo_key]["l1"].append(l1_gnn)
            by_combo[combo_key]["gap"].append(relative_gap_pct(l1_gnn, float(g.opt.item())))
            by_combo[combo_key]["better"].append(int(l1_gnn < l0))
    model.train()
    return {c: {"l1_mean": float(np.mean(v["l1"])),
                "gap_mean": float(np.mean(v["gap"])),
                "pct_better": float(100.0 * np.mean(v["better"])),
                "count": len(v["l1"])}
            for c, v in sorted(by_combo.items())}


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def ckpt_path(out_dir: Path, run_name: str, epoch: int) -> Path:
    return out_dir / f"ckpt_{run_name}_ep{epoch:04d}.pt"


def save_checkpoint(out_dir, run_name, epoch, model, optimizer, scheduler,
                    history, args, saved, keep, keep_every=10):
    path = ckpt_path(out_dir, run_name, epoch)
    torch.save({"epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "history": history, "args": vars(args)}, path)
    if epoch % keep_every == 0:
        return
    saved.append(path)
    if len(saved) > keep:
        saved.pop(0).unlink(missing_ok=True)


def find_latest_checkpoint(out_dir: Path, run_name: str) -> Path | None:
    ckpts = sorted(out_dir.glob(f"ckpt_{run_name}_ep*.pt"))
    return ckpts[-1] if ckpts else None


def save_history(path: Path, history: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(history, f, indent=2)
    tmp.replace(path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_root = Path(args.data_dir) if args.data_dir else REPO_ROOT / "data"
    out_dir   = Path(args.out_dir)  if args.out_dir  else Path(__file__).resolve().parent / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    # ── 1. Découverte + chargement ─────────────────────────────────────────────
    log("Découverte des données ...")
    pairs = discover_pairs(data_root, args.combos)
    if not pairs:
        log("ERREUR : aucun fichier trouvé — vérifiez --data_dir et --combos")
        sys.exit(1)
    for inst_f, nodes_f, E, T, N in pairs:
        log(f"  E={E} T={T} N={N}  {nodes_f.name}")

    log("Chargement ...")
    instances, all_records = load_and_merge(pairs, args.reservoir_k, args.seed)
    log(f"  {len(instances)} instances  {len(all_records)} records bruts")

    # ── 2. Sélection stratifiée + split ────────────────────────────────────────
    selected = select_stratified(all_records, args.target_graphs, args.root_frac, args.seed)
    all_ds   = LazyGraphDataset(instances, selected)
    train_ds, test_ds = split_by_instance(instances, all_ds, args.train_frac, args.seed)
    log(f"split {args.train_frac:.0%}/{1-args.train_frac:.0%} — "
        f"train={len(train_ds)}  test={len(test_ds)}")

    for label, ds in [("train", train_ds), ("test", test_ds)]:
        by_d = defaultdict(int)
        for r in ds.raw_records:
            by_d[r["depth"]] += 1
        log(f"  {label} par depth: {dict(sorted(by_d.items()))}")

    # ── 3. Ensemble d'évaluation ────────────────────────────────────────────────
    if args.eval_mixed:
        eval_recs = select_stratified(test_ds.raw_records, args.eval_sample, args.root_frac, args.seed)
        eval_desc = f"{len(eval_recs)} graphes test (même méthode que train)"
    else:
        root_test_recs = [r for r in test_ds.raw_records if r["depth"] == 0]
        n_eval = min(args.eval_sample, len(root_test_recs))
        random.Random(args.seed).shuffle(root_test_recs)
        eval_recs = root_test_recs[:n_eval]
        eval_desc = f"{len(eval_recs)} racines test"
    eval_lazy = LazyGraphDataset(instances, eval_recs)
    log(f"Construction eval set ({eval_desc}) ...")
    t0 = time.time()
    eager_eval  = EagerGraphDataset(eval_lazy)
    eval_graphs = [eager_eval[i].to(device) for i in range(len(eager_eval))]
    log(f"  fait en {time.time()-t0:.1f}s")

    log("Précalcul L1(λ=0) ...")
    l1_zeros = compute_l1_zero(eval_graphs, instances)
    log(f"  mean L1(0) = {np.mean(l1_zeros):.2f}")

    log(f"Précalcul baseline sous-gradient eval (T={args.eval_sg_T}, Polyak, {len(eval_graphs)} graphes) ...")
    t0_sg = time.time()
    l1_sg_eval = np.array([
        verify_v1cld_ssp(g.cpu(), instances[int(g.inst_id.item())],
                          T=args.eval_sg_T, v_lower=-float(instances[int(g.inst_id.item())]["opt"]))
        for g in eval_graphs
    ])
    for g in eval_graphs:
        g.to(device)
    log(f"  fait en {time.time()-t0_sg:.1f}s  mean L1_sg(T={args.eval_sg_T}) = {np.mean(l1_sg_eval):.2f}")

    gap_sg_eval = np.array([relative_gap_pct(l1, float(g.opt.item()))
                             for l1, g in zip(l1_sg_eval, eval_graphs)])
    gap_sg_by_combo_raw: dict = defaultdict(list)
    for g, gap in zip(eval_graphs, gap_sg_eval):
        gap_sg_by_combo_raw[(int(g.E), int(g.T), int(g.N))].append(gap)
    gap_sg_by_combo = {c: float(np.mean(v)) for c, v in sorted(gap_sg_by_combo_raw.items())}
    log(f"  mean gap_vs_opt sous-grad (T={args.eval_sg_T}) = {np.mean(gap_sg_eval):+.2f}%  "
        f"par combo: {{{', '.join(f'E{c[0]}T{c[1]}N{c[2]}={v:+.2f}%' for c, v in gap_sg_by_combo.items())}}}")

    # ── 4. Modèle ──────────────────────────────────────────────────────────────
    model = GNN(hidden=args.hidden, use_max_pool=args.use_max_pool).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.no_scheduler:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        sched_desc = "ConstantLR"
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.n_epochs, eta_min=args.eta_min)
        sched_desc = "CosineAnnealingLR"

    n_params = sum(p.numel() for p in model.parameters())
    log(f"GNN v1CLD SSP — hidden={args.hidden}  params={n_params:,}  "
        f"use_max_pool={args.use_max_pool}  scheduler={sched_desc}")

    combos_tag = "_".join(args.combos)
    tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
    run_name = f"scaled_v1cld_{combos_tag}_lr{args.lr}_bs{args.batch_size}_h{args.hidden}_seed{args.seed}{tag_suffix}"
    metrics_path = out_dir / f"metrics_{run_name}.json"
    log(f"run : {run_name}")

    history: dict = {
        "epochs_loss": [], "loss": [], "loss_std": [], "l1_train": [],
        "lam_abs_mean": [], "lam_abs_max": [], "grad_norm_max": [], "grad_norm_mean": [],
        "epochs_eval": [], "l1_eval": [], "gap_eval": [], "loss_eval": [], "pct_better": [],
        "l1_by_combo": [],
    }
    saved_ckpts: list[Path] = []
    start_epoch = 1

    latest = find_latest_checkpoint(out_dir, run_name)
    if latest:
        log(f"Checkpoint trouvé : {latest.name} — reprise automatique")
        ckpt = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        for _ in range(start_epoch - 1):
            scheduler.step()
        history = ckpt.get("history", history)
        for k in {"epochs_loss", "loss", "loss_std", "l1_train", "lam_abs_mean", "lam_abs_max",
                  "grad_norm_max", "grad_norm_mean",
                  "epochs_eval", "l1_eval", "gap_eval", "loss_eval", "pct_better", "l1_by_combo"}:
            if k not in history:
                history[k] = []
        saved_ckpts = [
            p for p in sorted(out_dir.glob(f"ckpt_{run_name}_ep*.pt"))
            if int(p.stem.rsplit("ep", 1)[-1]) % 10 != 0
        ]
        log(f"  reprise epoch {start_epoch}")
    else:
        log("Démarrage from scratch")

    # gap baseline (sous-gradient) — dépend uniquement de l'eval set (déterministe
    # via seed), toujours recalculé fraîchement plutôt que restauré du checkpoint.
    history["gap_sg_baseline"] = float(np.mean(gap_sg_eval))
    history["gap_sg_by_combo"] = {f"{c[0]}_{c[1]}_{c[2]}": v for c, v in gap_sg_by_combo.items()}

    # ── 5. Warmup supervisé ─────────────────────────────────────────────────────
    if args.n_warmup > 0 and start_epoch == 1:
        warmup_on_roots(model, train_ds, instances, args, device)

    # ── 6. DataLoader ────────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(2 if args.num_workers > 0 else None),
    )
    n_train = len(train_ds)
    log(f"DataLoader — {n_train} graphes  workers={args.num_workers}")
    log(f"Entraînement — {args.n_epochs} epochs  batch={args.batch_size}  lr={args.lr}")

    t_start = time.time()

    # ── 7. Boucle principale ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.n_epochs + 1):
        model.train()
        epoch_loss = epoch_loss_sq = 0.0
        epoch_l1_train = 0.0
        accum_count = 0
        accum_grads = None
        epoch_gnorm = epoch_gnorm_sum = 0.0
        epoch_nsteps = 0
        lam_abs_sum = lam_abs_max_ep = 0.0

        t_ep = time.perf_counter()

        for step_i, g in enumerate(train_loader):
            g = g.to(device, non_blocking=True)
            inst = instances[int(g.inst_id.item())]

            optimizer.zero_grad()
            lam = model(g)
            if args.center_lam:
                lam = _apply_center_ssp(lam, int(g.N_z), int(g.T), int(g.n_g))

            with torch.no_grad():
                lam_d = lam.detach().abs()
                lam_abs_sum   += lam_d.mean().item()
                lam_abs_max_ep = max(lam_abs_max_ep, lam_d.max().item())

            c_scale = max(float(g.c_scale), 1.0)
            l1_raw = lagrangian_v1cld_ssp(lam, g, inst)
            epoch_l1_train += l1_raw.detach().item()
            loss = l1_raw / c_scale
            loss.backward()

            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
            epoch_gnorm      = max(epoch_gnorm, gn.item())
            epoch_gnorm_sum += gn.item()

            if accum_grads is None:
                accum_grads = [p.grad.detach().clone() if p.grad is not None else None
                               for p in model.parameters()]
            else:
                for j, p in enumerate(model.parameters()):
                    if p.grad is not None and accum_grads[j] is not None:
                        accum_grads[j] += p.grad.detach()

            lv = loss.item()
            epoch_loss    += lv
            epoch_loss_sq += lv * lv
            accum_count   += 1
            is_last = (step_i == n_train - 1)

            if (step_i + 1) % args.batch_size == 0 or is_last:
                optimizer.zero_grad()
                for p, acc in zip(model.parameters(), accum_grads):
                    if acc is not None:
                        p.grad = acc / accum_count
                epoch_nsteps += 1
                optimizer.step()
                accum_grads = None
                accum_count = 0

        scheduler.step()
        mean_loss     = epoch_loss / n_train
        loss_std      = (max(epoch_loss_sq / n_train - mean_loss ** 2, 0.0)) ** 0.5
        l1_train      = epoch_l1_train / n_train
        lam_mean_ep   = lam_abs_sum / max(n_train, 1)
        gnorm_mean_ep = epoch_gnorm_sum / max(epoch_nsteps, 1)
        ep_total      = time.perf_counter() - t_ep

        history["epochs_loss"].append(epoch)
        history["loss"].append(mean_loss)
        history["loss_std"].append(loss_std)
        history["l1_train"].append(l1_train)
        history["lam_abs_mean"].append(lam_mean_ep)
        history["lam_abs_max"].append(lam_abs_max_ep)
        history["grad_norm_max"].append(epoch_gnorm)
        history["grad_norm_mean"].append(gnorm_mean_ep)

        do_eval = (epoch % args.eval_every == 0) or (epoch == args.n_epochs)
        eval_str = ""
        if do_eval:
            l1_eval, gap_eval, pct_vs_sg = eval_epoch(eval_graphs, l1_sg_eval, instances, model, args.center_lam)
            l1_by_combo = eval_by_combo(eval_graphs, l1_sg_eval, instances, model, args.center_lam)
            loss_eval = eval_loss_epoch(eval_graphs, instances, model, args.center_lam)
            history["epochs_eval"].append(epoch)
            history["l1_eval"].append(l1_eval)
            history["gap_eval"].append(gap_eval)
            history["loss_eval"].append(loss_eval)
            history["pct_better"].append(pct_vs_sg)
            history["l1_by_combo"].append({f"{c[0]}_{c[1]}_{c[2]}": v for c, v in l1_by_combo.items()})
            eval_str = (f"  eval: L1={l1_eval:.4f}  gap_vs_opt={gap_eval:+.2f}%"
                        f" (sous-grad_T{args.eval_sg_T}={history['gap_sg_baseline']:+.2f}%)"
                        f"  loss_eval={loss_eval:.4f}  pct_vs_sg={pct_vs_sg:.1f}%")
            by_combo_str = "  ".join(
                f"E{c[0]}T{c[1]}N{c[2]}: gap={v['gap_mean']:+.2f}%"
                f"(sg={gap_sg_by_combo.get(c, float('nan')):+.2f}%)"
                f" L1={v['l1_mean']:.2f} pct_vs_sg={v['pct_better']:.1f}% (n={v['count']})"
                for c, v in l1_by_combo.items())
            log(f"  [généralisation par combo] {by_combo_str}")

        elapsed   = time.time() - t_start
        done      = epoch - start_epoch + 1
        remaining = elapsed / done * (args.n_epochs - epoch) if done > 0 else 0
        log(f"ep {epoch:4d}/{args.n_epochs}  loss={mean_loss:.4f}  L1_train={l1_train:.2f}"
            f"  |λ|={lam_mean_ep:.3f}(max={lam_abs_max_ep:.2f})  gnorm={epoch_gnorm:.3f}"
            f"  lr={optimizer.param_groups[0]['lr']:.1e}"
            f"{eval_str}  t={ep_total:.0f}s  reste~{remaining/60:.0f}min")

        save_checkpoint(out_dir, run_name, epoch, model, optimizer, scheduler,
                        history, args, saved_ckpts, args.keep_ckpts)
        save_history(metrics_path, history)

    log("Entraînement terminé.")


if __name__ == "__main__":
    main()
