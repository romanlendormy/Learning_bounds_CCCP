"""
test_correspondence.py — Correspondance Python / C++ / TorchScript du GNN CCPO, sur des nœuds RÉELS
de la collecte (toutes cellules, profondeurs variées) :

  1. Graphe Python (ccpo_graph.py) vs C++ (ccpo_graph.so) : features de nœud v1/v2/v3, contexte,
     arêtes et poids d'arêtes façon CCMKP.
  2. Modèle eager CCPOGNN vs CCPOGNNScriptableBatched (B=1 et B>1) vs module TorchScript
     sauvegardé/rechargé, pour plusieurs configurations (poids aléatoires) ; attributs
     feat_version/edge_mode présents dans l'export ; centrage : somme nulle sur les scénarios éligibles.

Usage : python3 gnn/ccpo/test_correspondence.py [--data_dir DIR]
  (DIR : dossier de collecte, nodes_*_feasible.txt ; défaut $CCPO_DATA_DIR)
Sortie : "STATUT: OK" / "STATUT: ECHEC" (code de retour 1).
"""
from __future__ import annotations

import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import train_ccpo_gnn as T  # noqa: E402
from ccpo_graph import (build_context_features, build_edge_weights_mkp, build_edges,  # noqa: E402
                        build_node_features, build_node_features_v2, build_node_features_v3, EDGE_MODES)
from ccpo_graph_cpp_bind import (build_context_features_v_cpp, build_edge_weights_mkp_cpp,  # noqa: E402
                                 build_edges_cpp, build_node_features_cpp, build_node_features_v2_cpp,
                                 build_context_features_v3_cpp, build_node_features_v3_cpp)
from ccpo_gnn_scriptable_batched import CCPOGNNScriptableBatched, make_batched_edges, scriptable_from_args  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
N_PER_CELL = 12
TOL = 1e-5

fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)
        print("  !! " + msg, flush=True)


def pick_nodes(cells):
    """N_PER_CELL nœuds par cellule, profondeurs variées (racines comprises), plus
    des groupes (même instance, même n_free) pour les tests de batching."""
    rng = np.random.default_rng(0)
    out = []
    for (n, N), cell in sorted(cells.items()):
        cands = [T.Example(n, N, iid, r.depth, k) for iid, recs in cell.nodes_by_instance.items()
                 for k, r in enumerate(recs)]
        roots = [e for e in cands if e.depth == 0]
        deep = sorted([e for e in cands if e.depth > 0], key=lambda e: -e.depth)
        picks = list(rng.choice(len(cands), N_PER_CELL - 4, replace=False))
        sel = [cands[i] for i in picks] + roots[:2] + deep[:2]
        out.extend(sel)
    return out


def test_graph(cells, examples):
    print("== 1. Graphe Python vs C++ ==", flush=True)
    mx = defaultdict(float)
    n_ok = 0
    n_infeas_nodes = 0
    for ex in examples:
        b = T.build_from_example(cells, ex)
        if b is None:
            continue
        _, ri, _, _, _ = b
        args = (ri.n_free, ri.N, ri.cost_free, ri.Sigma_free, ri.B_res, ri.R_ext, ri.infeasible,
                ri.Q_res, ri.U, ri.c_scale, ri.sigma_scale)
        elig = np.where(~ri.infeasible)[0]
        n_infeas_nodes += int(len(elig) < ri.N)
        f1c, e1 = build_node_features_cpp(*args)
        f2c, e2 = build_node_features_v2_cpp(*args)
        check(np.array_equal(e1, elig) and np.array_equal(e2, elig), f"eligible {ex}")
        f1p, f2p = build_node_features(ri), build_node_features_v2(ri)
        check(f2p.shape == f2c.shape == (len(elig) * ri.n_free, 26), f"shape v2 {ex}")
        check(np.isfinite(f2p).all(), f"v2 non fini {ex}")
        mx["feat_v1"] = max(mx["feat_v1"], float(np.abs(f1p.astype(np.float64) - f1c).max(initial=0)))
        d2 = np.abs(f2p.astype(np.float64) - f2c)
        mx["feat_v2"] = max(mx["feat_v2"], float(d2.max(initial=0)))
        if d2.size and d2.max() > TOL:
            col = int(np.unravel_index(d2.argmax(), d2.shape)[1])
            print(f"  v2 col {col} écart {d2.max():.3e} {ex}", flush=True)
        check(ri.B_orig > 0, f"B_orig non renseigné {ex}")
        f3c, e3 = build_node_features_v3_cpp(ri.n_free, ri.N, ri.cost_free, ri.Sigma_free, ri.B_res, ri.R_ext,
                                             ri.infeasible, ri.Q_res, ri.U, ri.c_scale, ri.B_orig)
        f3p = build_node_features_v3(ri)
        check(np.array_equal(e3, elig), f"eligible v3 {ex}")
        check(f3p.shape == f3c.shape == (len(elig) * ri.n_free, 28), f"shape v3 {ex}")
        check(np.isfinite(f3p).all(), f"v3 non fini {ex}")
        d3 = np.abs(f3p.astype(np.float64) - f3c)
        mx["feat_v3"] = max(mx["feat_v3"], float(d3.max(initial=0)))
        if d3.size and d3.max() > TOL:
            col = int(np.unravel_index(d3.argmax(), d3.shape)[1])
            print(f"  v3 col {col} écart {d3.max():.3e} {ex}", flush=True)
        # colonnes inchangées v2 -> v3
        same = [2, 5, 10, 11, 12, 13, 21, 22, 25]
        mx["v3_vs_v2_cols_inchangees"] = max(mx["v3_vs_v2_cols_inchangees"],
                                             float(np.abs(f3p[:, same] - f2p[:, same]).max(initial=0)))
        for fv in (1, 2, 3):
            cp = build_context_features(ri, fv)
            cc = build_context_features_v3_cpp(ri.n_free, ri.N, ri.k_res, ri.Q_res, ri.U, ri.B_res,
                                               ri.infeasible, ri.depth, ri.n_total, ri.frac_fixed_active,
                                               ri.cost_fixed, ri.c_scale, ri.sigma_scale, ri.epsilon, fv, ri.B_orig)
            mx[f"ctx_v{fv}"] = max(mx[f"ctx_v{fv}"], float(np.abs(cp.astype(np.float64) - cc).max()))
            if fv < 3:  # ancien point d'entrée (sans B_orig) : non-régression
                co = build_context_features_v_cpp(ri.n_free, ri.N, ri.k_res, ri.Q_res, ri.U, ri.B_res,
                                                  ri.infeasible, ri.depth, ri.n_total, ri.frac_fixed_active,
                                                  ri.cost_fixed, ri.c_scale, ri.sigma_scale, ri.epsilon, fv)
                mx[f"ctx_v{fv}_ancien_point_entree"] = max(mx[f"ctx_v{fv}_ancien_point_entree"],
                                                          float(np.abs(cp.astype(np.float64) - co).max()))
        ep = build_edges(ri)
        ec = build_edges_cpp(ri.n_free, len(elig))
        check(ep.shape == ec.shape and np.array_equal(ep, ec), f"edges {ex}")
        wp = build_edge_weights_mkp(ri.n_free, len(elig))
        wc = build_edge_weights_mkp_cpp(ri.n_free, len(elig))
        check(wp.shape == (ep.shape[1],) and np.array_equal(wp, wc), f"poids mkp {ex}")
        n_ok += 1
    print(f"  {n_ok} nœuds ({n_infeas_nodes} avec |E|<N) ; max|diff| : " +
          ", ".join(f"{k}={v:.2e}" for k, v in mx.items()), flush=True)
    for k, v in mx.items():
        check(v < TOL, f"{k} max|diff|={v:.3e} >= {TOL}")


def batch_groups(cells, examples, want=3):
    """Groupes de nœuds (même instance, même n_free) -- contrat de batching du B&B."""
    groups = []
    seen = set()
    for ex in examples:
        key = (ex.n, ex.N, ex.instance_id)
        if key in seen:
            continue
        seen.add(key)
        by_nf = defaultdict(list)
        for k, r in enumerate(cells[(ex.n, ex.N)].nodes_by_instance[ex.instance_id][:60]):
            b = T.build_from_example(cells, T.Example(ex.n, ex.N, ex.instance_id, r.depth, k))
            if b is not None:
                by_nf[b[1].n_free].append(b)
        best = max(by_nf.values(), key=len)
        if len(best) >= 2:
            groups.append(best[:5])
        if len(groups) >= want:
            break
    return groups


def run_scriptable(m, builts, edge_mode_int):
    g0 = builts[0][0]
    n_free, n_elig = int(g0.n_free), int(g0.n_scenarios_eligible)
    X = torch.cat([b[0].x for b in builts])
    C = torch.cat([b[0].ctx for b in builts])
    ew = g0.edge_weight if edge_mode_int == EDGE_MODES["mkp"] else None
    ei, w = make_batched_edges(g0.edge_index, n_free * n_elig, len(builts), ew)
    return m(X, ei, w, C, n_free, n_elig, len(builts))


def test_models(cells, examples):
    print("== 2. Eager vs scriptable vs TorchScript (configurations v1/v2/v3) ==", flush=True)
    configs = [
        dict(feat_version=2, edge_mode="mkp", center_lam=True, relu_h1=False),
        dict(feat_version=2, edge_mode="mkp", center_lam=False, relu_h1=True),
        dict(feat_version=2, edge_mode="none", center_lam=True, relu_h1=True),
        dict(feat_version=1, edge_mode="mkp", center_lam=True, relu_h1=False),
        dict(feat_version=3, edge_mode="mkp", center_lam=False, relu_h1=True),
        dict(feat_version=3, edge_mode="none", center_lam=False, relu_h1=True),
    ]
    for cfg in configs:
        args = dict(hidden=128, **cfg)
        T.configure_graph_from_args(args)
        torch.manual_seed(1)
        eager = T.model_from_args(args).eval()
        scr = scriptable_from_args(args)
        missing, unexpected = scr.load_state_dict(eager.state_dict(), strict=True)
        scr.eval()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.pt"
            torch.jit.script(scr).save(str(path))
            jm = torch.jit.load(str(path)).eval()
        check(int(jm.feat_version) == cfg["feat_version"] and int(jm.edge_mode) == EDGE_MODES[cfg["edge_mode"]],
              f"attributs export {cfg}")
        groups = batch_groups(cells, examples)
        mx = 0.0
        sum_e = 0.0
        with torch.no_grad():
            for builts in groups:
                ref = torch.cat([eager(b[0]) for b in builts])
                for m in (scr, jm):
                    out1 = torch.cat([run_scriptable(m, [b], EDGE_MODES[cfg["edge_mode"]]) for b in builts])
                    outB = run_scriptable(m, builts, EDGE_MODES[cfg["edge_mode"]])
                    mx = max(mx, float((out1 - ref).abs().max()), float((outB - ref).abs().max()))
                if cfg["center_lam"]:
                    for b in builts:
                        h = eager(b[0]).view(int(b[0].n_scenarios_eligible), -1)
                        sum_e = max(sum_e, float(h.sum(0).abs().max()))
        print(f"  {cfg} : {sum(len(g) for g in groups)} graphes, max|diff|={mx:.2e}"
              + (f", max|Σ_E h|={sum_e:.2e}" if cfg["center_lam"] else ""), flush=True)
        check(mx < TOL, f"eager/scriptable {cfg} : {mx:.3e}")
        if cfg["center_lam"]:
            check(sum_e < 1e-4, f"centrage {cfg} : {sum_e:.3e}")
    T.configure_graph_from_args({})


def main():
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.environ.get("CCPO_DATA_DIR"))
    a = ap.parse_args()
    if not a.data_dir:
        sys.exit("--data_dir (ou CCPO_DATA_DIR) requis : dossier de collecte CCPO")
    torch.set_num_threads(1)
    cells = T.load_all(Path(a.data_dir), node_suffix="_feasible")
    examples = pick_nodes(cells)
    test_graph(cells, examples)
    test_models(cells, examples)
    print("STATUT: " + ("OK" if not fails else f"ECHEC ({len(fails)})"), flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
