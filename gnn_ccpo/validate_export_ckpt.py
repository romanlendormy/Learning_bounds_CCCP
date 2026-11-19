"""
validate_export_ckpt.py — vérifie un export TorchScript batché (ccpo_gnn_scriptable_batched.py) face au modèle eager
du checkpoint source, sur des nœuds réels (groupes même instance / même n_free = contrat de batching du B&B).
Usage : python validate_export_ckpt.py CKPT SCRIPTED --data_dir D
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import train_ccpo_gnn as T  # noqa: E402
from ccpo_graph import EDGE_MODES  # noqa: E402
from test_correspondence import TOL, batch_groups, pick_nodes, run_scriptable  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("scripted")
    ap.add_argument("--data_dir", default=os.environ.get("CCPO_DATA_DIR"), help="dossier de collecte (défaut $CCPO_DATA_DIR)")
    a = ap.parse_args()
    torch.set_num_threads(1)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    T.configure_graph_from_args(ck["args"])
    em = ck["args"].get("edge_mode") or "none"
    print(f"[ckpt] epoch {ck['epoch']} feat_version={T.GRAPH_CFG['feat_version']} edge_mode={em}", flush=True)
    eager = T.model_from_args(ck["args"]); eager.load_state_dict(ck["model"], strict=True); eager.eval()
    jm = torch.jit.load(a.scripted).eval()
    print(f"[export] feat_version={getattr(jm, 'feat_version', 'absent')} edge_mode={getattr(jm, 'edge_mode', 'absent')}")
    cells = T.load_all(Path(a.data_dir), node_suffix="_feasible")
    mx, n = 0.0, 0
    with torch.no_grad():
        for builts in batch_groups(cells, pick_nodes(cells), want=8):
            ref = torch.cat([eager(b[0]) for b in builts])
            out = run_scriptable(jm, builts, EDGE_MODES[em])
            mx = max(mx, float((out - ref).abs().max())); n += len(builts)
    ok = (getattr(jm, "feat_version", None) == T.GRAPH_CFG["feat_version"]) and mx < TOL and n > 0
    print(f"{n} graphes, max|diff| eager vs TorchScript = {mx:.2e} (tol {TOL})")
    print("STATUT:", "OK" if ok else "ECHEC")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
