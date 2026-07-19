#!/usr/bin/env python3
"""
probe_gru_decomposed_aoti_cuda.py — Valide la décomposition du GRUCell
(GatedGraphConvDecomposed, gnn_sd_edit_scriptable_batched.py) en deux temps :

  1. Équivalence numérique : gru_cell_decomposed(...) vs nn.GRUCell natif,
     sur les MÊMES poids/entrées aléatoires — doit être quasi exact (juste
     une défusion de la même formule).
  2. Le vrai test bloquant : torch.export + aoti_compile_and_package sur
     GNNSdEditScriptableBatchedAOTI (avec conv1/conv2 décomposés), en CUDA,
     B dynamique — reproduit exactement la sonde du 2026-07-17
     (probe_aoti_batched_dynamic_B.py) qui échouait sur
     aten._thnn_fused_gru_cell, pour confirmer que la décomposition lève le
     blocage.
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path("/lustre09/project/6062881/romanlen/Learning_bounds_CCCP")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "gnn_sd_edit"))

torch.use_deterministic_algorithms(True)

from gnn_sd_edit_scriptable_batched import (  # noqa: E402
    GatedGraphConvDecomposed, GNNSdEditScriptableBatchedAOTI, make_batched_edges,
)

CKPT = str(REPO_ROOT / "gnn_sd_edit/runs/train_scaled/"
           "ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt")
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cpu"


def check_gru_equivalence(hidden, device):
    print(f"[probe] --- 1. Équivalence numérique GRUCell décomposé vs natif (hidden={hidden}) ---")
    torch.manual_seed(42)
    conv = GatedGraphConvDecomposed(hidden, 2).to(device)
    B = 37
    inp = torch.randn(B, hidden, device=device)
    hid = torch.randn(B, hidden, device=device)

    with torch.no_grad():
        out_decomposed = conv.gru_cell_decomposed(inp, hid)
        out_native = conv.rnn(inp, hid)

    max_diff = (out_decomposed - out_native).abs().max().item()
    print(f"[probe] max|diff| decomposed vs natif = {max_diff:.3e}")
    if max_diff >= 1e-4:
        print("[probe] ÉCHEC : décomposition GRUCell incorrecte.")
        sys.exit(1)
    print("[probe] OK — décomposition GRUCell numériquement équivalente.\n")


def build_inputs(n_g, N_z, B, device):
    src, dst, ew = [], [], []
    for i in range(N_z):
        for k1 in range(n_g):
            for k2 in range(k1 + 1, n_g):
                a, b = i * n_g + k1, i * n_g + k2
                src += [a, b]; dst += [b, a]; ew += [1.0 / n_g, 1.0 / n_g]
    for kf in range(n_g):
        anchor = kf
        for i2 in range(1, N_z):
            spoke = i2 * n_g + kf
            src += [anchor, spoke]; dst += [spoke, anchor]; ew += [1.0, 1.0]
    e_single = torch.tensor([src, dst], dtype=torch.int64, device=device)
    ew_single = torch.tensor(ew, dtype=torch.float32, device=device)
    v_per_graph = N_z * n_g
    edge_index, edge_weight = make_batched_edges(e_single, ew_single, v_per_graph, B)
    return edge_index, edge_weight, v_per_graph


def main():
    print(f"[probe] device={DEVICE} torch={torch.__version__}")
    ckpt = torch.load(CKPT, map_location="cpu")
    args = ckpt["args"]
    m, hidden = int(args["m"]), int(args["hidden"])
    use_max_pool = bool(args.get("use_max_pool", False))
    center_lam = bool(args.get("center_lam", False))
    node_dim = 7 * m + 5
    print(f"[probe] m={m} hidden={hidden} use_max_pool={use_max_pool} center_lam={center_lam}")

    check_gru_equivalence(hidden, DEVICE)

    print("[probe] --- 2. Modèle complet : chargement checkpoint dans la variante décomposée ---")
    model = GNNSdEditScriptableBatchedAOTI(m, hidden, use_max_pool, center_lam)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval().to(DEVICE)
    print("[probe] load_state_dict(strict=True) OK — mêmes clés que le checkpoint de production.\n")

    n_g, N_z = 20, 100
    B_export = 64
    B_test = 13

    torch.manual_seed(0)
    edge_index_exp, edge_weight_exp, v_per_graph = build_inputs(n_g, N_z, B_export, DEVICE)
    X_exp = torch.rand(B_export * v_per_graph, node_dim, dtype=torch.float32, device=DEVICE)
    ctx_exp = torch.rand(B_export, 9, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        y_eager_export_shape = model(X_exp, edge_index_exp, edge_weight_exp, ctx_exp, n_g)
    print(f"[probe] forward eager (décomposé) OK à B={B_export} : y.shape={tuple(y_eager_export_shape.shape)}")

    print("[probe] --- 3. torch.export + aoti_compile_and_package (B dynamique, CUDA) ---")
    B_dim = torch.export.Dim("B", min=1, max=4096)
    V = v_per_graph
    E = edge_index_exp.size(1) // B_export
    dynamic_shapes = {
        "X": {0: B_dim * V},
        "edge_index": {1: B_dim * E},
        "edge_weight": {0: B_dim * E},
        "context": {0: B_dim},
        "n_g": None,
    }

    t0 = time.time()
    try:
        ep = torch.export.export(
            model, (X_exp, edge_index_exp, edge_weight_exp, ctx_exp, n_g),
            dynamic_shapes=dynamic_shapes,
        )
        print(f"[probe] torch.export.export OK ({time.time()-t0:.1f}s)")
    except Exception as exc:
        print(f"[probe] ÉCHEC torch.export.export : {type(exc).__name__}: {exc}")
        sys.exit(1)

    t0 = time.time()
    try:
        pkg_path = torch._inductor.aoti_compile_and_package(
            ep, package_path="/tmp/probe_gru_decomposed_ng20.pt2",
        )
        print(f"[probe] aoti_compile_and_package OK ({time.time()-t0:.1f}s) -> {pkg_path}")
    except Exception as exc:
        print(f"[probe] ÉCHEC aoti_compile_and_package : {type(exc).__name__}: {exc}")
        sys.exit(1)

    loaded = torch._inductor.aoti_load_package(pkg_path)

    with torch.no_grad():
        y_loaded_same_B = loaded(X_exp, edge_index_exp, edge_weight_exp, ctx_exp)
    diff_same_B = (y_loaded_same_B - y_eager_export_shape).abs().max().item()
    print(f"[probe] B={B_export} (vu à l'export) : max|diff| vs eager = {diff_same_B:.3e}")

    torch.manual_seed(1)
    edge_index_test, edge_weight_test, _ = build_inputs(n_g, N_z, B_test, DEVICE)
    X_test = torch.rand(B_test * v_per_graph, node_dim, dtype=torch.float32, device=DEVICE)
    ctx_test = torch.rand(B_test, 9, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        y_eager_test = model(X_test, edge_index_test, edge_weight_test, ctx_test, n_g)

    try:
        y_loaded_test = loaded(X_test, edge_index_test, edge_weight_test, ctx_test)
    except Exception as exc:
        print(f"[probe] ÉCHEC appel compilé à B={B_test} (jamais vu à l'export) : "
              f"{type(exc).__name__}: {exc}")
        sys.exit(1)

    diff_test = (y_loaded_test - y_eager_test).abs().max().item()
    print(f"[probe] B={B_test} (JAMAIS vu à l'export) : max|diff| vs eager = {diff_test:.3e}")

    # ── Comparaison finale : modèle décomposé compilé vs modèle ORIGINAL
    # (fusionné, TorchScript déjà en production) — confirme que la
    # décomposition n'a rien changé au calcul réel du modèle de production.
    from gnn_sd_edit_scriptable_batched import GNNSdEditScriptableBatched
    ref_model = GNNSdEditScriptableBatched(m, hidden, use_max_pool, center_lam)
    ref_model.load_state_dict(ckpt["model"], strict=True)
    ref_model.eval().to(DEVICE)
    with torch.no_grad():
        y_ref_fused = ref_model(X_test, edge_index_test, edge_weight_test, ctx_test, n_g, B_test)
    diff_vs_prod = (y_loaded_test - y_ref_fused).abs().max().item()
    print(f"[probe] B={B_test} : max|diff| compilé(décomposé) vs modèle de PRODUCTION (fusionné) = {diff_vs_prod:.3e}")

    if diff_test < 1e-4 and diff_same_B < 1e-4 and diff_vs_prod < 1e-3:
        print("[probe] VERDICT : décomposition GRUCell + B dynamique FONCTIONNENT sur CUDA.")
    else:
        print("[probe] VERDICT : écart trop grand quelque part — voir le détail ci-dessus.")


if __name__ == "__main__":
    main()
