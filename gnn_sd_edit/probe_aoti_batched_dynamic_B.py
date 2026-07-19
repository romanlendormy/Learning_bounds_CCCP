#!/usr/bin/env python3
"""
probe_aoti_batched_dynamic_B.py — Sonde de faisabilité (PAS un export de
production) : est-ce que GNNSdEditScriptableBatched peut être exporté via
AOTInductor avec B (taille du batch de nœuds) marqué DYNAMIQUE, tout en
gardant n_g/N_z figés en constantes (comme le fait déjà export_aoti_sdedit.py
pour le modèle non-batché) ?

Enjeu : contrairement à n_g (~20 valeurs possibles, une par profondeur), B
varie quasi continûment (1..K à chaque flush de bucket, cf. "Remplissage
moyen au flush: 58.63/K=64"). Si B doit être figé à l'export (comme n_g),
il faudrait jusqu'à K exports par n_g (K x n exports au total) — bien plus
coûteux que le cas non-batché. Si B peut être dynamique, un seul export par
n_g suffit (même échelle que le cas non-batché déjà validé).

Test : exporte à B=64 (n_g=20, N_z=100 — config de nos runs), puis recharge
et appelle avec B=13 (valeur différente, jamais vue à l'export) — si le
graphe compilé gère B=13 correctement (comparé au forward eager), la
dynamicité fonctionne vraiment (pas juste "l'export a réussi mais reste
figé sur B=64").
"""
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path("/lustre09/project/6062881/romanlen/Learning_bounds_CCCP")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "gnn_sd_edit"))

torch.use_deterministic_algorithms(True)

from gnn_sd_edit_scriptable_batched import GNNSdEditScriptableBatched, make_batched_edges  # noqa: E402


class GNNSdEditScriptableBatchedDynB(GNNSdEditScriptableBatched):
    """
    Variante de sonde (PAS le modèle de production) : B dérivé de
    context.shape[0] au lieu d'être un argument scalaire séparé — B est de
    toute façon TOUJOURS égal à context.size(0) par construction (une ligne
    de contexte par membre du batch), donc ce n'est pas une approximation,
    juste une autre façon d'obtenir la même valeur. Teste si l'échec
    AOTInductor ("SymInt et NoneType" sur v_total // B) vient spécifiquement
    du fait que B soit un ARGUMENT SCALAIRE dynamique séparé plutôt qu'une
    quantité dérivée d'une dimension de tenseur déjà marquée dynamique.
    """
    def forward(self, X: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor,
                context: torch.Tensor, n_g: int) -> torch.Tensor:
        B = context.size(0)
        v_total = X.size(0)
        v_per_graph = v_total // B
        n_z = v_per_graph // n_g
        dim_last = 2 * self.hidden

        ctx_norm = self.norm_ctx(context)

        h = self.norm_embed(self.embed(X))
        h1 = self.norm1(self.conv1(h, edge_index, edge_weight))
        h2 = self.norm2(self.conv2(h1, edge_index, edge_weight)) + h1
        h_v = self.proj(h2)

        h_v_4d = h_v.view(B, n_z, n_g, dim_last)
        h_item_mean = h_v_4d.mean(1)
        h_scen_mean = h_v_4d.mean(2)
        g_pool = h_item_mean.mean(1)

        if self.use_max_pool:
            h_item_max, _ = h_v_4d.max(1)
            h_scen_max, _ = h_v_4d.max(2)
            h_item = torch.cat([h_item_mean, h_item_max], dim=-1)
            h_scen = torch.cat([h_scen_mean, h_scen_max], dim=-1)
        else:
            h_item = h_item_mean
            h_scen = h_scen_mean

        item_dim = h_item.size(-1)
        scen_dim = h_scen.size(-1)

        h_item_exp = h_item.unsqueeze(1).expand(B, n_z, n_g, item_dim).reshape(v_total, item_dim)
        h_scen_exp = h_scen.unsqueeze(2).expand(B, n_z, n_g, scen_dim).reshape(v_total, scen_dim)
        g_exp = g_pool.unsqueeze(1).expand(B, v_per_graph, dim_last).reshape(v_total, dim_last)
        ctx_exp = ctx_norm.unsqueeze(1).expand(B, v_per_graph, self.CONTEXT_DIM).reshape(v_total, self.CONTEXT_DIM)

        cat = torch.cat([h_v, h_item_exp, h_scen_exp, g_exp, ctx_exp], dim=-1)
        h_out = self.head(cat).squeeze(-1)

        if self.center_lam:
            h_out_3d = h_out.view(B, n_z, n_g)
            h_out = (h_out_3d - h_out_3d.mean(1, keepdim=True)).reshape(-1)
        return h_out

CKPT = str(REPO_ROOT / "gnn_sd_edit/runs/train_scaled/"
           "ckpt_sdedit_scaled_n10_20_lr0.0003_bs32_h128_seed42_v1_sdedit_500ep_ep0500.pt")
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cpu"


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

    model = GNNSdEditScriptableBatchedDynB(m, hidden, use_max_pool, center_lam)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval().to(DEVICE)

    n_g, N_z = 20, 100
    B_export = 64
    B_test = 13   # jamais vu à l'export — teste la vraie dynamicité, pas juste "ça a compilé"

    torch.manual_seed(0)
    edge_index_exp, edge_weight_exp, v_per_graph = build_inputs(n_g, N_z, B_export, DEVICE)
    X_exp = torch.rand(B_export * v_per_graph, node_dim, dtype=torch.float32, device=DEVICE)
    ctx_exp = torch.rand(B_export, 9, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        y_eager_export_shape = model(X_exp, edge_index_exp, edge_weight_exp, ctx_exp, n_g)
    print(f"[probe] forward eager OK à B={B_export} : y.shape={tuple(y_eager_export_shape.shape)}")

    # ── Marquage dynamique de B (dérivé de context.shape[0], plus d'argument
    # scalaire B séparé) ─────────────────────────────────────────────────────
    B_dim = torch.export.Dim("B", min=1, max=256)
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
            model,
            (X_exp, edge_index_exp, edge_weight_exp, ctx_exp, n_g),
            dynamic_shapes=dynamic_shapes,
        )
        print(f"[probe] torch.export.export OK ({time.time()-t0:.1f}s)")
    except Exception as exc:
        print(f"[probe] ÉCHEC torch.export.export : {type(exc).__name__}: {exc}")
        sys.exit(1)

    t0 = time.time()
    try:
        pkg_path = torch._inductor.aoti_compile_and_package(
            ep, package_path=f"/tmp/probe_aoti_batched_ng{n_g}.pt2",
        )
        print(f"[probe] aoti_compile_and_package OK ({time.time()-t0:.1f}s) -> {pkg_path}")
    except Exception as exc:
        print(f"[probe] ÉCHEC aoti_compile_and_package : {type(exc).__name__}: {exc}")
        sys.exit(1)

    loaded = torch._inductor.aoti_load_package(pkg_path)

    # ── Test à B_export (sanity de base) ───────────────────────────────────
    with torch.no_grad():
        y_loaded_same_B = loaded(X_exp, edge_index_exp, edge_weight_exp, ctx_exp, n_g)
    diff_same_B = (y_loaded_same_B - y_eager_export_shape).abs().max().item()
    print(f"[probe] B={B_export} (vu à l'export) : max|diff| vs eager = {diff_same_B:.3e}")

    # ── Test à B_test (JAMAIS vu à l'export) — la vraie question ───────────
    torch.manual_seed(1)
    edge_index_test, edge_weight_test, _ = build_inputs(n_g, N_z, B_test, DEVICE)
    X_test = torch.rand(B_test * v_per_graph, node_dim, dtype=torch.float32, device=DEVICE)
    ctx_test = torch.rand(B_test, 9, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        y_eager_test = model(X_test, edge_index_test, edge_weight_test, ctx_test, n_g)

    try:
        y_loaded_test = loaded(X_test, edge_index_test, edge_weight_test, ctx_test, n_g)
    except Exception as exc:
        print(f"[probe] ÉCHEC appel compilé à B={B_test} (jamais vu à l'export) : "
              f"{type(exc).__name__}: {exc}")
        sys.exit(1)

    diff_test = (y_loaded_test - y_eager_test).abs().max().item()
    print(f"[probe] B={B_test} (JAMAIS vu à l'export) : max|diff| vs eager = {diff_test:.3e}")

    if diff_test < 1e-4 and diff_same_B < 1e-4:
        print("[probe] VERDICT : B dynamique FONCTIONNE — un seul export par n_g suffirait.")
    else:
        print("[probe] VERDICT : écart trop grand à un B non vu à l'export — B dynamique "
              "NE fonctionne PAS correctement (ou pas de cette façon).")


if __name__ == "__main__":
    main()
