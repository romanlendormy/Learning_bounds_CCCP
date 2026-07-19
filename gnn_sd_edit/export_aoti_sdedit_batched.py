#!/usr/bin/env python3
"""
export_aoti_sdedit_batched.py — exporte le modèle sd_edit BATCHÉ (checkpoint
de production) en artefacts AOTInductor, un par valeur de n_g rencontrée,
avec B (taille du batch de nœuds) DYNAMIQUE à l'intérieur de chaque artefact
— pas un export par (n_g, B), qui serait rédhibitoire puisque B varie quasi
continûment (1..K à chaque flush de bucket), contrairement à n_g (~n valeurs
possibles, une par profondeur). Faisabilité validée le 2026-07-17 (cf.
probe_aoti_batched_dynamic_B.py) : export à B=64 généralise correctement à
B=13 jamais vu à l'export (max|diff| ~1e-5).

Utilise GNNSdEditScriptableBatchedAOTI (gnn_sd_edit_scriptable_batched.py) —
IDENTIQUE à GNNSdEditScriptableBatched (mêmes poids, même calcul), sauf que B
est dérivé de context.shape[0] au lieu d'être un argument scalaire séparé
(nécessaire pour que torch.export/aoti_compile_and_package n'échoue pas sur
`v_total // B`, cf. docstring de la classe).

Appelé par GNNSdEditNativeAOTIBatched (solver_bab/mkp/gnn_sd_edit_native_aoti_batched.cpp)
via posix_spawn, même mécanisme que export_aoti_sdedit.py (non-batché) et
ensure_scripted_batched_export (TorchScript batché).

Cache : <checkpoint>.aoti_batched_N<n_z>/ng<K>.pt2 — préfixe "batched"
distinct du cache non-batché (<checkpoint>.aoti_N<n_z>/ng<K>.pt2) : modèles
différents (forward batché vs non-batché), ne doivent PAS partager de cache
même pour le même checkpoint/N.
"""
import argparse
import sys
from pathlib import Path

import torch

# Déterminisme requis — même raison que export_aoti_sdedit.py (cf. mémoire
# "aoti-native-integration-hang") : sans ceci, le forward AOTInductor compilé
# jitte de ~1e-5 entre deux appels à données identiques, suffisant pour faire
# osciller la borne entière après floor/scale et casser la convergence du
# point fixe Gecode.
torch.use_deterministic_algorithms(True)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from gnn_sd_edit_scriptable_batched import GNNSdEditScriptableBatchedAOTI, make_batched_edges  # noqa: E402

B_EXPORT_DEFAULT = 64  # valeur de trace — B reste dynamique au runtime (cf. docstring module), ce choix n'affecte que le temps de compilation


def build_sdedit_edges(n_g: int, N_z: int):
    """Identique à build_sdedit_edges (export_aoti_sdedit.py) — dupliqué,
    pas importé (convention du repo : chaque script d'export reste
    autonome)."""
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
    return src, dst, ew


def cache_dir_for(checkpoint: str, n_z: int) -> Path:
    return Path(f"{checkpoint}.aoti_batched_N{n_z}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n-g", type=int, nargs="+", required=True)
    p.add_argument("--n-z", type=int, required=True)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--batch-size-export", type=int, default=B_EXPORT_DEFAULT,
                    help="Taille de batch utilisée pour la trace (B reste dynamique au runtime)")
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_dir = cache_dir_for(args.checkpoint, args.n_z)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_mtime = ckpt_path.stat().st_mtime

    to_export = []
    for n_g in args.n_g:
        dest = out_dir / f"ng{n_g}.pt2"
        if dest.exists() and dest.stat().st_mtime >= ckpt_mtime:
            print(f"[export_batched] n_g={n_g} : cache à jour ({dest}) — skip")
        else:
            to_export.append(n_g)

    if not to_export:
        print("[export_batched] tous les n_g demandés sont déjà en cache.")
        return

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt["args"]
    m, hidden = int(ckpt_args["m"]), int(ckpt_args["hidden"])
    use_max_pool = bool(ckpt_args.get("use_max_pool", False))
    center_lam = bool(ckpt_args.get("center_lam", False))
    print(f"[export_batched] checkpoint={args.checkpoint} m={m} hidden={hidden} "
          f"use_max_pool={use_max_pool} center_lam={center_lam} "
          f"n_g à exporter={to_export} B_export={args.batch_size_export}")

    model = GNNSdEditScriptableBatchedAOTI(m, hidden, use_max_pool, center_lam)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval().to(args.device)

    node_dim = 7 * m + 5
    N_z = args.n_z
    B_export = args.batch_size_export
    B_sanity = max(1, B_export // 5 + 3)  # valeur volontairement DIFFÉRENTE de B_export, jamais vue à l'export

    for n_g in to_export:
        V = N_z * n_g
        src, dst, ew = build_sdedit_edges(n_g, N_z)
        e_single = torch.tensor([src, dst], dtype=torch.int64, device=args.device)
        ew_single = torch.tensor(ew, dtype=torch.float32, device=args.device)

        def make_batch(B, seed):
            torch.manual_seed(seed)
            ei, ewt = make_batched_edges(e_single, ew_single, V, B)
            X = torch.rand(B * V, node_dim, dtype=torch.float32, device=args.device)
            ctx = torch.rand(B, 9, dtype=torch.float32, device=args.device)
            return X, ei, ewt, ctx

        X, edge_index, edge_weight, context = make_batch(B_export, seed=0)

        with torch.no_grad():
            y_ref = model(X, edge_index, edge_weight, context, n_g)

        B_dim = torch.export.Dim("B", min=1, max=4096)
        E = edge_index.size(1) // B_export
        dynamic_shapes = {
            "X": {0: B_dim * V},
            "edge_index": {1: B_dim * E},
            "edge_weight": {0: B_dim * E},
            "context": {0: B_dim},
            "n_g": None,
        }

        # aoti_compile_and_package (PAS aot_compile+package=False) : même
        # raison que le modèle non-batché — nn.GRUCell -> aten._thnn_fused_gru_cell
        # nécessite le proxy executor généré par la voie "package" (.pt2).
        ep = torch.export.export(model, (X, edge_index, edge_weight, context, n_g),
                                  dynamic_shapes=dynamic_shapes)
        dest = out_dir / f"ng{n_g}.pt2"
        pkg_path = torch._inductor.aoti_compile_and_package(ep, package_path=str(dest))

        loaded = torch._inductor.aoti_load_package(pkg_path)

        # Sanity à B_export ET à un B DIFFÉRENT jamais vu à l'export — un
        # export "réussi" mais figé sur B_export serait un bug silencieux
        # bien pire qu'un échec net (cf. docstring module).
        y_loaded_same_B = loaded(X, edge_index, edge_weight, context)
        max_diff_same_B = (y_loaded_same_B - y_ref).abs().max().item()

        X2, ei2, ew2, ctx2 = make_batch(B_sanity, seed=1)
        with torch.no_grad():
            y_ref2 = model(X2, ei2, ew2, ctx2, n_g)
        y_loaded2 = loaded(X2, ei2, ew2, ctx2)
        max_diff_other_B = (y_loaded2 - y_ref2).abs().max().item()

        print(f"[export_batched] n_g={n_g} V={V} .pt2={pkg_path} "
              f"sanity B={B_export}: max|diff|={max_diff_same_B:.3e}  "
              f"sanity B={B_sanity} (jamais vu): max|diff|={max_diff_other_B:.3e}")
        assert max_diff_same_B < 1e-4, f"reload incohérent (B={B_export}) pour n_g={n_g}"
        assert max_diff_other_B < 1e-4, (
            f"B dynamique NE généralise PAS correctement pour n_g={n_g} "
            f"(B={B_sanity} jamais vu à l'export) — export potentiellement figé sur B={B_export}"
        )


if __name__ == "__main__":
    main()
