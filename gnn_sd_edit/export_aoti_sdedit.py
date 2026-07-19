#!/usr/bin/env python3
"""
export_aoti_sdedit.py — exporte le VRAI modèle sd_edit (checkpoint de
production) en artefacts AOTInductor statiques, un par valeur de n_g
rencontrée. Appelé par GNNSdEditNativeAOTI (solver_bab/mkp/gnn_sd_edit_native_aoti.cpp)
via posix_spawn, même mécanisme que ensure_scripted_export dans
gnn_sd_edit_native.cpp — mis en cache sur disque, pas recompilé à chaque run.

Topologie sd_edit répliquée à l'identique de gnn_sd_edit_native.cpp
(étoile scénario-scénario ancrée au scénario 0) — les VALEURS de X/context
sont synthétiques (aléatoires), seule la FORME compte pour un export
AOTInductor ; les poids du modèle, eux, sont réels (chargés du checkpoint).

n_g est un int Python (pas un tenseur) : torch.export le fige en constante
dans le graphe exporté — un artefact statique par valeur de n_g rencontrée
en B&B (comme le pré-chauffage TorchScript existant), pas de shape dynamique.

Cache : <checkpoint>.aoti_N<n_z>/ng<K>.pt2, un fichier par n_g. Skip si déjà
présent et pas plus vieux que le checkpoint (même logique de staleness que
ensure_scripted_export). N_z fait partie du chemin de cache car la forme des
arêtes (donc le graphe FX exporté) en dépend, contrairement à
ensure_scripted_export qui n'a qu'une seule forme JIT-profilée à la volée.
"""
import argparse
import sys
from pathlib import Path

import torch

# Déterminisme requis : le propagateur Gecode qui consomme cette borne
# suppose des appels reproductibles bit-à-bit à état identique (contrat de
# monotonicité des propagateurs — cf. mémoire "aoti-native-integration-hang").
# Sans ceci, le forward AOTInductor compilé (kernels Triton, réductions par
# atomicAdd) jitte de ~1e-5 entre deux appels à données IDENTIQUES —
# suffisant pour faire osciller la borne entière (obj.max()) d'une unité
# après floor/scale, cassant la convergence du point fixe Gecode à la racine
# (TIMEOUT, 0 nœud — jamais un vrai plantage de query() elle-même, qui reste
# rapide et correcte en isolation).
torch.use_deterministic_algorithms(True)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from gnn.gnn_v1ld_scriptable import GNNScriptable  # noqa: E402


def build_sdedit_edges(n_g: int, N_z: int):
    src, dst, ew = [], [], []
    for i in range(N_z):
        for k1 in range(n_g):
            for k2 in range(k1 + 1, n_g):
                a = i * n_g + k1
                b = i * n_g + k2
                src += [a, b]
                dst += [b, a]
                ew += [1.0 / n_g, 1.0 / n_g]
    for kf in range(n_g):
        anchor = kf  # scénario 0
        for i2 in range(1, N_z):
            spoke = i2 * n_g + kf
            src += [anchor, spoke]
            dst += [spoke, anchor]
            ew += [1.0, 1.0]
    return src, dst, ew


def cache_dir_for(checkpoint: str, n_z: int) -> Path:
    return Path(f"{checkpoint}.aoti_N{n_z}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n-g", type=int, nargs="+", required=True)
    p.add_argument("--n-z", type=int, required=True)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_dir = cache_dir_for(args.checkpoint, args.n_z)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_mtime = ckpt_path.stat().st_mtime

    to_export = []
    for n_g in args.n_g:
        dest = out_dir / f"ng{n_g}.pt2"
        if dest.exists() and dest.stat().st_mtime >= ckpt_mtime:
            print(f"[export] n_g={n_g} : cache à jour ({dest}) — skip")
        else:
            to_export.append(n_g)

    if not to_export:
        print("[export] tous les n_g demandés sont déjà en cache.")
        return

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt["args"]
    m, hidden = int(ckpt_args["m"]), int(ckpt_args["hidden"])
    use_max_pool = bool(ckpt_args.get("use_max_pool", False))
    center_lam = bool(ckpt_args.get("center_lam", False))
    print(f"[export] checkpoint={args.checkpoint} m={m} hidden={hidden} "
          f"use_max_pool={use_max_pool} center_lam={center_lam} "
          f"n_g à exporter={to_export}")

    model = GNNScriptable(m, hidden, use_max_pool, center_lam)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval().to(args.device)

    node_dim = 7 * m + 5
    for n_g in to_export:
        N_z = args.n_z
        V = N_z * n_g
        src, dst, ew = build_sdedit_edges(n_g, N_z)
        E = len(src)

        X = torch.rand(V, node_dim, dtype=torch.float32, device=args.device)
        edge_index = torch.tensor([src, dst], dtype=torch.int64, device=args.device)
        edge_weight = torch.tensor(ew, dtype=torch.float32, device=args.device)
        context = torch.rand(9, dtype=torch.float32, device=args.device)

        with torch.no_grad():
            y_ref = model(X, edge_index, edge_weight, context, n_g)

        # aoti_compile_and_package (PAS aot_compile+package=False) : le
        # modèle utilise nn.GRUCell -> aten._thnn_fused_gru_cell, un op SANS
        # C-shim AOTInductor natif — nécessite le "proxy executor" (table de
        # correspondance JSON) que seule la voie "package" (.pt2) génère.
        # Le .so brut (aot_compile package=False) plante à l'exécution avec
        # "Unable to find a proxy executor" (découvert empiriquement, cf.
        # mémoire "aoti-sdedit-wallclock-result").
        ep = torch.export.export(model, (X, edge_index, edge_weight, context, n_g))
        dest = out_dir / f"ng{n_g}.pt2"
        pkg_path = torch._inductor.aoti_compile_and_package(
            ep, package_path=str(dest),
        )

        loaded = torch._inductor.aoti_load_package(pkg_path)
        y_loaded = loaded(X, edge_index, edge_weight, context, n_g)
        max_diff = (y_loaded - y_ref).abs().max().item()
        print(f"[export] n_g={n_g} V={V} E={E} .pt2={pkg_path} "
              f"sanity max|diff|={max_diff:.3e}")
        assert max_diff < 1e-4, f"reload incohérent pour n_g={n_g}"


if __name__ == "__main__":
    main()
