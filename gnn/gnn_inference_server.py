#!/usr/bin/env python3
"""
gnn_inference_server.py — Serveur d'inférence GNN one-shot pour bab_mkp_gecode.

Process persistant, lancé une fois par bab_mkp_gecode (mode_bound=GNN) via
pipe()+fork()+execvp() (cf. solver_bab/mkp/gnn_v1ld_client.cpp), qui parle un
protocole binaire à taille fixe sur stdin/stdout (pas de JSON/pickle — on vise
l'efficacité, chaque octet de sérialisation compte sur des dizaines de
milliers d'appels par résolution B&B).

Séquence :
  1. INIT (une fois)  : n, m, N (int32), epsilon (f64), c[n], p[N],
                         A[N*m*n], b[N*m] (f64, row-major) → l'instance
                         complète du problème CCMKP résolu par le B&B.
     → ACK : status (int32, 0=ok) + msg_len (int32) + msg (utf-8, msg_len octets)
  2. Boucle : requête nœud = x_mask[n] (int8 : 0=fixé_0, 1=fixé_1, 2=libre)
              + curr_cost (f64, profit déjà fixé = fp) + depth (int32)
              → réponse = status (int32) + ub_relative (f64)
                          + t_graph_build_s + t_forward_s + t_bound_compute_s (f64)
     ub_relative est L1(λ) sur le sous-problème résiduel (mêmes conventions
     que v1LD_star côté C++) ; l'appelant calcule ub = fp + ub_relative.
  3. EOF sur stdin (pipe d'écriture fermé côté C++) → sortie propre.

z_mask n'est jamais transmis : le more() C++ actuel ne fixe jamais aucun
scénario (même epsilon/N à chaque nœud), donc on le fixe ici en interne à
"toujours libre" (Format B de build_graph, cf. train_gnn_v1ld.py) —
fidèle au comportement du solveur sous-gradient qu'on remplace.

Restriction connue : lagrangian_v1ld() suppose p_i = 1/N (scénarios
équiprobables), codé en dur dans son calcul de p_unif. On le vérifie à l'INIT
et on refuse (erreur explicite) une instance à probabilités non-uniformes
plutôt que de renvoyer silencieusement une borne invalide.
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from gnn.train_gnn_v1ld import GNN, build_graph, lagrangian_v1ld  # noqa: E402
from gnn_sd_edit.train_v1LD_sd_edit import (  # noqa: E402
    build_graph as build_graph_sd_edit,
    lagrangian_v1ld as lagrangian_v1ld_sd_edit,
)

INIT_HEADER_FMT = "<iiid"
INIT_HEADER_SIZE = struct.calcsize(INIT_HEADER_FMT)
ACK_FMT = "<ii"
NODE_RESP_FMT = "<idddd"
NODE_RESP_SIZE = struct.calcsize(NODE_RESP_FMT)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Chemin vers le checkpoint .pt")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    # "homogene" : arêtes scénario = clique (gnn.train_gnn_v1ld.build_graph).
    # "sd_edit"  : arêtes scénario = étoile ancrée sur le scénario 0
    #   (gnn_sd_edit.train_v1LD_sd_edit.build_graph) — seule la construction du
    #   graphe change, l'architecture GNN (et donc le state_dict) est identique
    #   dans les deux cas (vérifié : mêmes noms/shapes de paramètres).
    p.add_argument("--variant", default="homogene", choices=["homogene", "sd_edit"])
    return p.parse_args()


def read_exact(stream, nbytes: int) -> bytes:
    """Lit exactement nbytes depuis stream (les pipes peuvent renvoyer des
    lectures partielles). Retourne b"" si EOF avant tout octet (fin propre) ;
    lève EOFError si l'EOF survient au milieu d'un message (protocole rompu)."""
    chunks = []
    remaining = nbytes
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            if not chunks:
                return b""
            raise EOFError(f"EOF inattendu : {remaining}/{nbytes} octets manquants")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_ack(stream, status: int, msg: str = "") -> None:
    msg_b = msg.encode("utf-8")
    stream.write(struct.pack(ACK_FMT, status, len(msg_b)))
    stream.write(msg_b)
    stream.flush()


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "args" not in ckpt:
        raise ValueError(f"Checkpoint {checkpoint_path} sans clé 'args'.")
    ckpt_args = ckpt["args"]
    model = GNN(
        m=ckpt_args["m"],
        hidden=ckpt_args["hidden"],
        use_max_pool=ckpt_args.get("use_max_pool", False),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt_args


def apply_center(lam: torch.Tensor, N_z: int, n_g: int) -> torch.Tensor:
    lam_2d = lam.view(N_z, n_g)
    return (lam_2d - lam_2d.mean(dim=0, keepdim=True)).reshape(-1)


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)

    if args.variant == "sd_edit":
        build_graph_fn, lagrangian_v1ld_fn = build_graph_sd_edit, lagrangian_v1ld_sd_edit
    else:
        build_graph_fn, lagrangian_v1ld_fn = build_graph, lagrangian_v1ld

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    try:
        model, ckpt_args = load_model(args.checkpoint, device)
        center_lam = bool(ckpt_args.get("center_lam", False))
    except Exception as e:  # boundary : erreur de chargement modèle, remontée avant l'INIT
        print(f"[gnn_inference_server] échec chargement modèle : {e}", file=sys.stderr)
        return 1

    header = read_exact(stdin, INIT_HEADER_SIZE)
    if not header:
        return 0  # C++ a fermé le pipe avant d'envoyer quoi que ce soit
    n, m, N, epsilon = struct.unpack(INIT_HEADER_FMT, header)

    if m != ckpt_args["m"]:
        send_ack(stdout, 1, f"mismatch m : instance m={m}, checkpoint entraîné avec m={ckpt_args['m']}")
        return 1

    c = np.frombuffer(read_exact(stdin, 8 * n), dtype="<f8").copy()
    p = np.frombuffer(read_exact(stdin, 8 * N), dtype="<f8").copy()
    A = np.frombuffer(read_exact(stdin, 8 * N * m * n), dtype="<f8").copy().reshape(N, m, n)
    b = np.frombuffer(read_exact(stdin, 8 * N * m), dtype="<f8").copy().reshape(N, m)

    if not np.allclose(p, 1.0 / N, atol=1e-9):
        send_ack(stdout, 1, "p non-uniforme : lagrangian_v1ld suppose p_i = 1/N (scénarios équiprobables)")
        return 1

    # float32 : convention de load_instances() (pipeline d'entraînement normal),
    # requise pour que c_g (utilisé tel quel dans GNN.forward en mode
    # use_theta_attn) ait le même dtype que les poids du modèle. A/b sont
    # re-castés en float64 plus loin, à l'intérieur de lagrangian_v1ld, pour
    # l'appel au solveur ζ (bnb_solver, interface ctypes en double).
    c = c.astype(np.float32)
    A = A.astype(np.float32)
    b = b.astype(np.float32)

    inst = {"n": n, "m": m, "N": N, "c": c, "b": b, "A": A, "epsilon": epsilon, "opt": 0.0}
    z_mask_full = np.full(N, 2, dtype=np.int8)  # toujours libre — cf. docstring module

    send_ack(stdout, 0)

    with torch.no_grad():
        while True:
            req = read_exact(stdin, n + 8 + 4)
            if not req:
                break  # EOF propre : le B&B C++ a fermé le pipe, résolution terminée

            x_mask = np.frombuffer(req[:n], dtype=np.int8)
            curr_cost, depth = struct.unpack_from("<di", req, n)

            try:
                t0 = time.perf_counter()
                fixed_one = (x_mask == 1).astype(np.float32)
                b_res = b - np.einsum("k,ijk->ij", fixed_one, A)
                node_rec = {
                    "x_mask": x_mask, "z_mask": z_mask_full, "b_res": b_res,
                    "curr_cost": curr_cost, "depth": depth,
                }
                g = build_graph_fn(inst, node_rec)
                t1 = time.perf_counter()

                if g is None:
                    # Toutes les variables sont fixées (feuille) : borne résiduelle nulle.
                    ub_relative, t_forward, t_bound = 0.0, 0.0, 0.0
                else:
                    g = g.to(device)
                    lam = model(g)
                    if center_lam:
                        lam = apply_center(lam, g.N_z, g.n_g)
                    t2 = time.perf_counter()
                    ub_relative = lagrangian_v1ld_fn(lam, g).item()
                    t3 = time.perf_counter()
                    t_forward, t_bound = t2 - t1, t3 - t2

                t_graph = t1 - t0
                stdout.write(struct.pack(NODE_RESP_FMT, 0, ub_relative, t_graph, t_forward, t_bound))
                stdout.flush()
            except Exception as e:
                # Boundary : ne jamais laisser une exception Python planter le
                # process sans réponse — le C++ attendrait indéfiniment sur le
                # pipe. On renvoie un statut d'erreur (le client C++ lève une
                # exception à réception) et on trace côté stderr pour le debug.
                print(f"[gnn_inference_server] erreur au nœud (depth={depth}): {e}", file=sys.stderr)
                stdout.write(struct.pack(NODE_RESP_FMT, 1, 0.0, 0.0, 0.0, 0.0))
                stdout.flush()

    return 0


if __name__ == "__main__":
    sys.exit(main())
