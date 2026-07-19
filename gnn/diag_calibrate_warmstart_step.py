"""
Calibration du pas de sous-gradient pour un warm-start court (K=10-15
iterations), point de depart de train_gnn_v1ld_warmstart.py.

Diagnostic uniquement : compare, a K fixe, le schema de pas "sqrt" (1/sqrt(t+1),
historique, celui reellement utilise en B&B aujourd'hui) a "fixed" (plusieurs
constantes) et "polyak" (pas = (L1-v_lower)/||g||^2, avec v_lower = data.opt
comme approximation -- non disponible ainsi en B&B reel, ou l'incumbent jouerait
ce role de facon moins precise).

Objectif : choisir un schema de pas AVANT de le brancher dans la loss
d'entrainement K-step, pour ne pas confondre "le K-step training ne marche
pas" et "le pas est mal calibre" (meme piege que clip-gradient/c_scale,
cf. memoire projet).
"""
import sys
sys.path.insert(0, "/lustre09/project/6062881/romanlen/Learning_bounds_CCCP/gnn")

import numpy as np
import torch
torch.set_num_threads(4)

from train_gnn_v1ld import verify_v1ld, lagrangian_v1ld

REPO = "/lustre09/project/6062881/romanlen/Learning_bounds_CCCP"
DUMP = f"{REPO}/gnn/runs/ablation_hidden/ckpt_N500_h64/lambdas_final.pt"


class FakeData:
    def __init__(self, rec):
        self.N_z        = rec["N_z"]
        self.n_g        = rec["n_g"]
        self.N          = rec["N"]
        self.n_fixed_z1 = rec["n_fixed_z1"]
        self.epsilon    = torch.tensor(rec["epsilon"])
        self.A_g        = torch.from_numpy(rec["A_g"])
        self.b_res_g    = torch.from_numpy(rec["b_res_g"])
        self.c_g        = torch.from_numpy(rec["c_g"])
        self.c_scale    = rec["c_scale"]
        self.opt        = rec["opt"]


SCHEMES = [
    ("sqrt_t0=1",   dict(step_mode="sqrt", step_offset=1.0)),    # historique / prod actuelle
    ("sqrt_t0=10",  dict(step_mode="sqrt", step_offset=10.0)),   # 1er pas amorti (~0.32)
    ("sqrt_t0=50",  dict(step_mode="sqrt", step_offset=50.0)),   # 1er pas amorti (~0.14)
    ("fixed0.02", dict(step_mode="fixed", step_value=0.02)),
    ("fixed0.05", dict(step_mode="fixed", step_value=0.05)),
    ("fixed0.1",  dict(step_mode="fixed", step_value=0.1)),
    ("fixed0.15", dict(step_mode="fixed", step_value=0.15)),
    ("fixed0.2",  dict(step_mode="fixed", step_value=0.2)),
    ("fixed0.3",  dict(step_mode="fixed", step_value=0.3)),
    ("polyak_opt", dict(step_mode="polyak")),  # v_lower branche par instance (data.opt)
]


def main():
    lam_dump = torch.load(DUMP, weights_only=False)
    depth0 = [rec for rec in lam_dump.values()
              if rec["depth"] == 0 and rec["N"] == 500 and rec["N_z"] == 500]
    sample = depth0[:15]
    print(f"{len(sample)} instances (depth=0, N=N_z=500)")

    Ks = [10, 15]
    # results[K][scheme_name] = liste des L1 (une par instance)
    results = {K: {name: [] for name, _ in SCHEMES} for K in Ks}
    raw_gnn_list, cold50_list = [], []

    for idx, rec in enumerate(sample):
        data = FakeData(rec)
        lam_gnn = rec["lam_2d"].astype(np.float64)
        lam_gnn_t = torch.from_numpy(lam_gnn).view(-1)
        raw_gnn = lagrangian_v1ld(lam_gnn_t, data).item()
        cold_50 = verify_v1ld(data, T=50)
        raw_gnn_list.append(raw_gnn)
        cold50_list.append(cold_50)

        row = f"  inst {idx+1:2d}/{len(sample)}  opt={data.opt:8.1f}  raw_gnn={raw_gnn:9.2f}  cold_T50={cold_50:9.2f}"
        for K in Ks:
            for name, kwargs in SCHEMES:
                kw = dict(kwargs)
                if kw.get("step_mode") == "polyak":
                    kw["v_lower"] = float(data.opt)
                warm = verify_v1ld(data, T=K, lam_init=lam_gnn.copy(), **kw)
                results[K][name].append(warm)
        print(row, flush=True)

    raw_gnn_mean = np.mean(raw_gnn_list)
    cold50_mean  = np.mean(cold50_list)
    print(f"\nraw GNN (0 iter) moyenne : {raw_gnn_mean:.2f}")
    print(f"cold T=50 moyenne        : {cold50_mean:.2f}  (reference)")

    for K in Ks:
        print(f"\n=== K={K} ===")
        for name, _ in SCHEMES:
            vals = np.array(results[K][name])
            n_beats_cold50 = int(np.sum(vals < np.array(cold50_list)))
            n_beats_raw    = int(np.sum(vals < np.array(raw_gnn_list)))
            print(f"  {name:12s}: moyenne={vals.mean():9.2f}  "
                  f"bat cold_T50 sur {n_beats_cold50}/{len(sample)}  "
                  f"bat raw_gnn sur {n_beats_raw}/{len(sample)}")


if __name__ == "__main__":
    main()
