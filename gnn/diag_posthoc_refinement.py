"""
Test post-hoc (diagnostic uniquement, pas pour la production) : le lambda
appris par le GNN (one-shot) est-il un bon point de depart pour quelques
iterations de sous-gradient, comparees a un cold-start (lam=0) ?
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


def main():
    lam_dump = torch.load(DUMP, weights_only=False)
    print(f"Total entries in dump: {len(lam_dump)}")

    depth0 = [rec for rec in lam_dump.values() if rec["depth"] == 0 and rec["N"] == 500 and rec["N_z"] == 500]
    print(f"depth=0, N=N_z=500 entries: {len(depth0)}")
    sample = depth0[:15]

    Ks = [5, 10, 20, 50]
    results = {"cold_T50": [], "cold_T1000": [], "opt": [], "raw_gnn": []}
    for K in Ks:
        results[f"warm_T{K}"] = []

    for idx, rec in enumerate(sample):
        data = FakeData(rec)
        lam_gnn = rec["lam_2d"].astype(np.float64)
        lam_gnn_t = torch.from_numpy(lam_gnn).view(-1)
        raw_gnn = lagrangian_v1ld(lam_gnn_t, data).item()

        cold_50   = verify_v1ld(data, T=50)
        cold_1000 = verify_v1ld(data, T=1000)
        opt_val = rec["opt"]

        row = f"  inst {idx+1}/{len(sample)}  opt={opt_val:.1f}  raw_gnn={raw_gnn:.2f}  cold_T50={cold_50:.2f}  cold_T1000={cold_1000:.2f}"
        for K in Ks:
            warm = verify_v1ld(data, T=K, lam_init=lam_gnn.copy())
            results[f"warm_T{K}"].append(warm)
            row += f"  warm_T{K}={warm:.2f}"
        results["cold_T50"].append(cold_50)
        results["cold_T1000"].append(cold_1000)
        results["opt"].append(opt_val)
        results["raw_gnn"].append(raw_gnn)
        print(row, flush=True)

    print("\n=== Moyennes (n={}) ===".format(len(sample)))
    cold50_mean = np.mean(results["cold_T50"])
    cold1000_mean = np.mean(results["cold_T1000"])
    raw_gnn_mean = np.mean(results["raw_gnn"])
    n_raw_beats_cold50 = np.sum(np.array(results["raw_gnn"]) < np.array(results["cold_T50"]))
    print(f"  raw GNN (0 iter) : {raw_gnn_mean:.2f}   (bat cold_T50 sur {n_raw_beats_cold50}/{len(sample)} instances)")
    print(f"  cold T=50   : {cold50_mean:.2f}")
    print(f"  cold T=1000 : {cold1000_mean:.2f}")
    for K in Ks:
        warm_mean = np.mean(results[f"warm_T{K}"])
        n_beats_cold50 = np.sum(np.array(results[f"warm_T{K}"]) < np.array(results["cold_T50"]))
        print(f"  warm  T={K:<4}: {warm_mean:.2f}   (bat cold_T50 sur {n_beats_cold50}/{len(sample)} instances)")


if __name__ == "__main__":
    main()
