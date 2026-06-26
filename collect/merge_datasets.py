"""
collect/merge_datasets.py — Fusionne plusieurs data/train_N{N}/ en un seul répertoire.

Re-indexe les inst_ids pour éviter les collisions entre sources.
N=500 utilise data/train_base_s30/ si ce répertoire existe (données existantes).

Usage :
  python collect/merge_datasets.py \
      --n_scens 50 100 200 500 \
      --out_dir /tmp/merged/ \
      [--n_val 10] [--m_val 5] [--data_root data/]
"""
import argparse
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_scens",   type=int, nargs="+", required=True,
                   help="Liste de valeurs N (scénarios) à fusionner.")
    p.add_argument("--out_dir",   type=Path, required=True)
    p.add_argument("--n_val",     type=int,  default=10)
    p.add_argument("--m_val",     type=int,  default=5)
    p.add_argument("--data_root", type=Path, default=Path("data"))
    return p.parse_args()


def src_dir_for(n_scen, data_root):
    if n_scen == 500:
        candidate = data_root / "train_base_s30"
        if candidate.exists():
            return candidate
    return data_root / f"train_N{n_scen}"


def merge_file(suffix, src_dirs, out_dir):
    out_path = out_dir / suffix
    offset = 0
    total = 0
    with open(out_path, "w") as fout:
        for n_scen, src in src_dirs:
            src_file = src / suffix
            if not src_file.exists():
                print(f"  [WARN] {src_file} introuvable — N={n_scen} ignoré", file=sys.stderr)
                continue
            max_id = -1
            count = 0
            with open(src_file) as fin:
                for line in fin:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    parts = line.split(",")
                    new_id = int(parts[0]) + offset
                    parts[0] = str(new_id)
                    max_id = max(max_id, new_id)
                    fout.write(",".join(parts) + "\n")
                    count += 1
            if max_id >= 0:
                offset = max_id + 1
            total += count
            print(f"  N={n_scen:4d}: {count:7d} lignes  ({suffix})")
    print(f"  → {out_path} : {total} lignes au total")


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    src_dirs = [(n, src_dir_for(n, args.data_root)) for n in args.n_scens]

    print(f"Sources :")
    for n, d in src_dirs:
        print(f"  N={n:4d} → {d}")

    for suffix in [f"instances_n{args.n_val}_m{args.m_val}.txt",
                   f"nodes_n{args.n_val}_m{args.m_val}.txt"]:
        print(f"\n=== Fusion : {suffix} ===")
        merge_file(suffix, src_dirs, args.out_dir)


if __name__ == "__main__":
    main()
