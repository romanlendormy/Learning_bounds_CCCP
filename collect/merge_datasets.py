"""
collect/merge_datasets.py — Fusionne plusieurs data/train_N{N}/ en un seul répertoire.

Re-indexe les inst_ids pour éviter les collisions entre sources.
N=500 utilise data/train_base_s30/ si ce répertoire existe (données existantes).

Filtre automatiquement les instances par N cible quand la source est un répertoire
mixte (ex: train_base_s30 contient N=50/250/500 — seul N=500 est extrait pour N=500).

Usage :
  python collect/merge_datasets.py \
      --n_scens 100 200 500 \
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


def valid_inst_ids_for_N(inst_file: Path, target_N: int) -> set | None:
    """Retourne les inst_ids dont N == target_N. None si le fichier est introuvable."""
    if not inst_file.exists():
        return None
    valid = set()
    with open(inst_file) as f:
        for line in f:
            vals = line.strip().split(",")
            if len(vals) > 3 and int(vals[3]) == target_N:
                valid.add(int(vals[0]))
    return valid


def merge_file(suffix, src_dirs, out_dir, valid_ids_by_src):
    out_path = out_dir / suffix
    offset = 0
    total = 0
    with open(out_path, "w") as fout:
        for (n_scen, src), valid_ids in zip(src_dirs, valid_ids_by_src):
            src_file = src / suffix
            if not src_file.exists():
                print(f"  [WARN] {src_file} introuvable — N={n_scen} ignoré", file=sys.stderr)
                continue
            max_id = -1
            count = 0
            skipped = 0
            with open(src_file) as fin:
                for line in fin:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    parts = line.split(",")
                    orig_id = int(parts[0])
                    if valid_ids is not None and orig_id not in valid_ids:
                        skipped += 1
                        continue
                    new_id = orig_id + offset
                    parts[0] = str(new_id)
                    max_id = max(max_id, new_id)
                    fout.write(",".join(parts) + "\n")
                    count += 1
            if max_id >= 0:
                offset = max_id + 1
            total += count
            skip_msg = f"  ({skipped} filtrés)" if skipped else ""
            print(f"  N={n_scen:4d}: {count:7d} lignes  ({suffix}){skip_msg}")
    print(f"  → {out_path} : {total} lignes au total")


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    src_dirs = [(n, src_dir_for(n, args.data_root)) for n in args.n_scens]

    print(f"Sources :")
    for n, d in src_dirs:
        print(f"  N={n:4d} → {d}")

    inst_suffix = f"instances_n{args.n_val}_m{args.m_val}.txt"

    # Construire les filtres d'inst_ids par source (filtre N sur sources mixtes)
    valid_ids_by_src = []
    for n_scen, src in src_dirs:
        inst_file = src / inst_suffix
        valid_ids = valid_inst_ids_for_N(inst_file, n_scen)
        n_valid = len(valid_ids) if valid_ids is not None else "?"
        print(f"  N={n_scen:4d}: {n_valid} instances valides dans {src.name}")
        valid_ids_by_src.append(valid_ids)

    for suffix in [inst_suffix, f"nodes_n{args.n_val}_m{args.m_val}.txt"]:
        print(f"\n=== Fusion : {suffix} ===")
        merge_file(suffix, src_dirs, args.out_dir, valid_ids_by_src)

    # Fichiers per-N pour chargement stratifié efficace en mémoire
    print(f"\n=== Fichiers per-N (chargement stratifié) ===")
    nodes_suffix = f"nodes_n{args.n_val}_m{args.m_val}.txt"
    for (n_scen, src), valid_ids in zip(src_dirs, valid_ids_by_src):
        for suffix in [inst_suffix, nodes_suffix]:
            n_suffix = suffix.replace(f"_m{args.m_val}.txt",
                                      f"_m{args.m_val}_N{n_scen}.txt")
            out_path = args.out_dir / n_suffix
            src_file = src / suffix
            if not src_file.exists():
                continue
            count = 0
            with open(out_path, "w") as fout, open(src_file) as fin:
                for line in fin:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    if valid_ids is not None and int(line.split(",")[0]) not in valid_ids:
                        continue
                    fout.write(line + "\n")
                    count += 1
            print(f"  N={n_scen:4d}: {count:7d} lignes → {n_suffix}")


if __name__ == "__main__":
    main()
