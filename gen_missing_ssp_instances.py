"""gen_missing_ssp_instances.py — Génère les instances SSP manquantes pour
run_bench_ssp_bab_file_multi.sh (format texte 4-fichiers attendu par
ssp_loader.hpp, cf. export_to_txt.write_instance).

Idempotent : saute une combinaison si son dossier existe déjà. N'échoue pas
sur une combinaison qui ne calibre pas (RuntimeError de generate_episode_fixed_E,
cf. calibrate_adaptive.py) — log l'échec et continue avec les suivantes.

Usage (venv learning_bounds activé) :
    python gen_missing_ssp_instances.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "solver_generic" / "ssp"))

from ssp_stochastic.generator_adaptive import generate_episode_fixed_E  # noqa: E402
from export_to_txt import write_instance  # noqa: E402

PROBE_BIN = REPO / "solver_bab" / "build" / "ssp_probe"

COMBOS = [
    (3, 15, 50),
    (4, 8, 50),
    (4, 15, 50),
]

for E, T, N in COMBOS:
    out_dir = REPO / "solver_bab" / "ssp" / "instances" / f"E{E}_T{T}_N{N}_seed42"
    if (out_dir / "instance.txt").exists():
        print(f"[SKIP] {out_dir} existe déjà", flush=True)
        continue
    print(f"=== Génération E={E} T={T} N={N} -> {out_dir} ===", flush=True)
    try:
        ep = generate_episode_fixed_E(
            ep_id=0, E=E, T=T, N=N, epsilon=0.10,
            probe_bin=PROBE_BIN, master_seed=42,
        )
    except RuntimeError as e:
        print(f"[ECHEC] calibration E={E} T={T} N={N} : {e}", flush=True)
        continue
    write_instance(ep, out_dir)
    print(f"[OK] -> {out_dir}  (scale={ep['meta']['scale']:.4f}, "
          f"states={ep['dfa'].num_states}, attempts={ep['meta']['automaton_attempt']+1})",
          flush=True)
