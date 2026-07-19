#!/bin/bash
# run_bench_v1ld_subproblems.sh — Benchmark élémentaire du coût des sous-
# problèmes θ/ζ d'UNE itération v1LD (MKP), λ=0, racine, sans B&B.
#
# Compare 3 méthodes de calcul de ζ : cpsat (référence) / bnb (production
# actuelle) / surr (borne surrogate). Une 4e piste (lp, borne LP jointe GLOP
# — cf. mkp_problem_bnb_lp.cpp) a été testée puis écartée : correcte mais
# ~44x plus lente que bnb (job 15985510) — retirée de la comparaison active,
# code conservé. Rebuild la cible avant de lancer (source modifiée) plutôt
# que d'exiger un compile_cc.sh séparé au préalable.
#
# Mono-thread par construction (CP-SAT appelé avec num_workers=1 dans
# solve_zeta, cf. solver_generic/mkp/mkp_problem.cpp) — 1 CPU suffit à
# l'exécution ; on garde 2 CPUs pour que le rebuild (cmake --build -j2) ne
# soit pas strictement mono-thread.
#
# Soumettre :
#   sbatch run_bench_v1ld_subproblems.sh
#
# Varier le manifest / epsilon :
#   sbatch --export=ALL,EPS=0.05 run_bench_v1ld_subproblems.sh
#   sbatch --export=ALL,MANIFEST=solver_bab/mkp/instances_v1ld_bench/manifest.txt run_bench_v1ld_subproblems.sh

# ── Ressources SLURM ──────────────────────────────────────────────────────────
#SBATCH --account=def-qcappart
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=bench_v1ld_subproblems
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0:30:00
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err

set -euo pipefail

# ── Paramètres ────────────────────────────────────────────────────────────────
MANIFEST="${MANIFEST:-solver_bab/mkp/instances_v1ld_bench/manifest.txt}"
EPS="${EPS:-0.10}"
# refdir : ground truth (theta_value + N zeta_value, précision totale) par
# instance — sert à vérifier la reproductibilité et à comparer une future
# méthode de calcul de v1LD (doit retrouver ces mêmes valeurs).
REFDIR="${REFDIR:-solver_bab/mkp/instances_v1ld_bench/reference}"

# ── Chemins ───────────────────────────────────────────────────────────────────
REPO="${SLURM_SUBMIT_DIR:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"
BAB_BUILD="$REPO/solver_bab/build"
BINARY="$BAB_BUILD/bench_v1ld_subproblems"

# ── Modules ───────────────────────────────────────────────────────────────────
module load StdEnv/2023
module load gcc/12.3
module load cmake/3.27.7

export LD_LIBRARY_PATH="$REPO/external/gecode/lib:$REPO/external/ortools/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$REPO/logs"

echo "======================================================================"
echo "  bench_v1ld_subproblems — $(date)"
echo "  Job : ${SLURM_JOB_ID:-local}   Nœud : $(hostname)"
echo "  Manifest : $MANIFEST   epsilon=$EPS   refdir=$REFDIR"
echo "======================================================================"

mkdir -p "$REPO/$REFDIR"

# ── Rebuild (source modifiée : ajout de la méthode lp) ───────────────────────
echo ""
echo "=== Rebuild bench_v1ld_subproblems ==="
cmake --build "$BAB_BUILD" --target bench_v1ld_subproblems -j2 2>&1

[[ -x "$BINARY" ]] || { echo "ERREUR : binaire introuvable après rebuild : $BINARY"; exit 1; }

# ── Run ────────────────────────────────────────────────────────────────────────
echo ""
"$BINARY" "$REPO/$MANIFEST" "$EPS" "$REPO/$REFDIR"

echo ""
echo "======================================================================"
echo "  Terminé — $(date)"
echo "======================================================================"
