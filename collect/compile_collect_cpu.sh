#!/bin/bash
# collect/compile_collect_cpu.sh — Compile bab_mkp_gecode_base_collect dans un
# arbre de build CPU-only DÉDIÉ (solver_bab/build_collect_cpu), séparé de
# solver_bab/build (qui reste en GNN_DEVICE=cuda pour le binaire de
# production bab_mkp_gecode utilisé sur GPU H100).
#
# Contexte : solver_bab/build a été reconfiguré en cuda le 2026-07-14
# (validation native_aoti), ce qui a aussi relinké bab_mkp_gecode_base_collect
# contre libtorch-cuda — cassé sur les nœuds CPU (cpubase_bycore_b4, pas de
# CUDA) alors que ce binaire n'active jamais bound_mode=GNN. Un seul dossier
# de build = une seule valeur de GNN_DEVICE pour toutes les cibles ; on ne
# veut pas reconfigurer solver_bab/build en cpu (casserait le binaire GPU de
# prod), donc nouvel arbre séparé.
#
# Réutilise les dépendances déjà installées (Gecode, OR-Tools, LibTorch CPU,
# libsld_generic_core.a) — ne recompile que gecode_bab.cpp et les fichiers
# GNN natifs pour bab_mkp_gecode_base_collect.
#
# Soumission :
#   sbatch collect/compile_collect_cpu.sh

#SBATCH --account=def-qcappart_cpu
#SBATCH --partition=cpubase_bycore_b4
#SBATCH --chdir=/project/6062881/romanlen/Learning_bounds_CCCP
#SBATCH --job-name=compile_collect_cpu
#SBATCH --output=logs/compile_collect_cpu_%j.out
#SBATCH --error=logs/compile_collect_cpu_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXT="$REPO/external"

module load StdEnv/2023
module load gcc/12.3
module load cmake/3.27.7
module load python/3.11
module load scipy-stack

GENERIC_BUILD="$REPO/solver_generic/build_cc"
BAB_BUILD="$REPO/solver_bab/build_collect_cpu"
mkdir -p "$BAB_BUILD"

echo "=== Configuration (GNN_DEVICE=cpu, arbre dédié) ==="
cmake -S "$REPO/solver_bab" -B "$BAB_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$EXT/ortools" \
    -DGECODE_PREFIX="$EXT/gecode" \
    -DSOLVER_GENERIC_BUILD="$GENERIC_BUILD" \
    -DGNN_DEVICE=cpu

echo ""
echo "=== Compilation bab_mkp_gecode_base_collect ==="
cmake --build "$BAB_BUILD" --target bab_mkp_gecode_base_collect -j4

[[ -f "$BAB_BUILD/bab_mkp_gecode_base_collect" ]] || \
    { echo "ERREUR : binaire non généré"; exit 1; }

echo ""
echo "======================================================================"
echo "  Compilation réussie : $BAB_BUILD/bab_mkp_gecode_base_collect"
echo "======================================================================"
