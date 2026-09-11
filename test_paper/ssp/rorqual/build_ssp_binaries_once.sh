#!/bin/bash
# build_ssp_binaries_once.sh — Compile UNE SEULE FOIS, de façon sérialisée,
# les deux binaires utilisés par la campagne test_paper/ssp/rorqual/ssp
# (test_gnn_dld_bab, test_gnn_dld_bab_batched) dans solver_bab/build_gpu_ssp.
#
# Existe suite à l'incident du 2026-08-28 : run_ref_array_ssp.sh et
# run_gnn_batched_array_ssp.sh faisaient CHACUN leur propre "cmake --build"
# dans ce MÊME répertoire de build partagé. Quand plusieurs tâches d'array
# démarrent en même temps (typiquement après une file d'attente sur les
# nœuds GPU), leurs builds concurrents se marchent dessus : un lien qui
# échoue dans une tâche supprime le binaire qu'une autre tâche vient de
# produire avec succès. Résultat mesuré : 20 tâches FAILED d'un coup
# (erreur de liaison "undefined reference to solve_sld/solve_dld") + 1
# résultat dégradé (root_bound_value manquant, binaire disparu sous le run
# auxiliaire en cours d'exécution).
#
# Correctif : compiler une seule fois ICI (interactif ou petit job CPU),
# puis les deux scripts d'array (run_ref_array_ssp.sh,
# run_gnn_batched_array_ssp.sh) se contentent de VÉRIFIER que les binaires
# existent -- ils échouent bruyamment sinon, sans jamais retenter un build.
#
# Usage :
#   bash test_paper/ssp/rorqual/build_ssp_binaries_once.sh

set -euo pipefail

REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../.." && pwd )"
cd "$REPO"

module load StdEnv/2023
module load gcc/12.3
module load cmake/3.27.7

GENERIC_BUILD="$REPO/solver_generic/build_cc"
BAB_BUILD="$REPO/solver_bab/build_gpu_ssp"
mkdir -p "$BAB_BUILD"

echo "=== Configuration (GNN_DEVICE=cuda) ==="
cmake -S "$REPO/solver_bab" -B "$BAB_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$REPO/external/ortools" \
    -DGECODE_PREFIX="$REPO/external/gecode" \
    -DSOLVER_GENERIC_BUILD="$GENERIC_BUILD" \
    -DGNN_DEVICE=cuda

echo ""
echo "=== Compilation test_gnn_dld_bab_batched ==="
cmake --build "$BAB_BUILD" --target test_gnn_dld_bab_batched -j4

echo ""
echo "=== Compilation test_gnn_dld_bab ==="
cmake --build "$BAB_BUILD" --target test_gnn_dld_bab -j4

BIN_BATCHED="$BAB_BUILD/test_gnn_dld_bab_batched"
BIN_SEQ="$BAB_BUILD/test_gnn_dld_bab"

echo ""
if [[ -x "$BIN_BATCHED" && -x "$BIN_SEQ" ]]; then
    echo "OK -- les deux binaires sont présents et exécutables :"
    ls -la "$BIN_BATCHED" "$BIN_SEQ"
else
    echo "ERREUR : un des deux binaires manque après compilation." >&2
    exit 1
fi
