#!/bin/bash
# compile_cc.sh — Compilation de bab_mkp_gecode sur Compute Canada (Nibi)
#
# Usage dans une session interactive :
#   salloc --account=def-qcappart --cpus-per-task=4 --mem=8G --time=2:00:00
#   bash compile_cc.sh
#
# Si le nœud n'a pas internet, télécharger les archives D'ABORD depuis le login :
#   bash compile_cc.sh --download-only
# Puis relancer normalement dans la session interactive.

set -euo pipefail

REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXT="$REPO/external"

GECODE_VERSION="6.2.0"
GECODE_INSTALL="$EXT/gecode"
GECODE_URL="https://github.com/Gecode/gecode/archive/refs/tags/release-${GECODE_VERSION}.tar.gz"

ORTOOLS_VERSION="9.6.2534"
ORTOOLS_INSTALL="$EXT/ortools"
ORTOOLS_URL="https://github.com/google/or-tools/releases/download/v9.6/or-tools_amd64_ubuntu-20.04_cpp_v${ORTOOLS_VERSION}.tar.gz"

DOWNLOAD_ONLY="${1:-}"

# ── Téléchargements (faisables depuis login node) ──────────────────────────────
echo "=== Téléchargements ==="
mkdir -p "$EXT"

GECODE_TAR="$EXT/gecode-${GECODE_VERSION}.tar.gz"
if [[ ! -f "$GECODE_TAR" ]]; then
    echo "Téléchargement Gecode ${GECODE_VERSION}..."
    wget -q --show-progress -O "$GECODE_TAR" "$GECODE_URL" || \
    curl -fL --progress-bar -o "$GECODE_TAR" "$GECODE_URL"
fi
echo "[OK] $GECODE_TAR"

ORTOOLS_TAR="$EXT/ortools-${ORTOOLS_VERSION}.tar.gz"
if [[ ! -f "$ORTOOLS_TAR" ]]; then
    echo "Téléchargement OR-Tools ${ORTOOLS_VERSION}..."
    wget -q --show-progress -O "$ORTOOLS_TAR" "$ORTOOLS_URL" || \
    curl -fL --progress-bar -o "$ORTOOLS_TAR" "$ORTOOLS_URL"
fi
echo "[OK] $ORTOOLS_TAR"

if [[ "$DOWNLOAD_ONLY" == "--download-only" ]]; then
    echo ""
    echo "Téléchargements terminés. Lance maintenant depuis ta session interactive :"
    echo "  bash compile_cc.sh"
    exit 0
fi

# ── 1. Modules ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Modules ==="
module load StdEnv/2023
module load gcc/12.3
module load cmake/3.27.7
module load python/3.11
module load scipy-stack
echo "[OK] modules chargés"

# ── 2. OR-Tools C++ (extract du package binaire Ubuntu) ───────────────────────
echo ""
echo "=== OR-Tools C++ ${ORTOOLS_VERSION} ==="
if [[ -f "$ORTOOLS_INSTALL/lib/cmake/ortools/ORToolsConfig.cmake" ]]; then
    echo "[OK] déjà extrait dans $ORTOOLS_INSTALL — skip"
else
    rm -rf "$ORTOOLS_INSTALL"
    mkdir -p "$ORTOOLS_INSTALL"
    tar xzf "$ORTOOLS_TAR" -C "$ORTOOLS_INSTALL" --strip-components=1
    echo "[OK] extrait dans $ORTOOLS_INSTALL"
fi
ORTOOLS_CMAKE="$ORTOOLS_INSTALL/lib/cmake/ortools"
echo "     cmake dir : $ORTOOLS_CMAKE"

# ── 3. Gecode (compilation depuis les sources) ────────────────────────────────
echo ""
echo "=== Gecode ${GECODE_VERSION} ==="
if [[ -f "$GECODE_INSTALL/lib/libgecodeint.so" ]]; then
    echo "[OK] déjà compilé dans $GECODE_INSTALL — skip"
else
    GECODE_SRC="$EXT/gecode-src"
    if [[ ! -d "$GECODE_SRC" ]]; then
        mkdir -p "$GECODE_SRC"
        tar xzf "$GECODE_TAR" -C "$GECODE_SRC" --strip-components=1
    fi
    echo "Configuration..."
    cd "$GECODE_SRC"
    ./configure \
        --prefix="$GECODE_INSTALL" \
        --disable-qt \
        --disable-gist \
        --disable-examples \
        --disable-flatzinc \
        2>&1 | tail -3
    echo "Compilation (~10-15 min)..."
    make -j4 2>&1 | tail -3
    make install 2>&1 | tail -3
    cd "$REPO"
    echo "[OK] installé dans $GECODE_INSTALL"
fi

# ── 4. libv1ld_generic_core.a ──────────────────────────────────────────────────
echo ""
echo "=== libv1ld_generic_core.a ==="
GENERIC_BUILD="$REPO/solver_generic/build_cc"
rm -rf "$GENERIC_BUILD" && mkdir -p "$GENERIC_BUILD"

cmake -S "$REPO/solver_generic" -B "$GENERIC_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$ORTOOLS_INSTALL" \
    2>&1 | tail -5

cmake --build "$GENERIC_BUILD" --target v1ld_generic_core -j4

[[ -f "$GENERIC_BUILD/libv1ld_generic_core.a" ]] || \
    { echo "ERREUR : libv1ld_generic_core.a non générée"; exit 1; }
echo "[OK] $GENERIC_BUILD/libv1ld_generic_core.a"

# ── 5. bab_mkp_gecode ──────────────────────────────────────────────────────────
echo ""
echo "=== bab_mkp_gecode ==="
BAB_BUILD="$REPO/solver_bab/build"
rm -rf "$BAB_BUILD" && mkdir -p "$BAB_BUILD"

cmake -S "$REPO/solver_bab" -B "$BAB_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$ORTOOLS_INSTALL" \
    -DGECODE_PREFIX="$GECODE_INSTALL" \
    -DSOLVER_GENERIC_BUILD="$GENERIC_BUILD" \
    2>&1 | tail -5

cmake --build "$BAB_BUILD" \
    --target bab_mkp_gecode \
    --target bab_mkp_gecode_collect \
    --target bab_mkp_gecode_base \
    --target bab_mkp_gecode_base_collect \
    -j4

[[ -f "$BAB_BUILD/bab_mkp_gecode" ]] || \
    { echo "ERREUR : bab_mkp_gecode non généré"; exit 1; }
[[ -f "$BAB_BUILD/bab_mkp_gecode_collect" ]] || \
    { echo "ERREUR : bab_mkp_gecode_collect non généré"; exit 1; }
[[ -f "$BAB_BUILD/bab_mkp_gecode_base" ]] || \
    { echo "ERREUR : bab_mkp_gecode_base non généré"; exit 1; }
[[ -f "$BAB_BUILD/bab_mkp_gecode_base_collect" ]] || \
    { echo "ERREUR : bab_mkp_gecode_base_collect non généré"; exit 1; }

echo ""
echo "======================================================================"
echo "  Compilation réussie !"
echo "  Binaires : $BAB_BUILD/bab_mkp_gecode"
echo "             $BAB_BUILD/bab_mkp_gecode_collect"
echo "             $BAB_BUILD/bab_mkp_gecode_base"
echo "             $BAB_BUILD/bab_mkp_gecode_base_collect"
echo "  Prochaine étape : sbatch sweep_bab.sh  ou  sbatch collect_large.sh"
echo "======================================================================"
