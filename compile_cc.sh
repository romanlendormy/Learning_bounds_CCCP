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

# LibTorch CPU standalone (PAS le package pip "torch" : son CMake exige
# protobuf/CUDA même pour un usage CPU pur — cf. solver_bab/mkp/gnn_v1ld_native.cpp).
# Utilisé par le backend gnn_backend=native (mode=gnn) en device=cpu.
#
# 2.13.0, PAS 2.5.1 : requis pour que les artefacts AOTInductor exportés
# depuis le venv Python (torch 2.13.0, wheelhouse Alliance Canada) soient
# chargeables par le runtime C++ (couplage ABI cxx11 ET version quasi-exacte
# du C-shim aoti_torch_* — cf. mémoire "aoti-abi-version-mismatch" et
# "libtorch213-cuda-validated"). Le nom du zip n'a plus le suffixe
# "cxx11-abi" à cette version : c'est devenu l'unique ABI distribuée.
LIBTORCH_VERSION="2.13.0"
LIBTORCH_INSTALL="$EXT/libtorch"
LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip"

# LibTorch CUDA (opt-in, ~800 Mo — WITH_CUDA=1 bash compile_cc.sh) : requis
# uniquement pour un binaire construit avec -DGNN_DEVICE=cuda (device=cuda à
# l'exécution, cf. gnn_v1ld_native.cpp). cu126 : correspond au module
# cuda/12.6 du cluster.
#
# IMPORTANT — contrairement à 2.5.1 (self-contained, vendored), ce zip ne
# bundle plus cublas/cudnn/cusparse/cupti/nccl/nvshmem : il faut les modules
# système ci-dessous ET solver_bab/setup_gpu_env.sh au RUNTIME (pas
# seulement à la compilation) sous peine d'un "error while loading shared
# libraries" cryptique au premier lancement GPU. Cf. mémoire
# "libtorch213-cuda-validated" pour le détail de chaque dépendance.
WITH_CUDA="${WITH_CUDA:-0}"
LIBTORCH_CUDA_INSTALL="$EXT/libtorch-cuda"
LIBTORCH_CUDA_URL="https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcu126.zip"

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

LIBTORCH_ZIP="$EXT/libtorch-${LIBTORCH_VERSION}-cpu.zip"
if [[ ! -f "$LIBTORCH_ZIP" ]]; then
    echo "Téléchargement LibTorch CPU ${LIBTORCH_VERSION} (~170 Mo)..."
    wget -q --show-progress -O "$LIBTORCH_ZIP" "$LIBTORCH_URL" || \
    curl -fL --progress-bar -o "$LIBTORCH_ZIP" "$LIBTORCH_URL"
fi
echo "[OK] $LIBTORCH_ZIP"

if [[ "$WITH_CUDA" == "1" ]]; then
    LIBTORCH_CUDA_ZIP="$EXT/libtorch-${LIBTORCH_VERSION}-cu126.zip"
    if [[ ! -f "$LIBTORCH_CUDA_ZIP" ]]; then
        echo "Téléchargement LibTorch CUDA ${LIBTORCH_VERSION} (~2.3 Go)..."
        wget -q --show-progress -O "$LIBTORCH_CUDA_ZIP" "$LIBTORCH_CUDA_URL" || \
        curl -fL --progress-bar -o "$LIBTORCH_CUDA_ZIP" "$LIBTORCH_CUDA_URL"
    fi
    echo "[OK] $LIBTORCH_CUDA_ZIP"
fi

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
if [[ "$WITH_CUDA" == "1" ]]; then
    # cudnn/cusparselt : requis par libtorch-cuda 2.13.0 (non vendored, cf.
    # note ci-dessus) — version cusparselt COUPLÉE à cudacore (vérifier avec
    # `module spider cusparselt/<v>` si cuda/12.6 change un jour de version.
    module load cuda/12.6 cudnn/9.10.0.56 cusparselt/0.7.1.0
fi
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

# ── 2bis. LibTorch CPU (extraction directe, pas de compilation) ──────────────
echo ""
echo "=== LibTorch CPU ${LIBTORCH_VERSION} ==="
if [[ -f "$LIBTORCH_INSTALL/lib/libtorch_cpu.so" ]]; then
    echo "[OK] déjà extrait dans $LIBTORCH_INSTALL — skip"
else
    rm -rf "$LIBTORCH_INSTALL"
    unzip -q "$LIBTORCH_ZIP" -d "$EXT"   # produit $EXT/libtorch/...
    echo "[OK] extrait dans $LIBTORCH_INSTALL"
fi

if [[ "$WITH_CUDA" == "1" ]]; then
    echo ""
    echo "=== LibTorch CUDA ${LIBTORCH_VERSION} (cu126) ==="
    if [[ -f "$LIBTORCH_CUDA_INSTALL/lib/libtorch_cuda.so" ]]; then
        echo "[OK] déjà extrait dans $LIBTORCH_CUDA_INSTALL — skip"
    else
        rm -rf "$LIBTORCH_CUDA_INSTALL" "$EXT/.libtorch_cuda_tmp"
        mkdir -p "$EXT/.libtorch_cuda_tmp"
        unzip -q "$LIBTORCH_CUDA_ZIP" -d "$EXT/.libtorch_cuda_tmp"   # produit .../libtorch/...
        mv "$EXT/.libtorch_cuda_tmp/libtorch" "$LIBTORCH_CUDA_INSTALL"
        rm -rf "$EXT/.libtorch_cuda_tmp"
        echo "[OK] extrait dans $LIBTORCH_CUDA_INSTALL"
    fi

    # Stub libnvshmem_host.so.3 : NVSHMEM (communication multi-GPU) n'est
    # fourni par AUCUN module de ce cluster, mais libtorch_cuda.so a un
    # NEEDED dur dessus (via libtorch_nvshmem.so, bundlé dans le zip) —
    # sans lui le chargeur dynamique refuse de démarrer N'IMPORTE QUEL
    # binaire liant libtorch-cuda, même en usage mono-GPU qui n'appelle
    # jamais NVSHMEM. Stub sûr (19 fonctions no-op, jamais invoquées par
    # notre code) — cf. solver_bab/nvshmem_stub/, mémoire
    # "libtorch213-cuda-validated".
    NVSHMEM_STUB_INSTALL="$EXT/nvshmem_stub/lib"
    if [[ -f "$NVSHMEM_STUB_INSTALL/libnvshmem_host.so.3" ]]; then
        echo "[OK] stub nvshmem déjà compilé dans $NVSHMEM_STUB_INSTALL — skip"
    else
        mkdir -p "$NVSHMEM_STUB_INSTALL"
        gcc -shared -fPIC -o "$NVSHMEM_STUB_INSTALL/libnvshmem_host.so.3" \
            -Wl,-soname,libnvshmem_host.so.3 \
            -Wl,--version-script="$REPO/solver_bab/nvshmem_stub/nvshmem_stub.map" \
            "$REPO/solver_bab/nvshmem_stub/nvshmem_stub.c"
        echo "[OK] stub compilé dans $NVSHMEM_STUB_INSTALL"
    fi
fi

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
# Pas de rm -rf : réutilise CMakeCache.txt s'il existe déjà (reconfigure
# incrémentale) — find_package(ortools) parcourt des milliers de petits
# fichiers (deps vendorisées : abseil/protobuf/SCIP/...) et coûte ~10 min
# à chaque appel sur Lustre sous contention (cf. job 16006627, configure
# solver_generic seul = 555.7s). Wipe manuel si un rebuild propre est requis.
mkdir -p "$GENERIC_BUILD"

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
# Pas de rm -rf : même raison que GENERIC_BUILD ci-dessus (idem find_package
# ortools + OpenMP). GNN_DEVICE change bien de valeur d'un run à l'autre
# (cpu/cuda) mais -DGNN_DEVICE=... sur la ligne cmake ci-dessous met à jour
# la variable de cache existante — pas besoin de repartir de zéro.
mkdir -p "$BAB_BUILD"

GNN_DEVICE_FLAG="cpu"
[[ "$WITH_CUDA" == "1" ]] && GNN_DEVICE_FLAG="cuda"

cmake -S "$REPO/solver_bab" -B "$BAB_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$ORTOOLS_INSTALL" \
    -DGECODE_PREFIX="$GECODE_INSTALL" \
    -DSOLVER_GENERIC_BUILD="$GENERIC_BUILD" \
    -DGNN_DEVICE="$GNN_DEVICE_FLAG" \
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
if [[ "$WITH_CUDA" == "1" ]]; then
echo "  IMPORTANT (device=cuda) : source solver_bab/setup_gpu_env.sh AVANT"
echo "  de lancer un binaire — sinon 'error while loading shared libraries'"
echo "  (cudnn/cusparselt/nvshmem non résolus par le seul RPATH baké au link)."
fi
echo "  Prochaine étape : sbatch sweep_bab.sh  ou  sbatch collect_large.sh"
echo "======================================================================"
