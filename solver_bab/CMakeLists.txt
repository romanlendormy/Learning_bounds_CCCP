# solver_bab : B&B Gecode avec bornes lagrangiennes (MKP, CCPO).
# Inclus depuis le CMakeLists racine (sld_generic_core, ortools et OpenMP y sont définis).
# Les cibles MKP et CCPO sont ajoutées dans les sections dédiées plus bas.

# ── Gecode ────────────────────────────────────────────────────────────────────
set(GECODE_PREFIX "${EXTERNAL_DIR}/gecode" CACHE PATH "Préfixe d'installation Gecode 6.2")
set(GECODE_INCLUDE_DIRS "${GECODE_PREFIX}/include")
set(GECODE_LINK_FLAGS
    -L${GECODE_PREFIX}/lib -Wl,-rpath,${GECODE_PREFIX}/lib
    -lgecodeint -lgecodekernel -lgecodesupport -lgecodesearch -lgecodeminimodel
)

# ── LibTorch (inférence GNN dans le B&B) ──────────────────────────────────────
# Distribution standalone de LibTorch (pas le paquet pip) liée directement, sans
# find_package(Torch) dont le CMake exige protobuf/CUDA même en CPU.
# -DGNN_DEVICE=cpu (défaut) : external/libtorch ; =cuda : external/libtorch-cuda.
# Avec la version CUDA, le device (cpu ou cuda) reste un paramètre d'exécution.
set(GNN_DEVICE "cpu" CACHE STRING "Distribution LibTorch liée : cpu ou cuda")
set_property(CACHE GNN_DEVICE PROPERTY STRINGS cpu cuda)

if(GNN_DEVICE STREQUAL "cuda")
    set(LIBTORCH_PREFIX "${EXTERNAL_DIR}/libtorch-cuda" CACHE PATH "Préfixe LibTorch")
    # LibTorch >= 2.13 ne fournit plus cuBLAS/cuDNN/NCCL/NVSHMEM : ils viennent des
    # modules du cluster (module load cuda/12.6 cudnn/9.10 avant de configurer).
    # --allow-shlib-undefined tolère les symboles jamais appelés (NVSHMEM multi-GPU).
    set(LIBTORCH_LINK_FLAGS
        -L${LIBTORCH_PREFIX}/lib -Wl,-rpath,${LIBTORCH_PREFIX}/lib
        -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -Wl,--as-needed
        -Wl,--allow-shlib-undefined
    )
    # Certains en-têtes LibTorch incluent cuda_runtime_api.h ; EBROOTCUDA est défini
    # par le module cuda (CUDA_HOME ne l'est pas).
    if(DEFINED ENV{EBROOTCUDA})
        set(LIBTORCH_INCLUDE_DIRS_EXTRA "$ENV{EBROOTCUDA}/targets/x86_64-linux/include")
        list(APPEND LIBTORCH_LINK_FLAGS
            -L$ENV{EBROOTCUDA}/targets/x86_64-linux/lib -Wl,-rpath,$ENV{EBROOTCUDA}/targets/x86_64-linux/lib
            -L$ENV{EBROOTCUDA}/extras/CUPTI/lib64 -Wl,-rpath,$ENV{EBROOTCUDA}/extras/CUPTI/lib64
        )
    endif()
else()
    set(LIBTORCH_PREFIX "${EXTERNAL_DIR}/libtorch" CACHE PATH "Préfixe LibTorch")
    set(LIBTORCH_LINK_FLAGS
        -L${LIBTORCH_PREFIX}/lib -Wl,-rpath,${LIBTORCH_PREFIX}/lib
        -ltorch -ltorch_cpu -lc10
    )
endif()
set(LIBTORCH_INCLUDE_DIRS
    "${LIBTORCH_PREFIX}/include"
    "${LIBTORCH_PREFIX}/include/torch/csrc/api/include"
    ${LIBTORCH_INCLUDE_DIRS_EXTRA}
)

# ── Stub NVSHMEM (requis par libtorch_cuda 2.13 sur un cluster sans NVSHMEM) ──
# Construit dans le dossier de build ; env/setup_gpu_env.sh l'ajoute au LD_LIBRARY_PATH.
if(GNN_DEVICE STREQUAL "cuda")
    add_library(nvshmem_host SHARED nvshmem_stub/nvshmem_stub.c)
    set_target_properties(nvshmem_host PROPERTIES
        OUTPUT_NAME nvshmem_host SOVERSION 3
        LINK_FLAGS "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/nvshmem_stub/nvshmem_stub.map")
endif()
