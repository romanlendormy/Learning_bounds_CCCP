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

# ══ MKP ═══════════════════════════════════════════════════════════════════════
# mkp_bab_core : moteur B&B Gecode (REF / SLD / DLD / GNN mono-nœud), moteurs
# batchés (gnn_batched SLD et DLD) et backends GNN natifs (TorchScript).
add_library(mkp_bab_core STATIC
    mkp/gecode_bab.cpp
    mkp/gecode_bab_gnn_batched.cpp
    mkp/gecode_bab_dld_gnn_batched.cpp
    mkp/gnn_sd_edit_native.cpp
    mkp/gnn_sd_edit_native_batched.cpp
    mkp/gnn_dld_sd_edit_native.cpp
    mkp/gnn_dld_sd_edit_native_batched.cpp
    mkp/bnb_solver.cpp                      # ζ_i (B&B exact / surrogate), appelé par les backends GNN
)
target_include_directories(mkp_bab_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/mkp
    ${GECODE_INCLUDE_DIRS}
    ${LIBTORCH_INCLUDE_DIRS}
)
target_compile_definitions(mkp_bab_core PUBLIC _GLIBCXX_USE_CXX11_ABI=1)
target_compile_options(mkp_bab_core PRIVATE -O3 -march=native)
target_link_libraries(mkp_bab_core PUBLIC
    sld_generic_core ortools::ortools OpenMP::OpenMP_CXX
    ${LIBTORCH_LINK_FLAGS} ${GECODE_LINK_FLAGS}
)

# bab_mkp_gecode : CLI principale (modes ref, sld, dld, gnn, gnn_root, gnn_root_dld,
# gnn_batched, gnn_batched_dld ; GNN_ZERO_LAMBDA=1 -> ablation λ = 0 en gnn_batched).
add_executable(bab_mkp_gecode mkp/main_gecode.cpp)
target_compile_options(bab_mkp_gecode PRIVATE -O3 -march=native)
target_link_libraries(bab_mkp_gecode PRIVATE mkp_bab_core)

# bab_mkp_gecode_dld_gnn_batched : CLI dédiée à la campagne DLD + GNN batché.
add_executable(bab_mkp_gecode_dld_gnn_batched mkp/main_bab_dld_gnn_batched.cpp)
target_compile_options(bab_mkp_gecode_dld_gnn_batched PRIVATE -O3 -march=native)
target_link_libraries(bab_mkp_gecode_dld_gnn_batched PRIVATE mkp_bab_core)

# bab_mkp_gecode_base_collect : B&B Gecode de base + collecte des nœuds (données GNN).
add_executable(bab_mkp_gecode_base_collect
    ../gecode_base/main_collect_gecode_base.cpp
    ../gecode_base/collect_gecode_base.cpp
)
target_compile_options(bab_mkp_gecode_base_collect PRIVATE -O3 -march=native)
target_link_libraries(bab_mkp_gecode_base_collect PRIVATE mkp_bab_core)

# ══ CCPO ══════════════════════════════════════════════════════════════════════
# ccpo_bab_core : B&B Gecode CCPO (REF / SLD / FEAS, cf. gecode_bab_ccpo.hpp) et
# driver batché (GNN ou λ = 0, cf. gecode_bab_ccpo_gnn_batched.hpp).
add_library(ccpo_bab_core STATIC
    ccpo/gecode_bab_ccpo.cpp
    ccpo/gecode_bab_ccpo_gnn_batched.cpp
)
target_include_directories(ccpo_bab_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ccpo ${GECODE_INCLUDE_DIRS})
target_compile_options(ccpo_bab_core PRIVATE -O3 -march=native)
target_link_libraries(ccpo_bab_core PUBLIC ccpo_bound OpenMP::OpenMP_CXX ${GECODE_LINK_FLAGS})

# ccpo_gnn_native : inférence du GNN CCPO (LibTorch, graphe construit en C++).
# Compilé sans les en-têtes Gecode (cf. ccpo_gnn_backend_batch.hpp).
add_library(ccpo_gnn_native STATIC ccpo/gnn_ccpo_native_batched.cpp)
target_include_directories(ccpo_gnn_native PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ccpo PRIVATE ${LIBTORCH_INCLUDE_DIRS})
target_compile_options(ccpo_gnn_native PRIVATE -O3 -march=native)
target_link_libraries(ccpo_gnn_native PUBLIC OpenMP::OpenMP_CXX ${LIBTORCH_LINK_FLAGS})

# bab_ccpo : CLI de la campagne CCPO (modes ref, sld, feas, gnn_zero, gnn).
add_executable(bab_ccpo ccpo/main_gecode_ccpo.cpp)
target_compile_definitions(bab_ccpo PRIVATE CCPO_WITH_TORCH CCPO_WITH_SCIP)
target_compile_options(bab_ccpo PRIVATE -O3 -march=native)
target_link_libraries(bab_ccpo PRIVATE ccpo_bab_core ccpo_gnn_native ccpo_scip_zeta)

# bab_ccpo_rootcmp : borne à la racine, un forward GNN vs sous-gradient (t*).
add_executable(bab_ccpo_rootcmp ccpo/main_root_compare_ccpo.cpp)
target_compile_definitions(bab_ccpo_rootcmp PRIVATE CCPO_WITH_TORCH CCPO_WITH_SCIP)
target_compile_options(bab_ccpo_rootcmp PRIVATE -O3 -march=native)
target_link_libraries(bab_ccpo_rootcmp PRIVATE ccpo_bab_core ccpo_gnn_native ccpo_scip_zeta)

# bab_ccpo_collect : B&B REF + collecte des nœuds (données d'entraînement du GNN CCPO).
add_executable(bab_ccpo_collect ccpo/main_collect_gecode_ccpo.cpp)
target_compile_options(bab_ccpo_collect PRIVATE -O3 -march=native)
target_link_libraries(bab_ccpo_collect PRIVATE ccpo_bab_core)

# ccpo_graph.so : graphe du GNN CCPO construit en C++, appelé depuis Python (ctypes) pour
# vérifier la correspondance avec gnn_ccpo/ccpo_graph.py (gnn_ccpo/test_correspondence.py).
add_library(ccpo_graph SHARED ccpo/ccpo_graph_bind.cpp)
target_include_directories(ccpo_graph PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ccpo)
target_compile_options(ccpo_graph PRIVATE -O3 -march=native)
set_target_properties(ccpo_graph PROPERTIES PREFIX "" SUFFIX ".so" CXX_EXTENSIONS OFF)

# Même dialecte que la compilation de la campagne CCPO (-std=c++17, sans extensions GNU).
set_target_properties(bab_ccpo bab_ccpo_rootcmp bab_ccpo_collect ccpo_bab_core ccpo_gnn_native
    PROPERTIES CXX_EXTENSIONS OFF)
