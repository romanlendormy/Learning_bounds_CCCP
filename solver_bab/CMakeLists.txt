cmake_minimum_required(VERSION 3.14)
project(solver_bab_mkp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── OR-Tools (nécessaire pour MKPProblem / CP-SAT dans v1ld_generic_core) ─────
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew" "/usr/local")  # macOS ; ignoré ailleurs
find_package(ortools CONFIG REQUIRED)

# ── Bibliothèque générique pré-compilée (solver_generic) ─────────────────────
set(SOLVER_GENERIC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../solver_generic")
# Surchargeable via -DSOLVER_GENERIC_BUILD=... (ex: build_cc sur Compute Canada)
if(NOT DEFINED SOLVER_GENERIC_BUILD)
    set(SOLVER_GENERIC_BUILD "${SOLVER_GENERIC_ROOT}/build")
endif()

add_library(v1ld_generic_core STATIC IMPORTED)
set_target_properties(v1ld_generic_core PROPERTIES
    IMPORTED_LOCATION "${SOLVER_GENERIC_BUILD}/libv1ld_generic_core.a"
    INTERFACE_INCLUDE_DIRECTORIES
        "${SOLVER_GENERIC_ROOT};${SOLVER_GENERIC_ROOT}/mkp;${SOLVER_GENERIC_ROOT}/kp;${SOLVER_GENERIC_ROOT}/ssp;${SOLVER_GENERIC_ROOT}/covset"
)

# ── Gecode ────────────────────────────────────────────────────────────────────
# Surchargeable via -DGECODE_PREFIX=... (Compute Canada, Linux, etc.)
if(NOT DEFINED GECODE_PREFIX)
    set(GECODE_PREFIX "/opt/homebrew/opt/gecode")
endif()

# ── LibTorch (backend natif de la borne GNN, gnn_v1ld_native.cpp) ────────────
# Distribution standalone (download.pytorch.org), PAS le package pip "torch" :
# son CMake (TorchConfig.cmake -> Caffe2Config.cmake) exige protobuf/CUDA même
# pour un usage CPU pur — on lie directement les .so, sans find_package(Torch).
#
# -DGNN_DEVICE=cpu (défaut) ou =cuda : choisit QUELLE distribution LibTorch
# est liée (CPU-only vs CUDA 12.6, external/libtorch vs external/libtorch-cuda
# — toutes deux en 2.13.0 depuis 2026-07-14, cf. mémoire "libtorch213-cuda-validated" ;
# 2.5.1 archivé dans external/libtorch-2.5.1{,-cuda} au cas où
# — deux arbres séparés pour ne pas casser le build CPU existant). Le device
# RÉELLEMENT utilisé à l'exécution reste un paramètre runtime du CLI
# (bab_mkp_gecode ... gnn ... cuda native — cf. main_gecode.cpp) : lier
# libtorch-cuda permet d'utiliser device=cuda OU cpu au choix ; lier le
# libtorch CPU-only ne permet que device=cpu (cuda lève une exception claire).
# Surchargeable via -DLIBTORCH_PREFIX=... directement.
if(NOT DEFINED GNN_DEVICE)
    set(GNN_DEVICE "cpu")
endif()

if(GNN_DEVICE STREQUAL "cuda")
    if(NOT DEFINED LIBTORCH_PREFIX)
        set(LIBTORCH_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/../external/libtorch-cuda")
    endif()
    # --allow-shlib-undefined : nécessaire pour libtorch >= 2.13 (external/libtorch-cuda
    # 2.5.1 est self-contained, vendored cublas/cudnn/etc. dans son propre lib/ ;
    # les zips plus récents ("shared-with-deps") ne vendorisent plus ces libs CUDA et
    # s'attendent à ce qu'elles soient fournies par les modules système (cuda/cudnn/nccl)
    # — sans quoi le lien statique échoue sur des symboles jamais réellement appelés
    # par notre usage (nvshmem multi-GPU, ops cudnn non utilisées, etc.). Inoffensif
    # pour le libtorch 2.5.1 vendored (rien à laisser non résolu).
    set(LIBTORCH_LINK_FLAGS
        -L${LIBTORCH_PREFIX}/lib -Wl,-rpath,${LIBTORCH_PREFIX}/lib
        -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda
        -Wl,--allow-shlib-undefined
    )
    # Certains headers LibTorch (torch/csrc/inductor/aoti_runner/*_cuda.h ->
    # c10/cuda/CUDAStream.h) incluent directement cuda_runtime_api.h — pas
    # exposé par CPATH/LIBRARY_PATH du module `cuda` (cvmfs expose seulement
    # .../cudacore/<v>/lib, pas .../targets/x86_64-linux/{include,lib} ni
    # .../extras/CUPTI/lib64). Requiert `module load cuda/12.6` (et
    # `cudnn/9.x` pour libcudnn.so.9) AVANT de configurer/compiler.
    # EBROOTCUDA (convention EasyBuild du cluster) plutôt que CUDA_HOME, qui
    # n'est PAS défini par `module load cuda/12.6` sur ce cluster.
    if(DEFINED ENV{EBROOTCUDA})
        set(LIBTORCH_INCLUDE_DIRS_EXTRA "$ENV{EBROOTCUDA}/targets/x86_64-linux/include")
        list(APPEND LIBTORCH_LINK_FLAGS
            -L$ENV{EBROOTCUDA}/targets/x86_64-linux/lib -Wl,-rpath,$ENV{EBROOTCUDA}/targets/x86_64-linux/lib
            -L$ENV{EBROOTCUDA}/extras/CUPTI/lib64 -Wl,-rpath,$ENV{EBROOTCUDA}/extras/CUPTI/lib64
        )
    endif()
else()
    if(NOT DEFINED LIBTORCH_PREFIX)
        set(LIBTORCH_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/../external/libtorch")
    endif()
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
set(GNN_NATIVE_SOURCES
    mkp/gnn_v1ld_native.cpp
    mkp/gnn_sd_edit_native.cpp
    mkp/gnn_sd_edit_native_aoti.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../gnn/bnb_solver.cpp
)

# solve_zeta_batch (gnn/bnb_solver.cpp) résout N_z sous-problèmes ζ
# indépendants (un par scénario) via #pragma omp parallel for — ce pragma
# est un no-op sans -fopenmp au COMPILE (le lien transitif de libtorch_cpu
# vers libgomp ne suffit pas : c'est le compilateur qui doit traduire le
# pragma en appels GOMP_*). Initialement câblé UNIQUEMENT sur gnn_native_probe
# (sonde isolée) — constaté après coup (session 2026-07-13) que les vrais
# exécutables B&B (bab_mkp_gecode, bab_mkp_gecode_collect,
# bab_mkp_gecode_base_collect) en étaient dépourvus, donc le pragma y restait
# un no-op malgré la présence du #pragma omp dans les sources compilées.
# Corrigé en linkant OpenMP::OpenMP_CXX sur ces 3 cibles aussi (ci-dessous).
find_package(OpenMP REQUIRED)

# ── Exécutable bab_mkp_gecode (Gecode BAB + borne V1LD) ──────────────────────
# mode=gnn_batched (cf. main_gecode.cpp) intègre solve_gecode_bab_mkp_gnn_batched
# (gecode_bab_gnn_batched.cpp/hpp, Étape 2e du plan de batching, validé sur le
# sweep de robustesse job 16406490) SANS toucher gecode_bab.cpp/.hpp — fichiers
# séparés ajoutés à la même cible, même convention que bab_mkp_gecode_gnn_batched
# ci-dessous (qui reste disponible séparément pour la comparaison isolée).
add_executable(bab_mkp_gecode
    mkp/main_gecode.cpp
    mkp/gecode_bab.cpp
    mkp/gnn_v1ld_client.cpp
    mkp/gecode_bab_gnn_batched.cpp
    mkp/gnn_sd_edit_native_batched.cpp
    mkp/gnn_sd_edit_native_aoti_batched.cpp
    mkp/gnn_sd_edit_native_aoti_batched_static.cpp
    mkp/gecode_bab_v1cld_gnn_batched.cpp
    mkp/gnn_v1cld_sd_edit_native_batched.cpp
    mkp/gnn_v1cld_sd_edit_native.cpp
    ${GNN_NATIVE_SOURCES}
)

target_include_directories(bab_mkp_gecode PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/mkp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/mkp
    ${SOLVER_GENERIC_ROOT}/ssp
    ${SOLVER_GENERIC_ROOT}/covset
    ${GECODE_PREFIX}/include
    ${LIBTORCH_INCLUDE_DIRS}
)
target_compile_definitions(bab_mkp_gecode PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

target_link_libraries(bab_mkp_gecode PRIVATE
    v1ld_generic_core
    ortools::ortools
    ${LIBTORCH_LINK_FLAGS}
    OpenMP::OpenMP_CXX
    -L${GECODE_PREFIX}/lib
    -lgecodeint -lgecodekernel -lgecodesupport -lgecodesearch -lgecodeminimodel
)

target_compile_options(bab_mkp_gecode PRIVATE -O3 -march=native)

# ── Exécutable bab_mkp_gecode_bucketed (expérimental : bucketing des nœuds
#    fils par n_g avant pruning/incumbent, borne inchangée V1LD/V1CLD/REF —
#    cf. gecode_bab_bucketed.hpp) ─────────────────────────────────────────────
# ── Exécutable bab_mkp_gecode_v1cld_gnn_batched (équivalent V1CLD de
#    bab_mkp_gecode_gnn_batched — cf. gecode_bab_v1cld_gnn_batched.hpp/.cpp,
#    port quasi-verbatim du driver DFS+bucketing, seul le backend GNN change) ─
# NE compile PAS gecode_bab.cpp/gecode_bab_gnn_batched.cpp (greedy_mkp dupliqué
# localement dans gecode_bab_v1cld_gnn_batched.cpp, même convention que
# bab_mkp_gecode_gnn_batched) — a quand même besoin de LibTorch/OpenMP
# (GNNV1CLDSdEditNativeBatched) + Gecode.
add_executable(bab_mkp_gecode_v1cld_gnn_batched
    mkp/main_bab_v1cld_gnn_batched.cpp
    mkp/gecode_bab_v1cld_gnn_batched.cpp
    mkp/gnn_v1cld_sd_edit_native_batched.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../gnn/bnb_solver.cpp
)

target_include_directories(bab_mkp_gecode_v1cld_gnn_batched PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/mkp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/mkp
    ${SOLVER_GENERIC_ROOT}/ssp
    ${SOLVER_GENERIC_ROOT}/covset
    ${GECODE_PREFIX}/include
    ${LIBTORCH_INCLUDE_DIRS}
)
target_compile_definitions(bab_mkp_gecode_v1cld_gnn_batched PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

target_link_libraries(bab_mkp_gecode_v1cld_gnn_batched PRIVATE
    v1ld_generic_core
    ortools::ortools
    ${LIBTORCH_LINK_FLAGS}
    OpenMP::OpenMP_CXX
    -L${GECODE_PREFIX}/lib
    -lgecodeint -lgecodekernel -lgecodesupport -lgecodesearch -lgecodeminimodel
)

target_compile_options(bab_mkp_gecode_v1cld_gnn_batched PRIVATE -O3 -march=native)

# ── Exécutable bab_mkp_gecode_base_collect (CP pur + collecte GNN) ────────────
# Idem bab_mkp_gecode_collect : compile mkp/gecode_bab.cpp, même besoin de libs.
add_executable(bab_mkp_gecode_base_collect
    ../gecode_base/main_collect_gecode_base.cpp
    ../gecode_base/collect_gecode_base.cpp
    mkp/gecode_bab.cpp
    mkp/gnn_v1ld_client.cpp
    ${GNN_NATIVE_SOURCES}
)

target_include_directories(bab_mkp_gecode_base_collect PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/mkp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/mkp
    ${SOLVER_GENERIC_ROOT}/ssp
    ${SOLVER_GENERIC_ROOT}/covset
    ${GECODE_PREFIX}/include
    ${LIBTORCH_INCLUDE_DIRS}
)
target_compile_definitions(bab_mkp_gecode_base_collect PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

target_link_libraries(bab_mkp_gecode_base_collect PRIVATE
    v1ld_generic_core
    ortools::ortools
    ${LIBTORCH_LINK_FLAGS}
    OpenMP::OpenMP_CXX
    -L${GECODE_PREFIX}/lib
    -lgecodeint -lgecodekernel -lgecodesupport -lgecodesearch -lgecodeminimodel
)

target_compile_options(bab_mkp_gecode_base_collect PRIVATE -O3 -march=native)

# ── Exécutable bab_ssp_gecode_base_collect (CP pur + collecte GNN, SSP) ──────
# Analogue SSP de bab_mkp_gecode_base_collect : réutilise SSPSpaceBnB/
# SSPSharedData (ssp/gecode_bab_ssp.cpp) via une DFS manuelle avec
# use_lagrangian_bound=false (cf. ssp/collect_ssp_base.{hpp,cpp}).
add_executable(bab_ssp_gecode_base_collect
    ssp/main_collect_ssp_base.cpp
    ssp/collect_ssp_base.cpp
    ssp/gecode_bab_ssp.cpp
    ../gecode_base/gecode_solver.cpp
)

target_include_directories(bab_ssp_gecode_base_collect PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ssp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/ssp
    ${GECODE_PREFIX}/include
)

target_link_libraries(bab_ssp_gecode_base_collect PRIVATE
    v1ld_generic_core
    ortools::ortools
    OpenMP::OpenMP_CXX
    -L${GECODE_PREFIX}/lib
    -lgecodeint -lgecodekernel -lgecodesupport -lgecodesearch -lgecodeminimodel
)

target_compile_options(bab_ssp_gecode_base_collect PRIVATE -O3 -march=native)

# ── Exécutable test_gnn_v1cld_bab (étape 3 : intégration bout-en-bout du GNN
#    DANS le vrai B&B Gecode, SSPBoundMode::GNN, cf. ssp/gecode_bab_ssp.hpp) ─
add_executable(test_gnn_v1cld_bab
    ssp/test_gnn_v1cld_bab.cpp
    ssp/gnn_v1cld_native.cpp
    ssp/gecode_bab_ssp.cpp
)

target_include_directories(test_gnn_v1cld_bab PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ssp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/ssp
    ${GECODE_PREFIX}/include
    ${LIBTORCH_INCLUDE_DIRS}
)
target_compile_definitions(test_gnn_v1cld_bab PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

target_link_libraries(test_gnn_v1cld_bab PRIVATE
    v1ld_generic_core
    ortools::ortools
    OpenMP::OpenMP_CXX
    ${LIBTORCH_LINK_FLAGS}
    -L${GECODE_PREFIX}/lib
    -lgecodeint -lgecodekernel -lgecodesupport -lgecodesearch -lgecodeminimodel
)

# ── Exécutable test_gnn_v1cld_bab_batched (étape B3 : intégration bout-en-bout
#    du batching multi-nœuds SSP dans un vrai B&B, backend GNN réel -- cf.
#    ssp/test_gnn_v1cld_bab_batched.cpp) ──
add_executable(test_gnn_v1cld_bab_batched
    ssp/test_gnn_v1cld_bab_batched.cpp
    ssp/gecode_bab_ssp_gnn_batched.cpp
    ssp/gnn_v1cld_native.cpp
    ssp/gnn_v1cld_native_batched.cpp
)

target_include_directories(test_gnn_v1cld_bab_batched PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ssp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/ssp
    ${GECODE_PREFIX}/include
    ${LIBTORCH_INCLUDE_DIRS}
)
target_compile_definitions(test_gnn_v1cld_bab_batched PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

target_link_libraries(test_gnn_v1cld_bab_batched PRIVATE
    v1ld_generic_core
    ortools::ortools
    OpenMP::OpenMP_CXX
    ${LIBTORCH_LINK_FLAGS}
    -L${GECODE_PREFIX}/lib
    -lgecodeint -lgecodekernel -lgecodesupport -lgecodesearch -lgecodeminimodel
)

target_compile_options(test_gnn_v1cld_bab_batched PRIVATE -O3 -march=native)

target_compile_options(test_gnn_v1cld_bab PRIVATE -O3 -march=native)

# ── Exécutable campaign_v1ld_v1cld_vs_ref (campagne V1LD vs V1CLD sur
#    test_paper_trillium/ssp/, avec B&B REF exact comme référence
#    d'optimalité -- cf. solver_bab/ssp/campaign_v1ld_v1cld_vs_ref.cpp) ──────
add_executable(campaign_v1ld_v1cld_vs_ref
    ssp/campaign_v1ld_v1cld_vs_ref.cpp
    ssp/gecode_bab_ssp.cpp
    ../gecode_base/gecode_solver.cpp
)

target_include_directories(campaign_v1ld_v1cld_vs_ref PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ssp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/ssp
    ${GECODE_PREFIX}/include
)

target_link_libraries(campaign_v1ld_v1cld_vs_ref PRIVATE
    v1ld_generic_core
    ortools::ortools
    OpenMP::OpenMP_CXX
    -L${GECODE_PREFIX}/lib
    -lgecodeint -lgecodekernel -lgecodesupport -lgecodesearch -lgecodeminimodel
)

target_compile_options(campaign_v1ld_v1cld_vs_ref PRIVATE -O3 -march=native)

# ── Exécutable debug_v1cld_parallel_check (test de non-régression -- compare
#    solve_v1cld séquentiel vs solve_v1cld_parallel, campagne V1LD/V1CLD
#    trillium, cf. solver_generic/v1cld_solver_parallel.hpp) ──────────────────
add_executable(debug_v1cld_parallel_check
    ssp/debug_v1cld_parallel_check.cpp
    ssp/gecode_bab_ssp.cpp
    ../gecode_base/gecode_solver.cpp
)

target_include_directories(debug_v1cld_parallel_check PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ssp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/ssp
    ${GECODE_PREFIX}/include
)

target_link_libraries(debug_v1cld_parallel_check PRIVATE
    v1ld_generic_core
    ortools::ortools
    OpenMP::OpenMP_CXX
    -L${GECODE_PREFIX}/lib
    -lgecodeint -lgecodekernel -lgecodesupport -lgecodesearch -lgecodeminimodel
)

target_compile_options(debug_v1cld_parallel_check PRIVATE -O3 -march=native)

# ── Exécutable debug_replay_violation (diagnostic temporaire — rejoue un
#    (masque, lambda) capture au moment d'une violation reelle) ──────────────
add_executable(debug_replay_violation
    ssp/debug_replay_violation.cpp
    ${SOLVER_GENERIC_ROOT}/ssp/ssp_loader.cpp
)

target_include_directories(debug_replay_violation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ssp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/ssp
)

target_link_libraries(debug_replay_violation PRIVATE
    v1ld_generic_core
    ortools::ortools
)

target_compile_options(debug_replay_violation PRIVATE -O3 -march=native)

# ── Exécutable ssp_probe (sonde rapide : Gecode base seul, pour balayages) ────
add_executable(ssp_probe
    ssp/ssp_probe.cpp
    ../gecode_base/gecode_solver.cpp
)

target_include_directories(ssp_probe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ssp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/ssp
    ${GECODE_PREFIX}/include
)

target_link_libraries(ssp_probe PRIVATE
    v1ld_generic_core
    ortools::ortools
    -L${GECODE_PREFIX}/lib
    -lgecodeint -lgecodekernel -lgecodesupport -lgecodesearch -lgecodeminimodel
)

target_compile_options(ssp_probe PRIVATE -O3 -march=native)

