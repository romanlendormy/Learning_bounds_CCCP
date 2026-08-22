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
# Pas de dépendance LibTorch/OpenMP : bound_mode restreint à {V1LD,V1CLD,REF},
# le mode GNN (batching du forward) est hors périmètre de cette expérience.
# NE compile PAS gecode_bab.cpp (qui référence GNNV1LDNative/GNNSdEditNative/
# GNNV1LDClient, donc tirerait LibTorch même sans jamais les appeler) —
# gecode_bab_bucketed.cpp fournit sa propre copie locale de greedy_mkp
# (même convention que load_song_local, dupliqué dans chaque main_*.cpp de ce
# dossier pour éviter une dépendance croisée).
add_executable(bab_mkp_gecode_bucketed
    mkp/main_bab_bucketed.cpp
    mkp/gecode_bab_bucketed.cpp
)

target_include_directories(bab_mkp_gecode_bucketed PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/mkp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/mkp
    ${SOLVER_GENERIC_ROOT}/ssp
    ${SOLVER_GENERIC_ROOT}/covset
    ${GECODE_PREFIX}/include
)
target_compile_definitions(bab_mkp_gecode_bucketed PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

target_link_libraries(bab_mkp_gecode_bucketed PRIVATE
    v1ld_generic_core
    ortools::ortools
    OpenMP::OpenMP_CXX
    -L${GECODE_PREFIX}/lib
    -lgecodeint -lgecodekernel -lgecodesupport -lgecodesearch -lgecodeminimodel
)

target_compile_options(bab_mkp_gecode_bucketed PRIVATE -O3 -march=native)

# ── Exécutable bab_mkp_gecode_gnn_batched (Étape 2e du plan de batching :
#    intégration Gecode complète du backend GNN batché — cf.
#    gecode_bab_gnn_batched.hpp) ───────────────────────────────────────────────
# NE compile PAS gecode_bab.cpp (greedy_mkp dupliqué localement dans
# gecode_bab_gnn_batched.cpp, même convention que bab_mkp_gecode_bucketed) —
# a quand même besoin de LibTorch/OpenMP (GNNSdEditNativeBatched) + Gecode.
add_executable(bab_mkp_gecode_gnn_batched
    mkp/main_bab_gnn_batched.cpp
    mkp/gecode_bab_gnn_batched.cpp
    mkp/gnn_sd_edit_native_batched.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../gnn/bnb_solver.cpp
)

target_include_directories(bab_mkp_gecode_gnn_batched PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/mkp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/mkp
    ${SOLVER_GENERIC_ROOT}/ssp
    ${SOLVER_GENERIC_ROOT}/covset
    ${GECODE_PREFIX}/include
    ${LIBTORCH_INCLUDE_DIRS}
)
target_compile_definitions(bab_mkp_gecode_gnn_batched PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

target_link_libraries(bab_mkp_gecode_gnn_batched PRIVATE
    v1ld_generic_core
    ortools::ortools
    ${LIBTORCH_LINK_FLAGS}
    OpenMP::OpenMP_CXX
    -L${GECODE_PREFIX}/lib
    -lgecodeint -lgecodekernel -lgecodesupport -lgecodesearch -lgecodeminimodel
)

target_compile_options(bab_mkp_gecode_gnn_batched PRIVATE -O3 -march=native)

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

# ── Exécutable bab_mkp_gecode_gnn_batched_zaware (variante Z-AWARE de
#    bab_mkp_gecode_gnn_batched — exploite les z déjà résolus par la
#    propagation CP pendant le x-branching, cf.
#    gecode_bab_gnn_batched_zaware.hpp/.cpp et gnn_sd_edit_native_batched_zaware.hpp/.cpp).
# Fichiers séparés : gecode_bab_gnn_batched.cpp/gnn_sd_edit_native_batched.cpp
# (originaux, utilisés par bab_mkp_gecode/bab_mkp_gecode_gnn_batched) restent
# intacts et hors de cette cible — retour en arrière trivial.
add_executable(bab_mkp_gecode_gnn_batched_zaware
    mkp/main_bab_gnn_batched_zaware.cpp
    mkp/gecode_bab_gnn_batched_zaware.cpp
    mkp/gnn_sd_edit_native_batched_zaware.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../gnn/bnb_solver.cpp
)

target_include_directories(bab_mkp_gecode_gnn_batched_zaware PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/mkp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/mkp
    ${SOLVER_GENERIC_ROOT}/ssp
    ${SOLVER_GENERIC_ROOT}/covset
    ${GECODE_PREFIX}/include
    ${LIBTORCH_INCLUDE_DIRS}
)
target_compile_definitions(bab_mkp_gecode_gnn_batched_zaware PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

target_link_libraries(bab_mkp_gecode_gnn_batched_zaware PRIVATE
    v1ld_generic_core
    ortools::ortools
    ${LIBTORCH_LINK_FLAGS}
    OpenMP::OpenMP_CXX
    -L${GECODE_PREFIX}/lib
    -lgecodeint -lgecodekernel -lgecodesupport -lgecodesearch -lgecodeminimodel
)

target_compile_options(bab_mkp_gecode_gnn_batched_zaware PRIVATE -O3 -march=native)

# ── Exécutable bab_kp_gecode (Gecode BAB + borne V1LD, cas CC-KP m=1) ────────
# Pas de dépendance LibTorch/GNN : seuls REF et V1LD sont supportés (V1CLD est
# redondant à m=1, GNN est un fil de recherche séparé) — cf. kp_gecode_bab.hpp.
# ζ résolu via KPProblem (B&B 1D dédié, solver_generic/kp/) au lieu du chemin
# multi-dim générique MKPProblemBnB — ~1300x plus rapide à m=1 (cf. micro-
# benchmark de session comparant KPProblem à MKPProblem/CP-SAT).
add_executable(bab_kp_gecode
    kp/main_kp_gecode.cpp
    kp/kp_gecode_bab.cpp
)

target_include_directories(bab_kp_gecode PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/kp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/kp
    ${SOLVER_GENERIC_ROOT}/mkp
    ${SOLVER_GENERIC_ROOT}/ssp
    ${SOLVER_GENERIC_ROOT}/covset
    ${GECODE_PREFIX}/include
)
target_compile_definitions(bab_kp_gecode PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

target_link_libraries(bab_kp_gecode PRIVATE
    v1ld_generic_core
    ortools::ortools
    OpenMP::OpenMP_CXX
    -L${GECODE_PREFIX}/lib
    -lgecodeint -lgecodekernel -lgecodesupport -lgecodesearch -lgecodeminimodel
)

target_compile_options(bab_kp_gecode PRIVATE -O3 -march=native)

# ── Exécutable bab_mkp_gecode_collect (BAB + collecte données GNN) ────────────
# Compile mkp/gecode_bab.cpp comme bab_mkp_gecode, donc a besoin des mêmes
# sources/libs pour les deux backends GNN (IPC + natif) référencés par
# solve_gecode_bab_mkp, même si ce binaire n'active jamais bound_mode=GNN.
add_executable(bab_mkp_gecode_collect
    mkp/main_gecode_collect.cpp
    mkp/gecode_bab_collect.cpp
    mkp/gecode_bab.cpp
    mkp/gnn_v1ld_client.cpp
    ${GNN_NATIVE_SOURCES}
)

target_include_directories(bab_mkp_gecode_collect PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/mkp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/mkp
    ${SOLVER_GENERIC_ROOT}/ssp
    ${SOLVER_GENERIC_ROOT}/covset
    ${GECODE_PREFIX}/include
    ${LIBTORCH_INCLUDE_DIRS}
)
target_compile_definitions(bab_mkp_gecode_collect PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

target_link_libraries(bab_mkp_gecode_collect PRIVATE
    v1ld_generic_core
    ortools::ortools
    ${LIBTORCH_LINK_FLAGS}
    OpenMP::OpenMP_CXX
    -L${GECODE_PREFIX}/lib
    -lgecodeint -lgecodekernel -lgecodesupport -lgecodesearch -lgecodeminimodel
)

target_compile_options(bab_mkp_gecode_collect PRIVATE -O3 -march=native)

# ── Exécutable bab_mkp_gecode_base (Gecode CP pur, sans borne Lagrangienne) ──
add_executable(bab_mkp_gecode_base
    mkp/main_gecode_base.cpp
    ../gecode_base/gecode_solver.cpp
)

target_include_directories(bab_mkp_gecode_base PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/mkp
    ${SOLVER_GENERIC_ROOT}
    ${GECODE_PREFIX}/include
)

target_link_libraries(bab_mkp_gecode_base PRIVATE
    -L${GECODE_PREFIX}/lib
    -lgecodeint -lgecodekernel -lgecodesupport -lgecodesearch -lgecodeminimodel
)

target_compile_options(bab_mkp_gecode_base PRIVATE -O3 -march=native)

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

# ── Exécutable bench_ssp_bab (Gecode base vs B&B v1LD vs B&B v1CLD, SSP) ──────
add_executable(bench_ssp_bab
    ssp/bench_ssp_bab.cpp
    ssp/gecode_bab_ssp.cpp
    ../gecode_base/gecode_solver.cpp
)

target_include_directories(bench_ssp_bab PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ssp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/ssp
    ${GECODE_PREFIX}/include
)

target_link_libraries(bench_ssp_bab PRIVATE
    v1ld_generic_core
    ortools::ortools
    OpenMP::OpenMP_CXX
    -L${GECODE_PREFIX}/lib
    -lgecodeint -lgecodekernel -lgecodesupport -lgecodesearch -lgecodeminimodel
)

target_compile_options(bench_ssp_bab PRIVATE -O3 -march=native)

# ── Exécutable bench_ssp_bab_file (idem, instance chargée depuis fichiers texte
#    générés par ssp_stochastic — cf. solver_generic/ssp/export_to_txt.py) ────
add_executable(bench_ssp_bab_file
    ssp/bench_ssp_bab_file.cpp
    ssp/gecode_bab_ssp.cpp
    ../gecode_base/gecode_solver.cpp
)

target_include_directories(bench_ssp_bab_file PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ssp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/ssp
    ${GECODE_PREFIX}/include
)

target_link_libraries(bench_ssp_bab_file PRIVATE
    v1ld_generic_core
    ortools::ortools
    OpenMP::OpenMP_CXX
    -L${GECODE_PREFIX}/lib
    -lgecodeint -lgecodekernel -lgecodesupport -lgecodesearch -lgecodeminimodel
)

target_compile_options(bench_ssp_bab_file PRIVATE -O3 -march=native)

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

# ── Exécutable test_gnn_v1cld_native (validation numérique standalone de
#    l'intégration GNN→borne V1CLD SSP contre la référence Python, étape 2 du
#    portage MKP→SSP, cf. ssp/gnn_v1cld_native.hpp) — pas de dépendance Gecode
#    (SSPCLDProblemBnB/ssp_loader sont indépendants de Gecode, contrairement à
#    SSPSpaceBnB ; le branchement dans le vrai B&B est test_gnn_v1cld_bab
#    ci-dessous, étape 3) ──
add_executable(test_gnn_v1cld_native
    ssp/test_gnn_v1cld_native.cpp
    ssp/gnn_v1cld_native.cpp
)

target_include_directories(test_gnn_v1cld_native PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ssp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/ssp
    ${LIBTORCH_INCLUDE_DIRS}
)
target_compile_definitions(test_gnn_v1cld_native PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

target_link_libraries(test_gnn_v1cld_native PRIVATE
    v1ld_generic_core
    ortools::ortools
    OpenMP::OpenMP_CXX
    ${LIBTORCH_LINK_FLAGS}
)

target_compile_options(test_gnn_v1cld_native PRIVATE -O3 -march=native)

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

# ── Exécutable test_gnn_v1cld_bab_batched_skeleton (étape B1 du batching
#    multi-nœuds SSP, cf. ssp/gecode_bab_ssp_gnn_batched.hpp) : valide la DFS
#    manuelle + bucketing (G,N_z) SEULE, backend stub sans borne (équivalent
#    REF) -- aucune dépendance LibTorch, le GNN vient à l'étape B2 ──
add_executable(test_gnn_v1cld_bab_batched_skeleton
    ssp/test_gnn_v1cld_bab_batched_skeleton.cpp
    ssp/gecode_bab_ssp_gnn_batched.cpp
)

target_include_directories(test_gnn_v1cld_bab_batched_skeleton PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ssp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/ssp
    ${GECODE_PREFIX}/include
)

target_link_libraries(test_gnn_v1cld_bab_batched_skeleton PRIVATE
    v1ld_generic_core
    ortools::ortools
    OpenMP::OpenMP_CXX
    -L${GECODE_PREFIX}/lib
    -lgecodeint -lgecodekernel -lgecodesupport -lgecodesearch -lgecodeminimodel
)

target_compile_options(test_gnn_v1cld_bab_batched_skeleton PRIVATE -O3 -march=native)

# ── Exécutable test_gnn_v1cld_ssp_native_batched (étape B2 du batching
#    multi-nœuds SSP, cf. ssp/gnn_v1cld_native_batched.hpp) : valide
#    query_batch() contre la référence Python ET contre le mode séquentiel
#    (GNNV1CLDNative::query, déjà validé étape 2) -- suffixe _ssp pour éviter
#    la collision de nom avec test_gnn_v1cld_native_batched (MKP, plus bas
#    dans ce fichier) -- gecode_bab_ssp_gnn_batched.hpp (inclus pour
#    SSPGNNBoundBackendBatch) tire les types Gecode de gecode_bab_ssp.hpp,
#    donc les libs Gecode sont liées par précaution même si aucun Space n'est
#    jamais instancié ici ──
add_executable(test_gnn_v1cld_ssp_native_batched
    ssp/test_gnn_v1cld_native_batched.cpp
    ssp/gnn_v1cld_native.cpp
    ssp/gnn_v1cld_native_batched.cpp
)

target_include_directories(test_gnn_v1cld_ssp_native_batched PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ssp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/ssp
    ${GECODE_PREFIX}/include
    ${LIBTORCH_INCLUDE_DIRS}
)
target_compile_definitions(test_gnn_v1cld_ssp_native_batched PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

target_link_libraries(test_gnn_v1cld_ssp_native_batched PRIVATE
    v1ld_generic_core
    ortools::ortools
    OpenMP::OpenMP_CXX
    ${LIBTORCH_LINK_FLAGS}
    -L${GECODE_PREFIX}/lib
    -lgecodeint -lgecodekernel -lgecodesupport -lgecodesearch -lgecodeminimodel
)

target_compile_options(test_gnn_v1cld_ssp_native_batched PRIVATE -O3 -march=native)

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

# ── Exécutable debug_root_bound (diagnostic temporaire — validité de la borne
#    v1CLD à la racine, cf. investigation du FAIL sanity check E=4,T=8,N=20) ──
add_executable(debug_root_bound
    ssp/debug_root_bound.cpp
    ssp/gecode_bab_ssp.cpp
    ../gecode_base/gecode_solver.cpp
)

target_include_directories(debug_root_bound PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ssp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/ssp
    ${GECODE_PREFIX}/include
)

target_link_libraries(debug_root_bound PRIVATE
    v1ld_generic_core
    ortools::ortools
    OpenMP::OpenMP_CXX
    -L${GECODE_PREFIX}/lib
    -lgecodeint -lgecodekernel -lgecodesupport -lgecodesearch -lgecodeminimodel
)

target_compile_options(debug_root_bound PRIVATE -O3 -march=native)

# ── Variante debug_root_bound_v1ld (test de controle, 2026-08-12 -- v1LD n'a
#    pas d'approximation dans Delta_i, sert a isoler si l'invalidite observee
#    sur v1CLD est bien specifique a sa formule d'agregation) ────────────────
add_executable(debug_root_bound_v1ld
    ssp/debug_root_bound_v1ld.cpp
    ssp/gecode_bab_ssp.cpp
    ../gecode_base/gecode_solver.cpp
)

target_include_directories(debug_root_bound_v1ld PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ssp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/ssp
    ${GECODE_PREFIX}/include
)

target_link_libraries(debug_root_bound_v1ld PRIVATE
    v1ld_generic_core
    ortools::ortools
    OpenMP::OpenMP_CXX
    -L${GECODE_PREFIX}/lib
    -lgecodeint -lgecodekernel -lgecodesupport -lgecodesearch -lgecodeminimodel
)

target_compile_options(debug_root_bound_v1ld PRIVATE -O3 -march=native)

# ── Exécutable debug_warmstart_rounds (diagnostic temporaire — compare v1LD
#    et v1CLD sous le meme protocole de rafales courtes + warm-start) ─────────
add_executable(debug_warmstart_rounds
    ssp/debug_warmstart_rounds.cpp
    ssp/gecode_bab_ssp.cpp
    ../gecode_base/gecode_solver.cpp
)

target_include_directories(debug_warmstart_rounds PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ssp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/ssp
    ${GECODE_PREFIX}/include
)

target_link_libraries(debug_warmstart_rounds PRIVATE
    v1ld_generic_core
    ortools::ortools
    -L${GECODE_PREFIX}/lib
    -lgecodeint -lgecodekernel -lgecodesupport -lgecodesearch -lgecodeminimodel
)

target_compile_options(debug_warmstart_rounds PRIVATE -O3 -march=native)

# ── Exécutable debug_verify_oracle (diagnostic temporaire — verification force
#    brute de l'exactitude de theta/zeta, independante de Gecode) ────────────
add_executable(debug_verify_oracle
    ssp/debug_verify_oracle.cpp
    ${SOLVER_GENERIC_ROOT}/ssp/ssp_loader.cpp
)

target_include_directories(debug_verify_oracle PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ssp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/ssp
)

target_link_libraries(debug_verify_oracle PRIVATE
    v1ld_generic_core
    ortools::ortools
)

target_compile_options(debug_verify_oracle PRIVATE -O3 -march=native)

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

# ── Variante debug_replay_violation_rounded (test hypothese scaling/troncature
#    SCALE=1000, 2026-08-12 -- cf. salary_rounded EXPERIMENTAL dans
#    gecode_bab_ssp.hpp) — temporaire, a retirer une fois l'hypothese tranchee.
add_executable(debug_replay_violation_rounded
    ssp/debug_replay_violation_rounded.cpp
    ${SOLVER_GENERIC_ROOT}/ssp/ssp_loader.cpp
)

target_include_directories(debug_replay_violation_rounded PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ssp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/ssp
)

target_link_libraries(debug_replay_violation_rounded PRIVATE
    v1ld_generic_core
    ortools::ortools
)

target_compile_options(debug_replay_violation_rounded PRIVATE -O3 -march=native)

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

# ── Exécutable debug_capture_v1ld (diagnostic temporaire — lance UNIQUEMENT
#    le B&B v1LD, pour capturer un dump SSP_DEBUG_DUMP specifique au mode LD
#    sans que v1CLD ne le capture en premier, cf. bench_ssp_bab_file) ────────
add_executable(debug_capture_v1ld
    ssp/debug_capture_v1ld.cpp
    ssp/gecode_bab_ssp.cpp
    ../gecode_base/gecode_solver.cpp
)

target_include_directories(debug_capture_v1ld PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ssp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/ssp
    ${GECODE_PREFIX}/include
)

target_link_libraries(debug_capture_v1ld PRIVATE
    v1ld_generic_core
    ortools::ortools
    -L${GECODE_PREFIX}/lib
    -lgecodeint -lgecodekernel -lgecodesupport -lgecodesearch -lgecodeminimodel
)

target_compile_options(debug_capture_v1ld PRIVATE -O3 -march=native)

# ── Exécutable debug_verify_v1ld_zeta (diagnostic temporaire — verifie
#    zeta_i (v1LD, CP-SAT) contre une force brute JOINTE sur les E employes,
#    teste specifiquement l'hypothese d'arrondi CPSAT_SCALE) ─────────────────
add_executable(debug_verify_v1ld_zeta
    ssp/debug_verify_v1ld_zeta.cpp
    ${SOLVER_GENERIC_ROOT}/ssp/ssp_loader.cpp
)

target_include_directories(debug_verify_v1ld_zeta PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ssp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/ssp
)

target_link_libraries(debug_verify_v1ld_zeta PRIVATE
    v1ld_generic_core
    ortools::ortools
)

target_compile_options(debug_verify_v1ld_zeta PRIVATE -O3 -march=native)

# ── Exécutable bench_v1ld_subproblems (benchmark élémentaire — coût des
#    sous-problèmes θ/ζ d'UNE itération v1LD (MKP), λ=0, racine, sans B&B,
#    sur un lot d'instances — cf. solver_bab/mkp/instances_v1ld_bench/) ──────
add_executable(bench_v1ld_subproblems
    mkp/bench_v1ld_subproblems.cpp
    mkp/zeta_gecode_solver.cpp
)

target_include_directories(bench_v1ld_subproblems PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/mkp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/mkp
    ${GECODE_PREFIX}/include
)

target_link_libraries(bench_v1ld_subproblems PRIVATE
    v1ld_generic_core
    ortools::ortools
    -L${GECODE_PREFIX}/lib
    -lgecodeint -lgecodekernel -lgecodesupport -lgecodesearch -lgecodeminimodel
)

target_compile_options(bench_v1ld_subproblems PRIVATE -O3 -march=native)

# ── Exécutable bench_v1cld_subproblems (benchmark élémentaire — ζ_{i,j}
#    V1CLD (MKP), λ=0, deux méthodes : bnb (production) vs dp (jamais utilisée
#    en C++ avant ce benchmark) — cf. solver_bab/mkp/bench_v1cld_subproblems.cpp).
#    Pas besoin de Gecode/OR-Tools : MKPCLDProblem/MKPCLDProblemDP vivent
#    entièrement dans v1ld_generic_core (solver_generic/mkp/). ───────────────
add_executable(bench_v1cld_subproblems
    mkp/bench_v1cld_subproblems.cpp
)

target_include_directories(bench_v1cld_subproblems PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/mkp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/mkp
)

target_link_libraries(bench_v1cld_subproblems PRIVATE
    v1ld_generic_core
)

target_compile_options(bench_v1cld_subproblems PRIVATE -O3 -march=native)

# ── Exécutable debug_zeta_gecode (diagnostic temporaire — trouve le premier
#    scénario où solve_zeta_gecode diverge de CP-SAT, pour localiser un bug) ──
add_executable(debug_zeta_gecode
    mkp/debug_zeta_gecode.cpp
    mkp/zeta_gecode_solver.cpp
)

target_include_directories(debug_zeta_gecode PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/mkp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/mkp
    ${GECODE_PREFIX}/include
)

target_link_libraries(debug_zeta_gecode PRIVATE
    v1ld_generic_core
    ortools::ortools
    -L${GECODE_PREFIX}/lib
    -lgecodeint -lgecodekernel -lgecodesupport -lgecodesearch -lgecodeminimodel
)

target_compile_options(debug_zeta_gecode PRIVATE -O3 -march=native)

# ── Exécutable gnn_native_probe (diagnostic : GNNV1LDNative::query() isolé, ─
#    sans Gecode/B&B — cf. session "load train checkpoint into GECODE BAB") ──
add_executable(gnn_native_probe
    mkp/gnn_native_probe.cpp
    ${GNN_NATIVE_SOURCES}
)

target_include_directories(gnn_native_probe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${SOLVER_GENERIC_ROOT}
    ${LIBTORCH_INCLUDE_DIRS}
)
target_compile_definitions(gnn_native_probe PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

target_link_libraries(gnn_native_probe PRIVATE
    ${LIBTORCH_LINK_FLAGS}
    OpenMP::OpenMP_CXX
)

target_compile_options(gnn_native_probe PRIVATE -O3 -march=native)

# ── Exécutable gnn_sdedit_aoti_probe (diagnostic : GNNSdEditNativeAOTI::
#    query() isolé, sans Gecode/B&B — pour isoler le TIMEOUT à 0 nœuds
#    observé dans le vrai B&B avec gnn_backend=native_aoti) ─────────────────
add_executable(gnn_sdedit_aoti_probe
    mkp/gnn_sdedit_aoti_probe.cpp
    ${GNN_NATIVE_SOURCES}
)

target_include_directories(gnn_sdedit_aoti_probe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${SOLVER_GENERIC_ROOT}
    ${LIBTORCH_INCLUDE_DIRS}
)
target_compile_definitions(gnn_sdedit_aoti_probe PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

target_link_libraries(gnn_sdedit_aoti_probe PRIVATE
    ${LIBTORCH_LINK_FLAGS}
    OpenMP::OpenMP_CXX
)

target_compile_options(gnn_sdedit_aoti_probe PRIVATE -O3 -march=native)

# ── Exécutable gnn_batch_consistency_probe (Étape 2d du plan de batching :
#    GNNSdEditNativeBatched::query_batch() vs B appels de GNNSdEditNative::
#    query() [production, INCHANGÉ], sur une vraie instance MKP) ────────────
add_executable(gnn_batch_consistency_probe
    mkp/gnn_batch_consistency_probe.cpp
    mkp/gnn_sd_edit_native_batched.cpp
    ${GNN_NATIVE_SOURCES}
)

target_include_directories(gnn_batch_consistency_probe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/mkp
    ${SOLVER_GENERIC_ROOT}
    ${LIBTORCH_INCLUDE_DIRS}
)
target_compile_definitions(gnn_batch_consistency_probe PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

target_link_libraries(gnn_batch_consistency_probe PRIVATE
    ${LIBTORCH_LINK_FLAGS}
    OpenMP::OpenMP_CXX
)

target_compile_options(gnn_batch_consistency_probe PRIVATE -O3 -march=native)

# ── Exécutable test_gnn_v1cld_native_batched (validation autonome du backend
#    GNN V1CLD AVANT tout branchement B&B — cf. test_gnn_v1cld_native_batched.cpp
#    et test_paper/validate_v1cld_native_reference.py, la référence Python
#    compagnon). Fichiers séparés du backend V1LD (gnn_sd_edit_native_batched.cpp) —
#    aucune modification de la production. ──────────────────────────────────
add_executable(test_gnn_v1cld_native_batched
    mkp/test_gnn_v1cld_native_batched.cpp
    mkp/gnn_v1cld_sd_edit_native_batched.cpp
    ${GNN_NATIVE_SOURCES}
)

target_include_directories(test_gnn_v1cld_native_batched PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/mkp
    ${SOLVER_GENERIC_ROOT}
    ${LIBTORCH_INCLUDE_DIRS}
)
target_compile_definitions(test_gnn_v1cld_native_batched PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

target_link_libraries(test_gnn_v1cld_native_batched PRIVATE
    ${LIBTORCH_LINK_FLAGS}
    OpenMP::OpenMP_CXX
)

target_compile_options(test_gnn_v1cld_native_batched PRIVATE -O3 -march=native)

# ── Exécutable gnn_cudagraph_probe (diagnostic : faisabilité CUDA graphs pour
#    amortir l'overhead de dispatch du forward GNN natif — cf. mémoire
#    "sdedit-vs-homogene-forward-overhead-confirmed") — ne dépend PAS de
#    GNN_NATIVE_SOURCES/OpenMP, charge un module scripté directement ─────────
add_executable(gnn_cudagraph_probe
    mkp/gnn_cudagraph_probe.cpp
)

target_include_directories(gnn_cudagraph_probe PRIVATE
    ${LIBTORCH_INCLUDE_DIRS}
)
target_compile_definitions(gnn_cudagraph_probe PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

target_link_libraries(gnn_cudagraph_probe PRIVATE
    ${LIBTORCH_LINK_FLAGS}
    -lcudart
)

target_compile_options(gnn_cudagraph_probe PRIVATE -O3 -march=native)

# ── Exécutable gnn_batch_forward_probe (diagnostic : gain réel du batching de
#    B nœuds B&B en un seul appel GNN — go/no-go le moins cher de l'Étape 2
#    du plan de batching, cf. gnn_sd_edit_scriptable_batched.py) — ne dépend
#    PAS de GNN_NATIVE_SOURCES/OpenMP, charge deux modules scriptés
#    directement (non batché + batché) ───────────────────────────────────────
add_executable(gnn_batch_forward_probe
    mkp/gnn_batch_forward_probe.cpp
)

target_include_directories(gnn_batch_forward_probe PRIVATE
    ${LIBTORCH_INCLUDE_DIRS}
)
target_compile_definitions(gnn_batch_forward_probe PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

target_link_libraries(gnn_batch_forward_probe PRIVATE
    ${LIBTORCH_LINK_FLAGS}
)

target_compile_options(gnn_batch_forward_probe PRIVATE -O3 -march=native)

# ── Exécutable gnn_dense_msgpassing_probe (diagnostic : gain de la
#    reformulation dense du message passing sd_edit — sans scatter/gather —
#    validée numériquement équivalente dans
#    gnn_sd_edit/prototype_dense_msgpassing.py) ─────────────────────────────
add_executable(gnn_dense_msgpassing_probe
    mkp/gnn_dense_msgpassing_probe.cpp
)

target_include_directories(gnn_dense_msgpassing_probe PRIVATE
    ${LIBTORCH_INCLUDE_DIRS}
)
target_compile_definitions(gnn_dense_msgpassing_probe PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

target_link_libraries(gnn_dense_msgpassing_probe PRIVATE
    ${LIBTORCH_LINK_FLAGS}
)

target_compile_options(gnn_dense_msgpassing_probe PRIVATE -O3 -march=native)

# ── Exécutable gnn_edge_vectorized_probe (diagnostic : construction scalaire
#    vs vectorisée torch des arêtes du graphe GNN — cf. décomposition
#    "arêtes = 59% de la construction de graphe", job 16028283) — ne dépend
#    d'aucun checkpoint/modèle, ni de GNN_NATIVE_SOURCES/OpenMP ─────────────
add_executable(gnn_edge_vectorized_probe
    mkp/gnn_edge_vectorized_probe.cpp
)

target_include_directories(gnn_edge_vectorized_probe PRIVATE
    ${LIBTORCH_INCLUDE_DIRS}
)
target_compile_definitions(gnn_edge_vectorized_probe PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

target_link_libraries(gnn_edge_vectorized_probe PRIVATE
    ${LIBTORCH_LINK_FLAGS}
)

target_compile_options(gnn_edge_vectorized_probe PRIVATE -O3 -march=native)

# ── Exécutable gnn_aoti_compat_probe (diagnostic : un artefact AOTInductor
#    compilé par le venv Python (torch 2.12.1) est-il chargeable par le
#    runtime C++ figé sur libtorch 2.5.1 ? — ÉTAPE A de l'investigation
#    AOTInductor, cf. gnn_sd_edit/aoti_compat_probe.py) ─────────────────────
add_executable(gnn_aoti_compat_probe
    mkp/gnn_aoti_compat_probe.cpp
)

target_include_directories(gnn_aoti_compat_probe PRIVATE
    ${LIBTORCH_INCLUDE_DIRS}
)
target_compile_definitions(gnn_aoti_compat_probe PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
if(GNN_DEVICE STREQUAL "cuda")
    target_compile_definitions(gnn_aoti_compat_probe PRIVATE GNN_AOTI_WITH_CUDA)
endif()

target_link_libraries(gnn_aoti_compat_probe PRIVATE
    ${LIBTORCH_LINK_FLAGS}
)

target_compile_options(gnn_aoti_compat_probe PRIVATE -O3 -march=native)

# ── Exécutable gnn_aoti_sdedit_probe (mesure wall-clock RÉELLE du gain
#    AOTInductor vs TorchScript sur le VRAI modèle sd_edit, cf.
#    gnn_sd_edit/export_aoti_sdedit.py) — CUDA uniquement (compare le
#    forward GPU, pas de chemin CPU) ─────────────────────────────────────────
if(GNN_DEVICE STREQUAL "cuda")
    add_executable(gnn_aoti_sdedit_probe
        mkp/gnn_aoti_sdedit_probe.cpp
    )

    target_include_directories(gnn_aoti_sdedit_probe PRIVATE
        ${LIBTORCH_INCLUDE_DIRS}
    )
    target_compile_definitions(gnn_aoti_sdedit_probe PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

    target_link_libraries(gnn_aoti_sdedit_probe PRIVATE
        ${LIBTORCH_LINK_FLAGS}
    )

    target_compile_options(gnn_aoti_sdedit_probe PRIVATE -O3 -march=native)
endif()

# ── Exécutable gnn_cudagraph_batched_probe (diagnostic : CUDA graphs sur le
#    modèle BATCHÉ sd_edit après freeze+optimize_for_inference — baseline de
#    production actuelle — cf. gnn_cudagraph_probe pour la version non
#    batchée, gain ~10% validé) — charge le module scripté batché via
#    ensure_scripted_batched_export (spawn python si cache absent/périmé) ────
add_executable(gnn_cudagraph_batched_probe
    mkp/gnn_cudagraph_batched_probe.cpp
)

target_include_directories(gnn_cudagraph_batched_probe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/mkp
    ${LIBTORCH_INCLUDE_DIRS}
)
target_compile_definitions(gnn_cudagraph_batched_probe PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

target_link_libraries(gnn_cudagraph_batched_probe PRIVATE
    ${LIBTORCH_LINK_FLAGS}
    -lcudart
)

target_compile_options(gnn_cudagraph_batched_probe PRIVATE -O3 -march=native)

# ── Exécutable debug_mkp_root_bound (test empirique, 2026-08-12 -- meme
#    question que debug_root_bound_v1ld/SSP mais pour MKP : le meme defaut
#    d'agregation Delta_i (v1cld_solver.cpp, partage entre MKP et SSP) se
#    manifeste-t-il aussi empiriquement sur MKP ?) ──────────────────────────
add_executable(debug_mkp_root_bound
    mkp/debug_mkp_root_bound.cpp
    ${SOLVER_GENERIC_ROOT}/mkp/mkp_cld_problem.cpp
)

target_include_directories(debug_mkp_root_bound PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/mkp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/mkp
)

target_link_libraries(debug_mkp_root_bound PRIVATE
    v1ld_generic_core
    ortools::ortools
)

target_compile_options(debug_mkp_root_bound PRIVATE -O3 -march=native)

# ── Exécutable bench_ssp_bab_file_scentered (diagnostic, 2026-08-13) : meme
#    comparaison Gecode base / B&B v1CLD / B&B v1LD que bench_ssp_bab_file,
#    mais B&B v1CLD utilise solve_v1cld_scentered (centrage PAR SCENARIO)
#    au lieu du centrage global -- teste la validite sur une recherche
#    complete (des milliers de noeuds), pas juste un noeud capture a la main.
#    Fichiers separes de gecode_bab_ssp.hpp/.cpp et v1cld_solver.cpp : ne
#    touche a rien du code de production existant.
add_executable(bench_ssp_bab_file_scentered
    ssp/bench_ssp_bab_file_scentered.cpp
    ssp/gecode_bab_ssp_scentered.cpp
    ssp/v1cld_solver_scentered.cpp
    ../gecode_base/gecode_solver.cpp
)

target_include_directories(bench_ssp_bab_file_scentered PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ssp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/ssp
    ${GECODE_PREFIX}/include
)

target_link_libraries(bench_ssp_bab_file_scentered PRIVATE
    v1ld_generic_core
    ortools::ortools
    OpenMP::OpenMP_CXX
    -L${GECODE_PREFIX}/lib
    -lgecodeint -lgecodekernel -lgecodesupport -lgecodesearch -lgecodeminimodel
)

target_compile_options(bench_ssp_bab_file_scentered PRIVATE -O3 -march=native)
