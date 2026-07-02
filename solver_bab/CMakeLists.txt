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
        "${SOLVER_GENERIC_ROOT};${SOLVER_GENERIC_ROOT}/mkp;${SOLVER_GENERIC_ROOT}/ssp;${SOLVER_GENERIC_ROOT}/covset"
)

# ── Gecode ────────────────────────────────────────────────────────────────────
# Surchargeable via -DGECODE_PREFIX=... (Compute Canada, Linux, etc.)
if(NOT DEFINED GECODE_PREFIX)
    set(GECODE_PREFIX "/opt/homebrew/opt/gecode")
endif()

# ── Exécutable bab_mkp_gecode (Gecode BAB + borne V1LD) ──────────────────────
add_executable(bab_mkp_gecode
    mkp/main_gecode.cpp
    mkp/gecode_bab.cpp
)

target_include_directories(bab_mkp_gecode PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/mkp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/mkp
    ${SOLVER_GENERIC_ROOT}/ssp
    ${SOLVER_GENERIC_ROOT}/covset
    ${GECODE_PREFIX}/include
)

target_link_libraries(bab_mkp_gecode PRIVATE
    v1ld_generic_core
    ortools::ortools
    -L${GECODE_PREFIX}/lib
    -lgecodeint -lgecodekernel -lgecodesupport -lgecodesearch -lgecodeminimodel
)

target_compile_options(bab_mkp_gecode PRIVATE -O3 -march=native)

# ── Exécutable bab_mkp_gecode_collect (BAB + collecte données GNN) ────────────
add_executable(bab_mkp_gecode_collect
    mkp/main_gecode_collect.cpp
    mkp/gecode_bab_collect.cpp
    mkp/gecode_bab.cpp
)

target_include_directories(bab_mkp_gecode_collect PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/mkp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/mkp
    ${SOLVER_GENERIC_ROOT}/ssp
    ${SOLVER_GENERIC_ROOT}/covset
    ${GECODE_PREFIX}/include
)

target_link_libraries(bab_mkp_gecode_collect PRIVATE
    v1ld_generic_core
    ortools::ortools
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
add_executable(bab_mkp_gecode_base_collect
    ../gecode_base/main_collect_gecode_base.cpp
    ../gecode_base/collect_gecode_base.cpp
    mkp/gecode_bab.cpp
)

target_include_directories(bab_mkp_gecode_base_collect PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/mkp
    ${SOLVER_GENERIC_ROOT}
    ${SOLVER_GENERIC_ROOT}/mkp
    ${SOLVER_GENERIC_ROOT}/ssp
    ${SOLVER_GENERIC_ROOT}/covset
    ${GECODE_PREFIX}/include
)

target_link_libraries(bab_mkp_gecode_base_collect PRIVATE
    v1ld_generic_core
    ortools::ortools
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
    -L${GECODE_PREFIX}/lib
    -lgecodeint -lgecodekernel -lgecodesupport -lgecodesearch -lgecodeminimodel
)

target_compile_options(bench_ssp_bab PRIVATE -O3 -march=native)
