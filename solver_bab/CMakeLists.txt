cmake_minimum_required(VERSION 3.14)
project(solver_bab_mkp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── OR-Tools (nécessaire pour MKPProblem / CP-SAT dans v1ld_generic_core) ─────
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew" "/usr/local")
find_package(ortools CONFIG REQUIRED)

# ── Bibliothèque générique pré-compilée (solver_generic) ─────────────────────
set(SOLVER_GENERIC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../solver_generic")
set(SOLVER_GENERIC_BUILD "${SOLVER_GENERIC_ROOT}/build")

add_library(v1ld_generic_core STATIC IMPORTED)
set_target_properties(v1ld_generic_core PROPERTIES
    IMPORTED_LOCATION "${SOLVER_GENERIC_BUILD}/libv1ld_generic_core.a"
    INTERFACE_INCLUDE_DIRECTORIES
        "${SOLVER_GENERIC_ROOT};${SOLVER_GENERIC_ROOT}/mkp;${SOLVER_GENERIC_ROOT}/ssp;${SOLVER_GENERIC_ROOT}/covset"
)

# ── Gecode ────────────────────────────────────────────────────────────────────
set(GECODE_PREFIX "/opt/homebrew/opt/gecode")

# ── Exécutable bab_mkp_gecode (Gecode BAB + borne V1LD) ──────────────────────
add_executable(bab_mkp_gecode
    main_gecode.cpp
    gecode_bab.cpp
)

target_include_directories(bab_mkp_gecode PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
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
