cmake_minimum_required(VERSION 3.14)
project(sld_generic_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── OR-Tools (CP-SAT pour MKP) ────────────────────────────────────────────────
# Installation : brew install or-tools
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew" "/usr/local")
find_package(ortools CONFIG REQUIRED)

# ── OpenMP (parallélisation de la boucle sur les N scénarios dans sld_solver.cpp) ──
find_package(OpenMP REQUIRED)

# ── Bibliothèque statique : corps générique + problèmes concrets ──────────────
add_library(sld_generic_core STATIC
    sld_solver.cpp
    dld_solver.cpp
    dld_solver_parallel.cpp
    mkp/mkp_problem.cpp
    mkp/mkp_problem_bnb.cpp
    mkp/mkp_problem_bnb_surrogate.cpp
    mkp/mkp_dld_problem.cpp
    mkp/mkp_dld_problem_dp.cpp
    ssp/ssp_problem.cpp
    ssp/ssp_dld_problem.cpp
    ssp/ssp_loader.cpp
)
target_include_directories(sld_generic_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/mkp
    ${CMAKE_CURRENT_SOURCE_DIR}/ssp
)

# ── Exécutables de test et benchmark SSP ─────────────────────────────────────
add_executable(test_ssp_shift ssp/test_ssp.cpp)
target_link_libraries(test_ssp_shift PRIVATE sld_generic_core)
target_compile_options(test_ssp_shift PRIVATE -O2)

add_executable(test_ssp_bnb ssp/test_ssp_bnb.cpp)
target_link_libraries(test_ssp_bnb PRIVATE sld_generic_core)
target_compile_options(test_ssp_bnb PRIVATE -O2)

target_compile_options(sld_generic_core PRIVATE -O3 -march=native)
target_link_libraries(sld_generic_core PUBLIC ortools::ortools OpenMP::OpenMP_CXX)

