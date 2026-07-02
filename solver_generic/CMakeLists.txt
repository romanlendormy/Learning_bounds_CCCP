cmake_minimum_required(VERSION 3.14)
project(v1ld_generic_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── OR-Tools (CP-SAT pour MKP) ────────────────────────────────────────────────
# Installation : brew install or-tools
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew" "/usr/local")
find_package(ortools CONFIG REQUIRED)

# ── Python (chemins par défaut macOS ; surchargeable via -D sur autre plateforme) ──
if(NOT DEFINED PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE
        "/Users/roman.lendormy/opt/miniconda3/bin/python"
        CACHE FILEPATH "" FORCE
    )
endif()
if(NOT DEFINED PYTHON_INCLUDE_DIR)
    set(PYTHON_INCLUDE_DIR
        "/Users/roman.lendormy/opt/miniconda3/include/python3.11"
        CACHE PATH "" FORCE
    )
endif()

# ── pybind11 (optionnel : nécessaire uniquement pour le module Python) ─────────
if(NOT DEFINED pybind11_DIR)
    set(pybind11_DIR
        "/Users/roman.lendormy/opt/miniconda3/lib/python3.11/site-packages/pybind11/share/cmake/pybind11"
    )
endif()
find_package(pybind11 CONFIG QUIET)

# ── Bibliothèque statique : corps générique + problèmes concrets ──────────────
add_library(v1ld_generic_core STATIC
    v1ld_solver.cpp
    v1cld_solver.cpp
    mkp/mkp_problem.cpp
    mkp/mkp_problem_bnb.cpp
    mkp/mkp_cld_problem.cpp
    mkp/mkp_cld_problem_dp.cpp
    ssp/ssp_problem.cpp
    ssp/ssp_cld_problem.cpp
    ssp/ssp_cld2_problem.cpp
    ssp/ssp_loader.cpp
    covset/covset_problem.cpp
    covset/covset_cld_problem.cpp
)
target_include_directories(v1ld_generic_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/mkp
    ${CMAKE_CURRENT_SOURCE_DIR}/ssp
    ${CMAKE_CURRENT_SOURCE_DIR}/covset
)

# ── Exécutables de test et benchmark SSP ─────────────────────────────────────
add_executable(test_ssp_shift ssp/test_ssp.cpp)
target_link_libraries(test_ssp_shift PRIVATE v1ld_generic_core)
target_compile_options(test_ssp_shift PRIVATE -O2)

add_executable(test_ssp_bnb ssp/test_ssp_bnb.cpp)
target_link_libraries(test_ssp_bnb PRIVATE v1ld_generic_core)
target_compile_options(test_ssp_bnb PRIVATE -O2)

add_executable(bench_ssp tests/bench_ssp.cpp)
target_link_libraries(bench_ssp PRIVATE v1ld_generic_core)
target_compile_options(bench_ssp PRIVATE -O3 -march=native)

add_executable(bench_ssp_v1cld tests/bench_ssp_v1cld.cpp)
target_link_libraries(bench_ssp_v1cld PRIVATE v1ld_generic_core)
target_compile_options(bench_ssp_v1cld PRIVATE -O3 -march=native)

add_executable(bench_ssp_v1cld2 tests/bench_ssp_v1cld2.cpp)
target_link_libraries(bench_ssp_v1cld2 PRIVATE v1ld_generic_core)
target_compile_options(bench_ssp_v1cld2 PRIVATE -O3 -march=native)

target_compile_options(v1ld_generic_core PRIVATE -O3 -march=native)
target_link_libraries(v1ld_generic_core PUBLIC ortools::ortools)

# ── Gecode (pour bench_ssp : optimal exact via BAB sans borne) ────────────────
set(GECODE_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/../external/gecode"
    CACHE PATH "Répertoire d'installation Gecode")
add_library(gecode_solver_core STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../gecode_base/gecode_solver.cpp)
target_include_directories(gecode_solver_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../gecode_base
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GECODE_PREFIX}/include)
target_compile_options(gecode_solver_core PRIVATE -O2)
target_link_libraries(gecode_solver_core PUBLIC
    -L${GECODE_PREFIX}/lib
    -lgecodeint -lgecodekernel -lgecodesupport -lgecodesearch -lgecodeminimodel)
target_link_libraries(bench_ssp PRIVATE gecode_solver_core)
target_link_libraries(bench_ssp_v1cld PRIVATE gecode_solver_core)
target_link_libraries(bench_ssp_v1cld2 PRIVATE gecode_solver_core)

# ── Module Python v1LD générique (uniquement si pybind11 est disponible) ──────
if(pybind11_FOUND)
    pybind11_add_module(v1ld_generic_cpp bindings.cpp)
    target_link_libraries(v1ld_generic_cpp PRIVATE v1ld_generic_core)
    target_compile_options(v1ld_generic_cpp PRIVATE -O3 -march=native)

    add_custom_command(TARGET v1ld_generic_cpp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/../lib
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:v1ld_generic_cpp>
            ${CMAKE_CURRENT_SOURCE_DIR}/../lib/$<TARGET_FILE_NAME:v1ld_generic_cpp>
        COMMENT "Copie v1ld_generic_cpp dans lib/"
    )
else()
    message(STATUS "pybind11 non trouvé — module Python ignoré (seule libv1ld_generic_core.a sera compilée)")
endif()

