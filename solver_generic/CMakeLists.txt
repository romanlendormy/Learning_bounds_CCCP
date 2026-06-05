cmake_minimum_required(VERSION 3.14)
project(v1ld_generic_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── OR-Tools (CP-SAT pour MKP) ────────────────────────────────────────────────
# Installation : brew install or-tools
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew" "/usr/local")
find_package(ortools CONFIG REQUIRED)

# ── Python (conda 3.11) ───────────────────────────────────────────────────────
set(PYTHON_EXECUTABLE
    "/Users/roman.lendormy/opt/miniconda3/bin/python"
    CACHE FILEPATH "" FORCE
)
set(PYTHON_INCLUDE_DIR
    "/Users/roman.lendormy/opt/miniconda3/include/python3.11"
    CACHE PATH "" FORCE
)

# ── pybind11 ──────────────────────────────────────────────────────────────────
set(pybind11_DIR
    "/Users/roman.lendormy/opt/miniconda3/lib/python3.11/site-packages/pybind11/share/cmake/pybind11"
)
find_package(pybind11 CONFIG REQUIRED)

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
    covset/covset_problem.cpp
    covset/covset_cld_problem.cpp
)
target_include_directories(v1ld_generic_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/mkp
    ${CMAKE_CURRENT_SOURCE_DIR}/ssp
    ${CMAKE_CURRENT_SOURCE_DIR}/covset
)
target_compile_options(v1ld_generic_core PRIVATE -O3 -march=native)
target_link_libraries(v1ld_generic_core PUBLIC ortools::ortools)

# ── Module Python v1LD générique ──────────────────────────────────────────────
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

