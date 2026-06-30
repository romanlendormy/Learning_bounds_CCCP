cmake_minimum_required(VERSION 3.14)
project(gecode_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Python (conda 3.11) ───────────────────────────────────────────────────────
set(PYTHON_EXECUTABLE
    "/Users/roman.lendormy/opt/miniconda3/bin/python"
    CACHE FILEPATH "" FORCE
)

# ── pybind11 ──────────────────────────────────────────────────────────────────
set(pybind11_DIR
    "/Users/roman.lendormy/opt/miniconda3/lib/python3.11/site-packages/pybind11/share/cmake/pybind11"
)
find_package(pybind11 CONFIG REQUIRED)

# ── Gecode (brew) ─────────────────────────────────────────────────────────────
set(GECODE_PREFIX "/opt/homebrew/opt/gecode")

# ── Bibliothèque statique : solveur Gecode ────────────────────────────────────
add_library(gecode_solver_core STATIC gecode_solver.cpp)
target_include_directories(gecode_solver_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}           # gecode_solver.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../solver_generic  # types.hpp
    ${GECODE_PREFIX}/include
)
target_compile_options(gecode_solver_core PRIVATE -O3 -march=native)
target_link_libraries(gecode_solver_core PUBLIC
    -L${GECODE_PREFIX}/lib
    -lgecodeint -lgecodekernel -lgecodesupport -lgecodesearch -lgecodeminimodel
)

# ── Module Python gecode_cpp ──────────────────────────────────────────────────
pybind11_add_module(gecode_cpp gecode_bindings.cpp)
target_link_libraries(gecode_cpp PRIVATE gecode_solver_core)
target_compile_options(gecode_cpp PRIVATE -O3 -march=native)

# ── Runner de tests SSP ───────────────────────────────────────────────────────
add_executable(test_ssp test_ssp.cpp)
target_link_libraries(test_ssp PRIVATE gecode_solver_core)
target_compile_options(test_ssp PRIVATE -O2)

# ── Copie du .so dans la racine du projet ─────────────────────────────────────
add_custom_command(TARGET gecode_cpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:gecode_cpp>
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/$<TARGET_FILE_NAME:gecode_cpp>
    COMMENT "Copie gecode_cpp dans lib/"
)
