cmake_minimum_required(VERSION 3.14)
project(v1ld_cpp2 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Python (forcer conda 3.11) ────────────────────────────────────────────────
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

# ── Bibliothèque solver (partagée entre module Python et binaire de test) ─────
add_library(v1ld_core2 STATIC v1ld_core.cpp)
target_include_directories(v1ld_core2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(v1ld_core2 PRIVATE -O3 -march=native)

# ── Module Python ─────────────────────────────────────────────────────────────
pybind11_add_module(v1ld_cpp2 v1ld_bindings.cpp)
target_link_libraries(v1ld_cpp2 PRIVATE v1ld_core2)
target_compile_options(v1ld_cpp2 PRIVATE -O3 -march=native)

# Copie le .so dans le dossier parent (racine du projet) après la build
add_custom_command(TARGET v1ld_cpp2 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:v1ld_cpp2>
        ${CMAKE_CURRENT_SOURCE_DIR}/../$<TARGET_FILE_NAME:v1ld_cpp2>
    COMMENT "Copie du module v1ld_cpp2 dans la racine du projet"
)

# ── Binaire de test standalone ────────────────────────────────────────────────
add_executable(v1ld_test2 main.cpp)
target_link_libraries(v1ld_test2 PRIVATE v1ld_core2)
target_compile_options(v1ld_test2 PRIVATE -O3 -march=native)

# ── Bibliothèque antisymétrique ───────────────────────────────────────────────
add_library(v1ld_core_antisym STATIC v1ld_core_antisym.cpp)
target_include_directories(v1ld_core_antisym PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(v1ld_core_antisym PRIVATE -O3 -march=native)

# ── Module Python antisymétrique ──────────────────────────────────────────────
pybind11_add_module(v1ld_cpp2_antisym v1ld_bindings_antisym.cpp)
target_link_libraries(v1ld_cpp2_antisym PRIVATE v1ld_core_antisym)
target_compile_options(v1ld_cpp2_antisym PRIVATE -O3 -march=native)

add_custom_command(TARGET v1ld_cpp2_antisym POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:v1ld_cpp2_antisym>
        ${CMAKE_CURRENT_SOURCE_DIR}/../$<TARGET_FILE_NAME:v1ld_cpp2_antisym>
    COMMENT "Copie du module v1ld_cpp2_antisym dans la racine du projet"
)
