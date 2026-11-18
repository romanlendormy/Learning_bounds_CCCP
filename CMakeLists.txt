cmake_minimum_required(VERSION 3.14)
project(learning_bounds_cccp CXX C)

# Construit tout le C++ du dépôt en une fois :
#   cmake -S . -B build -DGNN_DEVICE=cuda   (ou cpu, défaut)
#   cmake --build build -j
#
# Bibliothèques tierces attendues sous external/ (cf. README) ; chaque préfixe
# est surchargeable : -DGECODE_PREFIX=... -DLIBTORCH_PREFIX=... -DCMAKE_PREFIX_PATH=<ortools>

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXTERNAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external" CACHE PATH "Bibliothèques tierces")
list(APPEND CMAKE_PREFIX_PATH "${EXTERNAL_DIR}/ortools")

find_package(ortools CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

add_subdirectory(solver_generic)   # bibliothèque sld_generic_core (bornes SLD / DLD)
add_subdirectory(solver_bab)       # B&B Gecode (MKP, CCPO)
