cmake_minimum_required(VERSION 3.15)
project(ccsp_solver CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Gecode (Homebrew Apple Silicon) ──────────────────────────
set(GECODE_PREFIX "/opt/homebrew/opt/gecode")
include_directories(${GECODE_PREFIX}/include)
link_directories(${GECODE_PREFIX}/lib)

# ── Exécutable de test étape 1 ────────────────────────────────
add_executable(test_reader
    main.cpp
    instance_reader.cpp
)

# Gecode sera ajouté à l'étape 2 (ScenarioSubproblem)
# target_link_libraries(test_reader gecodeint gecodesearch ...)
