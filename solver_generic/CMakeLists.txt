# sld_generic_core : algorithmes de sous-gradient SLD / DLD (génériques) et
# sous-problèmes MKP (B&B exact, surrogate, DP). Inclus depuis le CMakeLists racine.

add_library(sld_generic_core STATIC
    sld_solver.cpp
    dld_solver.cpp
    mkp/mkp_problem.cpp
    mkp/mkp_problem_bnb.cpp
    mkp/mkp_problem_bnb_surrogate.cpp
    mkp/mkp_dld_problem.cpp
    mkp/mkp_dld_problem_dp.cpp
)
target_include_directories(sld_generic_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/mkp
)
target_compile_options(sld_generic_core PRIVATE -O3 -march=native)
target_link_libraries(sld_generic_core PUBLIC ortools::ortools OpenMP::OpenMP_CXX)

# ── CCPO ──────────────────────────────────────────────────────────────────────
# ccpo_bound : borne lagrangienne CCPO (sous-problèmes θ / ζ, presolve exact),
# en en-têtes (ccpo_sld.hpp, ccpo_presolve_exact.hpp).
add_library(ccpo_bound INTERFACE)
target_include_directories(ccpo_bound INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/ccpo)

# ccpo_scip_zeta : ζ résolu par SCIP pour les grands sous-problèmes (hybride
# B&B / SCIP, activé à l'exécution par CCPO_ZETA_SCIP=1). SCIP 9.0 partagé, celui du
# module scipoptsuite/9.0.0 sur les clusters Alliance (EBROOTSCIPOPTSUITE). Lien direct
# vers libscip.so : la cible CMake « libscip » est déjà définie par OR-Tools, qui embarque
# sa propre copie statique de SCIP.
set(SCIP_PREFIX "$ENV{EBROOTSCIPOPTSUITE}" CACHE PATH "Préfixe d'installation SCIP 9")
find_path(SCIP_INCLUDE_DIR scip/scip.h PATHS "${SCIP_PREFIX}/include" NO_DEFAULT_PATH)
find_library(SCIP_LIBRARY scip PATHS "${SCIP_PREFIX}/lib64" "${SCIP_PREFIX}/lib" NO_DEFAULT_PATH)
if(NOT SCIP_INCLUDE_DIR OR NOT SCIP_LIBRARY)
    message(FATAL_ERROR "SCIP 9 introuvable sous SCIP_PREFIX='${SCIP_PREFIX}' (module load scipoptsuite/9.0.0)")
endif()
get_filename_component(SCIP_LIBRARY_DIR "${SCIP_LIBRARY}" DIRECTORY)
# Dépendances de libscip.so (Ipopt, TBB, BLAS) : trouvées à l'exécution par son propre RPATH,
# mais l'éditeur de liens a besoin de -rpath-link vers les mêmes dossiers.
execute_process(COMMAND readelf -d "${SCIP_LIBRARY}" OUTPUT_VARIABLE _scip_dyn ERROR_QUIET)
string(REGEX MATCH "rpath: \\[([^]]*)\\]" _ "${_scip_dyn}")
string(REPLACE ":" ";" _scip_rpath "${CMAKE_MATCH_1}")
list(FILTER _scip_rpath EXCLUDE REGEX "ORIGIN")
set(SCIP_RPATH_LINK "")
foreach(_d IN LISTS _scip_rpath)
    list(APPEND SCIP_RPATH_LINK "-Wl,-rpath-link,${_d}")
endforeach()

add_library(ccpo_scip_zeta STATIC ccpo/scip_zeta.cpp)
set_target_properties(ccpo_scip_zeta PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(ccpo_scip_zeta PRIVATE ${SCIP_INCLUDE_DIR})
target_compile_options(ccpo_scip_zeta PRIVATE -O3 -march=native)
target_link_libraries(ccpo_scip_zeta PUBLIC ccpo_bound ${SCIP_LIBRARY} -Wl,-rpath,${SCIP_LIBRARY_DIR} ${SCIP_RPATH_LINK} OpenMP::OpenMP_CXX)

# Bibliothèques partagées appelées depuis Python (ctypes, gnn_ccpo/ccpo_libs.py) :
#   ccpo_lagrangian.so : L(λ), sous-gradient, sous-gradient complet (entraînement du GNN CCPO) ;
#   presolve_exact.so  : presolve exact certifié (recalcul des drapeaux de scénarios, tests).
add_library(ccpo_lagrangian SHARED ccpo/ccpo_lagrangian_bind.cpp)
target_link_libraries(ccpo_lagrangian PRIVATE ccpo_bound OpenMP::OpenMP_CXX)
add_library(presolve_exact SHARED ccpo/presolve_bind.cpp)
target_link_libraries(presolve_exact PRIVATE ccpo_bound)
set_target_properties(ccpo_lagrangian presolve_exact PROPERTIES PREFIX "" SUFFIX ".so" CXX_EXTENSIONS OFF)
target_compile_options(ccpo_lagrangian PRIVATE -O3 -march=native)
target_compile_options(presolve_exact PRIVATE -O3 -march=native)
