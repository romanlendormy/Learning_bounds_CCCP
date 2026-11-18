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
