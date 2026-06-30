/*
 * test_ssp.cpp — Lance run_ssp_tests() et retourne 0 si tout passe.
 *
 * Usage : ./test_ssp
 */
#include "gecode_solver.hpp"

int main() {
    return run_ssp_tests() ? 0 : 1;
}
