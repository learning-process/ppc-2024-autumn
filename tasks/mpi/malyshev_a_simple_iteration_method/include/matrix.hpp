#pragma once

#include <cinttypes>
#include <cmath>
#include <ctime>
#include <limits>
#include <random>
#include <vector>

namespace malyshev_a_simple_iteration_method_mpi {

double determinant(const std::vector<double>& matrix, uint32_t n);
int rank(const std::vector<double>& matrix, uint32_t n, uint32_t m);

}  // namespace malyshev_a_simple_iteration_method_mpi