#pragma once

#include <cinttypes>
#include <cmath>
#include <ctime>
#include <limits>
#include <random>
#include <vector>

namespace malyshev_a_simple_iteration_method_mpi {

double determinant(const std::vector<std::vector<double>>& matrix);
int rank(const std::vector<std::vector<double>>& matrix);

}  // namespace malyshev_a_simple_iteration_method_mpi