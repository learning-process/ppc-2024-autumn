#pragma once

#include <vector>
#include <boost/mpi.hpp>

namespace nasedkin_e_matrix_column_max_value_mpi {

std::vector<int> FindColumnMaxMPI(const std::vector<std::vector<int>>& matrix);

}
