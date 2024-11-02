#pragma once
#include <vector>
#include <mpi.h>

namespace nasedkin_e_matrix_column_max_value_mpi {

    std::vector<int> find_max_by_columns(const std::vector<int>& matrix, int rows, int cols, int rank, int size);
    void distribute_columns(int cols, int size, int rank, int& start_col, int& end_col);
    std::vector<int> gather_global_max(const std::vector<int>& local_max, int cols, int rank, int size);

}
