#include "ops_mpi.hpp"
#include <algorithm>

namespace nasedkin_e_matrix_column_max_value_mpi {

    std::vector<int> find_max_by_columns(const std::vector<int>& matrix, int rows, int cols, int rank, int size) {
        int start_col, end_col;
        distribute_columns(cols, size, rank, start_col, end_col);

        std::vector<int> local_max(end_col - start_col, -1);
        for (int col = start_col; col < end_col; ++col) {
            int max_val = -1;
            for (int row = 0; row < rows; ++row) {
                max_val = std::max(max_val, matrix[row * cols + col]);
            }
            local_max[col - start_col] = max_val;
        }
        return gather_global_max(local_max, cols, rank, size);
    }

    void distribute_columns(int cols, int size, int rank, int& start_col, int& end_col) {
        int cols_per_proc = cols / size;
        int remainder = cols % size;
        start_col = rank * cols_per_proc + std::min(rank, remainder);
        end_col = start_col + cols_per_proc + (rank < remainder ? 1 : 0);
    }

    std::vector<int> gather_global_max(const std::vector<int>& local_max, int cols, int rank, int size) {
        std::vector<int> global_max(cols);
        MPI_Gather(local_max.data(), local_max.size(), MPI_INT, global_max.data(), local_max.size(), MPI_INT, 0, MPI_COMM_WORLD);
        return global_max;
    }

}
