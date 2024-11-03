#include <vector>
#include <limits>
#include <mpi.h>

namespace nasedkin_e_matrix_column_max_value_mpi {

std::vector<double> findMaxInColumns(const std::vector<std::vector<double>>& matrix) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t rows = matrix.size();
    if (rows == 0) return {};

    size_t cols = matrix[0].size();
    std::vector<double> localMax(cols, std::numeric_limits<double>::lowest());

    // Определяем диапазон строк для текущего процесса
    size_t localRows = rows / size;
    size_t startRow = rank * localRows;
    size_t endRow = (rank == size - 1) ? rows : startRow + localRows;

    // Находим максимальные значения в своих строках
    for (size_t i = startRow; i < endRow; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            localMax[j] = std::max(localMax[j], matrix[i][j]);
        }
    }

    std::vector<double> globalMax(cols);
    MPI_Reduce(localMax.data(), globalMax.data(), cols, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    return globalMax;
}

}  // namespace nasedkin_e_matrix_column_max_value_mpi
