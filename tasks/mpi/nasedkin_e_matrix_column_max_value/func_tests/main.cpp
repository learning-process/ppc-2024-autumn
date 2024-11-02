#include <gtest/gtest.h>
#include "ops_mpi.hpp"
#include <mpi.h>

namespace nasedkin_e_matrix_column_max_value_mpi {

TEST(nasedkin_e_matrix_column_max_value_mpi, test_column_max) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<int> matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int rows = 2, cols = 5;
    std::vector<int> result = find_max_by_columns(matrix, rows, cols, rank, 1);

    if (rank == 0) {
        std::vector<int> expected = {6, 7, 8, 9, 10};
        EXPECT_EQ(result, expected);
    }
}

}

