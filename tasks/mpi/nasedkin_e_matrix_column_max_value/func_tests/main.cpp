#include <gtest/gtest.h>
#include "ops_mpi.hpp"

namespace nasedkin_e_matrix_column_max_value_mpi {

    TEST(nasedkin_e_matrix_column_max_value_mpi, test_findMaxInColumns) {
        std::vector<std::vector<double>> matrix = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0}
        };

        std::vector<double> expected = {7.0, 8.0, 9.0};
        auto result = findMaxInColumns(matrix);

        EXPECT_EQ(result, expected);
    }

}  // namespace nasedkin_e_matrix_column_max_value_mpi
