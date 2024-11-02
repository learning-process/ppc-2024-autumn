#include <gtest/gtest.h>
#include "ops_seq.hpp"

namespace nasedkin_e_matrix_column_max_value {

    TEST(nasedkin_e_matrix_column_max_value_seq, test_case_1) {
        std::vector<std::vector<int>> matrix = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        };
        std::vector<int> expected = {7, 8, 9};
        EXPECT_EQ(getMaxValuesPerColumn(matrix), expected);
    }

    TEST(nasedkin_e_matrix_column_max_value_seq, test_case_2) {
        std::vector<std::vector<int>> matrix = {
            {10, 20},
            {15, 25},
            {5, 35}
        };
        std::vector<int> expected = {15, 35};
        EXPECT_EQ(getMaxValuesPerColumn(matrix), expected);
    }

}
