#include <gtest/gtest.h>
#include "ops_seq.hpp"

using namespace nasedkin_e_matrix_column_max_value_seq;

TEST(nasedkin_e_matrix_column_max_value_seq, find_max_in_each_column) {
    std::vector<std::vector<int>> matrix = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
    };
    std::vector<int> expected = {7, 8, 9};
    EXPECT_EQ(FindColumnMaxSequential(matrix), expected);
}

TEST(nasedkin_e_matrix_column_max_value_seq, handle_empty_matrix) {
    std::vector<std::vector<int>> matrix = {};
    std::vector<int> expected = {};
    EXPECT_EQ(FindColumnMaxSequential(matrix), expected);
}

TEST(nasedkin_e_matrix_column_max_value_seq, handle_single_column) {
    std::vector<std::vector<int>> matrix = {
        {3},
        {1},
        {4},
    };
    std::vector<int> expected = {4};
    EXPECT_EQ(FindColumnMaxSequential(matrix), expected);
}
