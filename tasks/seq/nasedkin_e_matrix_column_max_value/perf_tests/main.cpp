#include <gtest/gtest.h>
#include "ops_seq.hpp"

using namespace nasedkin_e_matrix_column_max_value_seq;

TEST(nasedkin_e_matrix_column_max_value_seq, test_pipeline_run) {
    std::vector<std::vector<int>> matrix(1000, std::vector<int>(1000, 1));
    auto result = FindColumnMaxSequential(matrix);
    ASSERT_EQ(result.size(), 1000);
}

TEST(nasedkin_e_matrix_column_max_value_seq, test_task_run) {
    std::vector<std::vector<int>> matrix(1000, std::vector<int>(1000, 1));
    auto result = FindColumnMaxSequential(matrix);
    ASSERT_EQ(result.size(), 1000);
}
