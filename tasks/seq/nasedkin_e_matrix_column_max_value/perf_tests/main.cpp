#include <gtest/gtest.h>
#include <vector>
#include "ops_seq.hpp"
#include <chrono>
#include <iostream>

namespace nasedkin_e_matrix_column_max_value {

    TEST(nasedkin_e_matrix_column_max_value_seq, test_task_run) {
        const int numRows = 10000;
        const int numCols = 1000;
        std::vector<std::vector<int>> largeMatrix(numRows, std::vector<int>(numCols, 1));

        auto start = std::chrono::high_resolution_clock::now();
        auto result = getMaxValuesPerColumn(largeMatrix);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> duration = end - start;
        std::cout << "Execution time for large matrix: " << duration.count() << " seconds" << std::endl;

        EXPECT_EQ(result.size(), numCols);
    }

    TEST(nasedkin_e_matrix_column_max_value_seq, test_pipeline_run) {
        const int numRows = 5000;
        const int numCols = 500;
        std::vector<std::vector<int>> matrix(numRows, std::vector<int>(numCols, 2));

        auto start = std::chrono::high_resolution_clock::now();
        auto result = getMaxValuesPerColumn(matrix);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> duration = end - start;
        std::cout << "Execution time in pipeline context: " << duration.count() << " seconds" << std::endl;

        EXPECT_EQ(result.size(), numCols);
    }

}
