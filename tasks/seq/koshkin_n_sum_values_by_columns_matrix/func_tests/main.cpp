// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/koshkin_n_sum_values_by_columns_matrix/include/ops_seq.hpp"

TEST(koshkin_n_sum_values_by_columns_matrix_seq, Test_sum_values_by_columns_SquareMatrixSmall) {
	int rows = 2;
    int columns = 2;

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    koshkin_n_sum_values_by_columns_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);

    std::vector<int> matrix = {1, 2, 3, 4};
    std::vector<int> res_out = {0, 0};  // Sum column
    std::vector<int> exp_res = {4, 6};

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data())); 
    //After post processing it should look like this 
    //input_ = {
    // {1, 2},
    // {3, 4}
    //};
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(columns);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out.data()));
    // res[0] (sum of the first column) = 1 + 3 = 4
    // res[1] (sum of the second column) = 2 + 4 = 6

    taskDataSeq->outputs_count.emplace_back(res_out.size());

    ASSERT_EQ(testTaskSequential.validation(), true);
    ASSERT_TRUE(testTaskSequential.pre_processing());
    ASSERT_TRUE(testTaskSequential.run());
    ASSERT_TRUE(testTaskSequential.post_processing());
    ASSERT_EQ(res_out, exp_res);
}

TEST(koshkin_n_sum_values_by_columns_matrix_seq, Test_sum_values_by_columns_SquareMatrixLarge) {
    const int rows = 1000;
    const int columns = 1000;

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    koshkin_n_sum_values_by_columns_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);

    std::vector<int> matrix(rows * columns);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < columns; ++j) {
        matrix[i * columns + j] = i + j;  
      }
    }
    std::vector<int> res_out(columns, 0);  
    std::vector<int> exp_res(columns, 0);
    for (int j = 0; j < columns; ++j) {
      for (int i = 0; i < rows; ++i) {
        exp_res[j] += (i + j);  
      }
    }

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(columns);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out.data()));
    taskDataSeq->outputs_count.emplace_back(res_out.size());

    ASSERT_EQ(testTaskSequential.validation(), true);
    ASSERT_TRUE(testTaskSequential.pre_processing());
    ASSERT_TRUE(testTaskSequential.run());
    ASSERT_TRUE(testTaskSequential.post_processing());
    ASSERT_EQ(res_out, exp_res);
}

TEST(koshkin_n_sum_values_by_columns_matrix_seq, Test_sum_values_by_columns_MatrixSmall4x10) {
    const int rows = 4;
    const int columns = 10;

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    koshkin_n_sum_values_by_columns_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);

    std::vector<int> matrix = {
        1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  // 1 row
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,  // 2
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,  // 3
        31, 32, 33, 34, 35, 36, 37, 38, 39, 40   // 4
    };
    std::vector<int> res_out(columns, 0);  
    std::vector<int> exp_res(columns, 0);
    for (int j = 0; j < columns; ++j) {
      exp_res[j] = matrix[j] + matrix[j + columns] + matrix[j + 2 * columns] + matrix[j + 3 * columns];
    }

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(columns);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out.data()));
    taskDataSeq->outputs_count.emplace_back(res_out.size());

    ASSERT_EQ(testTaskSequential.validation(), true);
    ASSERT_TRUE(testTaskSequential.pre_processing());
    ASSERT_TRUE(testTaskSequential.run());
    ASSERT_TRUE(testTaskSequential.post_processing());
    ASSERT_EQ(res_out, exp_res);
}









