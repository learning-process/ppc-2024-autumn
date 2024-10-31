// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/borisov_sum_of_rows/include/ops_seq.hpp"

TEST(borisov_sum_of_rows, Test_Sum_Matrix_10) {
  const int rows = 10;
  const int cols = 10;

  // Create data
  std::vector<int> matrix(rows * cols, 1);
  std::vector<int> row_sums(rows, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(row_sums.data()));
  taskDataSeq->outputs_count.emplace_back(row_sums.size());

  // Create Task
  borisov_sum_of_rows::SumOfRowsTaskSequential sumOfRowsTask(taskDataSeq);
  ASSERT_TRUE(sumOfRowsTask.validation());

  sumOfRowsTask.pre_processing();
  sumOfRowsTask.run();
  sumOfRowsTask.post_processing();

  for (size_t i = 0; i < rows; i++) {
    ASSERT_EQ(row_sums[i], cols);
  }
}

TEST(borisov_sum_of_rows, Test_Sum_Matrix_30) {
  const int rows = 30;
  const int cols = 30;

  // Create data
  std::vector<int> matrix(rows * cols, 1);
  std::vector<int> row_sums(rows, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(row_sums.data()));
  taskDataSeq->outputs_count.emplace_back(row_sums.size());

  // Create Task
  borisov_sum_of_rows::SumOfRowsTaskSequential sumOfRowsTask(taskDataSeq);
  ASSERT_TRUE(sumOfRowsTask.validation());

  sumOfRowsTask.pre_processing();
  sumOfRowsTask.run();
  sumOfRowsTask.post_processing();

  for (size_t i = 0; i < rows; i++) {
    ASSERT_EQ(row_sums[i], cols);
  }
}

TEST(borisov_sum_of_rows, Test_Sum_Matrix_100) {
  const int rows = 100;
  const int cols = 100;

  // Create data
  std::vector<int> matrix(rows * cols, 1);
  std::vector<int> row_sums(rows, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(row_sums.data()));
  taskDataSeq->outputs_count.emplace_back(row_sums.size());

  // Create Task
  borisov_sum_of_rows::SumOfRowsTaskSequential sumOfRowsTask(taskDataSeq);
  ASSERT_TRUE(sumOfRowsTask.validation());

  sumOfRowsTask.pre_processing();
  sumOfRowsTask.run();
  sumOfRowsTask.post_processing();

  for (size_t i = 0; i < rows; i++) {
    ASSERT_EQ(row_sums[i], cols);
  }
}

TEST(borisov_sum_of_rows, EmptyMatrix) {
  const int rows = 0;
  const int cols = 0;

  std::vector<int> matrix;
  std::vector<int> row_sums;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.push_back(rows);
  taskDataSeq->inputs_count.push_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(row_sums.data()));
  taskDataSeq->outputs_count.push_back(row_sums.size());

  borisov_sum_of_rows::SumOfRowsTaskSequential sumOfRowsTask(taskDataSeq);
  ASSERT_EQ(sumOfRowsTask.validation(), false);

  sumOfRowsTask.pre_processing();
  sumOfRowsTask.run();
  sumOfRowsTask.post_processing();

  ASSERT_TRUE(row_sums.empty());
}
