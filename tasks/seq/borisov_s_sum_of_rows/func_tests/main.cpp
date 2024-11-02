// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <climits>
#include <random>
#include <vector>

#include "seq/borisov_s_sum_of_rows/include/ops_seq.hpp"

TEST(borisov_s_sum_of_rows, Test_Sum_Matrix_10) {
  size_t rows = 10;
  size_t cols = 10;

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
  borisov_s_sum_of_rows::SumOfRowsTaskSequential sumOfRowsTask(taskDataSeq);
  ASSERT_TRUE(sumOfRowsTask.validation());

  sumOfRowsTask.pre_processing();
  sumOfRowsTask.run();
  sumOfRowsTask.post_processing();

  for (size_t i = 0; i < rows; i++) {
    ASSERT_EQ(row_sums[i], 10);
  }
}

TEST(borisov_s_sum_of_rows, Test_Sum_Matrix_30) {
  size_t rows = 30;
  size_t cols = 30;

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
  borisov_s_sum_of_rows::SumOfRowsTaskSequential sumOfRowsTask(taskDataSeq);
  ASSERT_TRUE(sumOfRowsTask.validation());

  sumOfRowsTask.pre_processing();
  sumOfRowsTask.run();
  sumOfRowsTask.post_processing();

  for (size_t i = 0; i < rows; i++) {
    ASSERT_EQ(row_sums[i], 30);
  }
}

TEST(borisov_s_sum_of_rows, Test_Sum_Matrix_100) {
  size_t rows = 100;
  size_t cols = 100;

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
  borisov_s_sum_of_rows::SumOfRowsTaskSequential sumOfRowsTask(taskDataSeq);
  ASSERT_TRUE(sumOfRowsTask.validation());

  sumOfRowsTask.pre_processing();
  sumOfRowsTask.run();
  sumOfRowsTask.post_processing();

  for (size_t i = 0; i < rows; i++) {
    ASSERT_EQ(row_sums[i], 100);
  }
}

TEST(borisov_s_sum_of_rows, EmptyMatrix) {
  size_t rows = 0;
  size_t cols = 0;

  std::vector<int> matrix;
  std::vector<int> row_sums;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.push_back(rows);
  taskDataSeq->inputs_count.push_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(row_sums.data()));
  taskDataSeq->outputs_count.push_back(row_sums.size());

  borisov_s_sum_of_rows::SumOfRowsTaskSequential sumOfRowsTask(taskDataSeq);
  ASSERT_EQ(sumOfRowsTask.validation(), false);

  sumOfRowsTask.pre_processing();
  sumOfRowsTask.run();
  sumOfRowsTask.post_processing();

  ASSERT_TRUE(row_sums.empty());
}

TEST(borisov_s_sum_of_rows, Test_Negative_Numbers) {
  size_t rows = 5;
  size_t cols = 5;

  std::vector<int> matrix(rows * cols);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(-100, 100);
  for (auto &elem : matrix) {
    elem = dist(gen);
  }
  std::vector<int> row_sums(rows, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(row_sums.data()));
  taskDataSeq->outputs_count.emplace_back(row_sums.size());

  borisov_s_sum_of_rows::SumOfRowsTaskSequential sumOfRowsTask(taskDataSeq);
  ASSERT_TRUE(sumOfRowsTask.validation());

  sumOfRowsTask.pre_processing();
  sumOfRowsTask.run();
  sumOfRowsTask.post_processing();

  for (size_t i = 0; i < rows; i++) {
    int expected_sum = 0;
    for (size_t j = 0; j < cols; j++) {
      expected_sum += matrix[(i * cols) + j];
    }
    ASSERT_EQ(row_sums[i], expected_sum);
  }
}

TEST(borisov_s_sum_of_rows, Test_NonDivisibleDimensions) {
  size_t rows = 7;
  size_t cols = 3;

  std::vector<int> matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
  std::vector<int> row_sums(rows, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(row_sums.data()));
  taskDataSeq->outputs_count.emplace_back(row_sums.size());

  borisov_s_sum_of_rows::SumOfRowsTaskSequential sumOfRowsTask(taskDataSeq);
  ASSERT_TRUE(sumOfRowsTask.validation());

  sumOfRowsTask.pre_processing();
  sumOfRowsTask.run();
  sumOfRowsTask.post_processing();

  for (size_t i = 0; i < rows; i++) {
    int expected_sum = 0;
    for (size_t j = 0; j < cols; j++) {
      expected_sum += matrix[(i * cols) + j];
    }
    ASSERT_EQ(row_sums[i], expected_sum);
  }
}

TEST(borisov_s_sum_of_rows, Test_Max_Min_Int) {
  size_t rows = 2;
  size_t cols = 2;

  std::vector<int> matrix = {INT_MAX, INT_MIN, INT_MAX, INT_MIN};
  std::vector<int> row_sums(rows, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(row_sums.data()));
  taskDataSeq->outputs_count.emplace_back(row_sums.size());

  borisov_s_sum_of_rows::SumOfRowsTaskSequential sumOfRowsTask(taskDataSeq);
  ASSERT_TRUE(sumOfRowsTask.validation());

  sumOfRowsTask.pre_processing();
  sumOfRowsTask.run();
  sumOfRowsTask.post_processing();

  for (size_t i = 0; i < rows; i++) {
    int expected_sum = INT_MAX + INT_MIN;
    ASSERT_EQ(row_sums[i], expected_sum);
  }
}

TEST(borisov_s_sum_of_rows, Test_Single_Row_Matrix) {
  size_t rows = 1;
  size_t cols = 10;

  std::vector<int> matrix(cols, 1);
  std::vector<int> row_sums(rows, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(row_sums.data()));
  taskDataSeq->outputs_count.emplace_back(row_sums.size());

  borisov_s_sum_of_rows::SumOfRowsTaskSequential sumOfRowsTask(taskDataSeq);
  ASSERT_TRUE(sumOfRowsTask.validation());

  sumOfRowsTask.pre_processing();
  sumOfRowsTask.run();
  sumOfRowsTask.post_processing();

  ASSERT_EQ(row_sums[0], 10);
}

TEST(borisov_s_sum_of_rows, Test_Single_Column_Matrix) {
  size_t rows = 10;
  size_t cols = 1;

  std::vector<int> matrix(rows, 1);
  std::vector<int> row_sums(rows, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(row_sums.data()));
  taskDataSeq->outputs_count.emplace_back(row_sums.size());

  borisov_s_sum_of_rows::SumOfRowsTaskSequential sumOfRowsTask(taskDataSeq);
  ASSERT_TRUE(sumOfRowsTask.validation());

  sumOfRowsTask.pre_processing();
  sumOfRowsTask.run();
  sumOfRowsTask.post_processing();

  for (size_t i = 0; i < rows; i++) {
    ASSERT_EQ(row_sums[i], 1);
  }
}

TEST(borisov_s_sum_of_rows, Test_Null_Pointers) {
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  size_t rows = 10;
  size_t cols = 10;

  taskDataPar->inputs.emplace_back(nullptr);
  taskDataPar->outputs.emplace_back(nullptr);
  taskDataPar->inputs_count.push_back(rows);
  taskDataPar->inputs_count.push_back(cols);
  taskDataPar->outputs_count.push_back(rows);

  borisov_s_sum_of_rows::SumOfRowsTaskSequential sumOfRowsTaskSequential(taskDataPar);
  ASSERT_FALSE(sumOfRowsTaskSequential.validation());
}

TEST(borisov_s_sum_of_rows, Test_Validation_Invalid_Output_Count_Sequential) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  size_t rows = 10;
  size_t cols = 10;

  std::vector<int> matrix(rows * cols, 1);
  std::vector<int> row_sums(rows - 1, 0);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.push_back(rows);
  taskDataSeq->inputs_count.push_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(row_sums.data()));
  taskDataSeq->outputs_count.push_back(row_sums.size());

  borisov_s_sum_of_rows::SumOfRowsTaskSequential sumOfRowsTask(taskDataSeq);

  ASSERT_FALSE(sumOfRowsTask.validation());
}

TEST(borisov_s_sum_of_rows, Test_Validation_Null_Inputs_Outputs_Sequential) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  size_t rows = 10;
  size_t cols = 10;

  taskDataSeq->inputs.emplace_back(nullptr);
  taskDataSeq->outputs.emplace_back(nullptr);
  taskDataSeq->inputs_count.push_back(rows);
  taskDataSeq->inputs_count.push_back(cols);
  taskDataSeq->outputs_count.push_back(rows);

  borisov_s_sum_of_rows::SumOfRowsTaskSequential sumOfRowsTask(taskDataSeq);

  ASSERT_FALSE(sumOfRowsTask.validation());
}
