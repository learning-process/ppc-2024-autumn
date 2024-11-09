// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "seq/naumov_b_min_colum_matrix/include/ops_seq.hpp"

TEST(naumov_b_min_colum_matrix_seq, Test_Min_Column_Values) {
  std::vector<int> input_data = {3, 5, 1, 4, 2, 6, 7, 8, 0};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count = {3, 3};
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));

  int *output_data = new int[3];
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data));
  taskDataSeq->outputs_count.emplace_back(3);

  naumov_b_min_colum_matrix_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  ASSERT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();

  EXPECT_EQ(output_data[0], 3);
  EXPECT_EQ(output_data[1], 2);
  EXPECT_EQ(output_data[2], 0);

  delete[] output_data;
}

TEST(naumov_b_min_colum_matrix_seq, Test_Equal_Elements) {
  std::vector<int> input_data = {5, 5, 5, 5, 5, 5, 5, 5, 5};

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count = {3, 3};
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));

  int *output_data = new int[3];
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data));
  taskDataSeq->outputs_count.emplace_back(3);

  naumov_b_min_colum_matrix_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  ASSERT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();

  EXPECT_EQ(output_data[0], 5);
  EXPECT_EQ(output_data[1], 5);
  EXPECT_EQ(output_data[2], 5);

  delete[] output_data;
}

TEST(naumov_b_min_colum_matrix_seq, Test_Random_Matrix_5_5) {
  const int rows = 5;
  const int cols = 5;
  std::vector<int> input_data(rows * cols);

  std::generate(input_data.begin(), input_data.end(), []() { return rand() % 100; });

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count = {rows, cols};
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));

  int *output_data = new int[cols];
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data));
  taskDataSeq->outputs_count.emplace_back(cols);

  naumov_b_min_colum_matrix_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  ASSERT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();

  for (int j = 0; j < cols; ++j) {
    int expected_min = std::numeric_limits<int>::max();
    for (int i = 0; i < rows; ++i) {
      expected_min = std::min(expected_min, input_data[i * cols + j]);
    }
    EXPECT_EQ(output_data[j], expected_min);
  }

  delete[] output_data;
}

TEST(naumov_b_min_colum_matrix_seq, Test_Random_Matrix_5_10) {
  const int rows = 5;
  const int cols = 10;
  std::vector<int> input_data(rows * cols);

  std::generate(input_data.begin(), input_data.end(), []() { return rand() % 100; });

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count = {rows, cols};
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));

  int *output_data = new int[cols];
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data));
  taskDataSeq->outputs_count.emplace_back(cols);

  naumov_b_min_colum_matrix_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  ASSERT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();

  for (int j = 0; j < cols; ++j) {
    int expected_min = std::numeric_limits<int>::max();
    for (int i = 0; i < rows; ++i) {
      expected_min = std::min(expected_min, input_data[i * cols + j]);
    }
    EXPECT_EQ(output_data[j], expected_min);
  }

  delete[] output_data;
}

TEST(naumov_b_min_colum_matrix_seq, Test_Random_Matrix_15_10) {
  const int rows = 15;
  const int cols = 10;
  std::vector<int> input_data(rows * cols);

  std::generate(input_data.begin(), input_data.end(), []() { return rand() % 100; });

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count = {rows, cols};
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));

  int *output_data = new int[cols];
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data));
  taskDataSeq->outputs_count.emplace_back(cols);

  naumov_b_min_colum_matrix_seq::TestTaskSequential TestTaskSequential(taskDataSeq);
  ASSERT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();

  for (int j = 0; j < cols; ++j) {
    int expected_min = std::numeric_limits<int>::max();
    for (int i = 0; i < rows; ++i) {
      expected_min = std::min(expected_min, input_data[i * cols + j]);
    }
    EXPECT_EQ(output_data[j], expected_min);
  }

  delete[] output_data;
}
