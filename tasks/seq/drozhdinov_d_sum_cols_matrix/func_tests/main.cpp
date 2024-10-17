// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/drozhdinov_d_sum_cols_matrix/include/ops_seq.hpp"


TEST(drozhdinov_d_sum_cols_matrix_seq, SquareMatrixTests1) {
  int cols = 2;
  int rows = 2;

  // Create data
  std::vector<int> matrix = {1, 0, 2, 1};
  std::vector<int> expres(cols, 0);
  std::vector<int> ans = {3, 1};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  // taskDataSeq->inputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres.data()));
  taskDataSeq->outputs_count.emplace_back(expres.size());

  // Create Task
  drozhdinov_d_sum_cols_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expres, ans);
}

TEST(drozhdinov_d_sum_cols_matrix_seq, SquareMatrixTests2) {
  int cols = 2000;
  int rows = 2000;

  // Create data
  std::vector<int> matrix(cols * rows, 0);
  matrix[1] = 1;
  std::vector<int> expres(cols, 0);
  std::vector<int> ans(cols, 0);
  ans[1] = 1;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  // taskDataSeq->inputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres.data()));
  taskDataSeq->outputs_count.emplace_back(expres.size());

  // Create Task
  drozhdinov_d_sum_cols_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expres, ans);
}

TEST(drozhdinov_d_sum_cols_matrix_seq, SquareMatrixTests3) {
  int cols = 10000;
  int rows = 10000;

  // Create data
  std::vector<int> matrix(cols * rows, 0);
  matrix[1] = 1;
  std::vector<int> expres(cols, 0);
  std::vector<int> ans(cols, 0);
  ans[1] = 1;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  // taskDataSeq->inputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres.data()));
  taskDataSeq->outputs_count.emplace_back(expres.size());

  // Create Task
  drozhdinov_d_sum_cols_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expres, ans);
}

TEST(drozhdinov_d_sum_cols_matrix_seq, RectangleMatrixTests1) {
  int cols = 4;
  int rows = 1;

  // Create data
  std::vector<int> matrix = {1, 0, 2, 1};
  std::vector<int> expres(cols, 0);
  std::vector<int> ans = {1, 0, 2, 1};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  // taskDataSeq->inputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres.data()));
  taskDataSeq->outputs_count.emplace_back(expres.size());

  // Create Task
  drozhdinov_d_sum_cols_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expres, ans);
}

TEST(drozhdinov_d_sum_cols_matrix_seq, RectangleMatrixTests2) {
  int cols = 1;
  int rows = 100;

  // Create data
  std::vector<int> matrix(cols * rows, 0);
  matrix[1] = 1;
  std::vector<int> expres(cols, 0);
  std::vector<int> ans(cols, 0);
  ans[0] = 1;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  // taskDataSeq->inputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres.data()));
  taskDataSeq->outputs_count.emplace_back(expres.size());

  // Create Task
  drozhdinov_d_sum_cols_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expres, ans);
}

TEST(drozhdinov_d_sum_cols_matrix_seq, RectangleMatrixTests3) {
  int cols = 2000;
  int rows = 1000;

  // Create data
  std::vector<int> matrix(cols * rows, 0);
  matrix[1] = 1;
  std::vector<int> expres(cols, 0);
  std::vector<int> ans(cols, 0);
  ans[1] = 1;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  // taskDataSeq->inputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres.data()));
  taskDataSeq->outputs_count.emplace_back(expres.size());

  // Create Task
  drozhdinov_d_sum_cols_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expres, ans);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
