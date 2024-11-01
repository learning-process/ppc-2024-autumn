// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/zinoviev_a_sum_cols_matrix/include/ops_seq.hpp"

TEST(zinoviev_a_sum_cols_matrix, TrialEmptyMatrix) {
  int cols = 0;
  int rows = 5;

  std::vector<int> matrix = {};
  std::vector<int> expres;
  std::vector<int> ans(cols, 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres.data()));
  taskDataSeq->outputs_count.emplace_back(expres.size());

  zinoviev_a_sum_cols_matrix::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expres, ans);
}

TEST(zinoviev_a_sum_cols_matrix, Test1x1Matrix) {
  int cols = 1;
  int rows = 1;

  std::vector<int> matrix = {5};
  std::vector<int> expres(cols, 0);
  std::vector<int> ans = {5};

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres.data()));
  taskDataSeq->outputs_count.emplace_back(expres.size());

  zinoviev_a_sum_cols_matrix::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expres, ans);
}

TEST(zinoviev_a_sum_cols_matrix, Test3x3Matrix) {
  int cols = 3;
  int rows = 3;

  std::vector<int> matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> expres(cols, 0);
  std::vector<int> ans = {12, 15, 18};  // Ñףללû ןמ סעמכבצאל

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres.data()));
  taskDataSeq->outputs_count.emplace_back(expres.size());

  zinoviev_a_sum_cols_matrix::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expres, ans);
}

TEST(zinoviev_a_sum_cols_matrix, Test5x5IdentityMatrix) {
  int cols = 5;
  int rows = 5;

  std::vector<int> matrix = {1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1};
  std::vector<int> expres(cols, 0);
  std::vector<int> ans = {1, 1, 1, 1, 1};  // Ñףללû ןמ סעמכבצאל

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres.data()));
  taskDataSeq->outputs_count.emplace_back(expres.size());

  zinoviev_a_sum_cols_matrix::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expres, ans);
}

TEST(zinoviev_a_sum_cols_matrix, Test2x100Matrix) {
  int cols = 2;
  int rows = 100;

  std::vector<int> matrix(200, 1);  // גסו והטםטצû
  std::vector<int> expres(cols, 0);
  std::vector<int> ans = {100, 100};  // Ñףללû ןמ סעמכבצאל

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres.data()));
  taskDataSeq->outputs_count.emplace_back(expres.size());

  zinoviev_a_sum_cols_matrix::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expres, ans);
}

TEST(zinoviev_a_sum_cols_matrix, TestNegativeValuesMatrix) {
  int cols = 4;
  int rows = 2;

  std::vector<int> matrix = {-1, -2, -3, -4, -5, -6, -7, -8};
  std::vector<int> expres(cols, 0);
  std::vector<int> ans = {-6, -8, -10, -12};  // Ñףללû ןמ סעמכבצאל

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres.data()));
  taskDataSeq->outputs_count.emplace_back(expres.size());

  zinoviev_a_sum_cols_matrix::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expres, ans);
}

TEST(zinoviev_a_sum_cols_matrix, TestLargeValuesMatrix) {
  int cols = 3;
  int rows = 2;

  std::vector<int> matrix = {1000000, 2000000, 3000000, 4000000, 5000000, 6000000};
  std::vector<int> expres(cols, 0);
  std::vector<int> ans = {5000000, 7000000, 9000000};  // Ñףללû ןמ סעמכבצאל

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres.data()));
  taskDataSeq->outputs_count.emplace_back(expres.size());

  zinoviev_a_sum_cols_matrix::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expres, ans);
}

TEST(zinoviev_a_sum_cols_matrix, TestRectangularMatrix) {
  int cols = 3;
  int rows = 4;

  std::vector<int> matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int> expres(cols, 0);
  std::vector<int> ans = {22, 26, 30};  // Ñףללû ןמ סעמכבצאל

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres.data()));
  taskDataSeq->outputs_count.emplace_back(expres.size());

  zinoviev_a_sum_cols_matrix::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expres, ans);
}
