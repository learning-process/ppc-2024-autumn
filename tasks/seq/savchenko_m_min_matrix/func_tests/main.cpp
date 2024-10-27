// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <climits>
#include <random>
#include <vector>

#include "seq/savchenko_m_min_matrix/include/ops_seq.hpp"

TEST(savchenko_m_min_matrix_seq, test_min_10x10) {
  std::vector<int> matrix;
  std::vector<int32_t> min_value(1, INT_MAX);

  std::random_device dev;
  std::mt19937 gen(dev());

  // Create data
  const size_t rows = 10;
  const size_t columns = 10;
  const int gen_min = -1000;
  const int gen_max = 1000;
  const int ref = INT_MIN;

  matrix = savchenko_m_min_matrix_seq::getRandomMatrix(rows, columns, gen_min, gen_max);
  const size_t index = gen() % (rows * columns);
  matrix[index] = INT_MIN;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(min_value.data()));
  taskDataSeq->outputs_count.emplace_back(min_value.size());

  // Create Task
  savchenko_m_min_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ref, min_value[0]);
}

TEST(savchenko_m_min_matrix_seq, test_min_100x10) {
  std::vector<int> matrix;
  std::vector<int32_t> min_value(1, INT_MAX);

  std::random_device dev;
  std::mt19937 gen(dev());

  // Create data
  const size_t rows = 100;
  const size_t columns = 10;
  const int gen_min = -1000;
  const int gen_max = 1000;
  const int ref = INT_MIN;

  matrix = savchenko_m_min_matrix_seq::getRandomMatrix(rows, columns, gen_min, gen_max);
  const size_t index = gen() % (rows * columns);
  matrix[index] = INT_MIN;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(min_value.data()));
  taskDataSeq->outputs_count.emplace_back(min_value.size());

  // Create Task
  savchenko_m_min_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ref, min_value[0]);
}

TEST(savchenko_m_min_matrix_seq, test_min_10x100) {
  std::vector<int> matrix;
  std::vector<int32_t> min_value(1, INT_MAX);

  std::random_device dev;
  std::mt19937 gen(dev());

  // Create data
  const size_t rows = 10;
  const size_t columns = 100;
  const int gen_min = -1000;
  const int gen_max = 1000;
  const int ref = INT_MIN;

  matrix = savchenko_m_min_matrix_seq::getRandomMatrix(rows, columns, gen_min, gen_max);
  const size_t index = gen() % (rows * columns);
  matrix[index] = INT_MIN;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(min_value.data()));
  taskDataSeq->outputs_count.emplace_back(min_value.size());

  // Create Task
  savchenko_m_min_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ref, min_value[0]);
}

TEST(savchenko_m_min_matrix_seq, test_min_100x100) {
  std::vector<int> matrix;
  std::vector<int32_t> min_value(1, INT_MAX);

  std::random_device dev;
  std::mt19937 gen(dev());

  // Create data
  const size_t rows = 100;
  const size_t columns = 100;
  const int gen_min = -1000;
  const int gen_max = 1000;
  const int ref = INT_MIN;

  matrix = savchenko_m_min_matrix_seq::getRandomMatrix(rows, columns, gen_min, gen_max);
  const size_t index = gen() % (rows * columns);
  matrix[index] = INT_MIN;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(min_value.data()));
  taskDataSeq->outputs_count.emplace_back(min_value.size());

  // Create Task
  savchenko_m_min_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ref, min_value[0]);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
