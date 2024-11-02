// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <climits>
#include <random>
#include <vector>

#include "seq/savchenko_m_min_matrix/include/ops_seq_savchenko.hpp"

TEST(savchenko_m_min_matrix_seq, test_min_10x10) {
  std::vector<int> matrix;
  std::vector<int32_t> min_value(1, INT_MAX);

  std::random_device dev;
  std::mt19937 gen(dev());

  // Create data
  const int rows = 10;
  const int columns = 10;
  const int gen_min = -1000;
  const int gen_max = 1000;
  const int ref = INT_MIN;

  matrix = savchenko_m_min_matrix_seq::getRandomMatrix(rows, columns, gen_min, gen_max);
  const int index = gen() % (rows * columns);
  matrix[index] = INT_MIN;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(columns);

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
  const int rows = 100;
  const int columns = 10;
  const int gen_min = -1000;
  const int gen_max = 1000;
  const int ref = INT_MIN;

  matrix = savchenko_m_min_matrix_seq::getRandomMatrix(rows, columns, gen_min, gen_max);
  const int index = gen() % (rows * columns);
  matrix[index] = INT_MIN;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(columns);

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
  const int rows = 10;
  const int columns = 100;
  const int gen_min = -1000;
  const int gen_max = 1000;
  const int ref = INT_MIN;

  matrix = savchenko_m_min_matrix_seq::getRandomMatrix(rows, columns, gen_min, gen_max);
  const int index = gen() % (rows * columns);
  matrix[index] = INT_MIN;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(columns);

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
  const int rows = 100;
  const int columns = 100;
  const int gen_min = -1000;
  const int gen_max = 1000;
  const int ref = INT_MIN;

  matrix = savchenko_m_min_matrix_seq::getRandomMatrix(rows, columns, gen_min, gen_max);
  const int index = gen() % (rows * columns);
  matrix[index] = INT_MIN;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(columns);

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

TEST(savchenko_m_min_matrix_seq, test_min_100x1) {
  std::vector<int> matrix;
  std::vector<int32_t> min_value(1, INT_MAX);

  std::random_device dev;
  std::mt19937 gen(dev());

  // Create data
  const int rows = 100;
  const int columns = 1;
  const int gen_min = -1000;
  const int gen_max = 1000;
  const int ref = INT_MIN;

  matrix = savchenko_m_min_matrix_seq::getRandomMatrix(rows, columns, gen_min, gen_max);
  const int index = gen() % (rows * columns);
  matrix[index] = INT_MIN;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(columns);

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

TEST(savchenko_m_min_matrix_seq, test_min_1000x1) {
  std::vector<int> matrix;
  std::vector<int32_t> min_value(1, INT_MAX);

  std::random_device dev;
  std::mt19937 gen(dev());

  // Create data
  const int rows = 1000;
  const int columns = 1;
  const int gen_min = -1000;
  const int gen_max = 1000;
  const int ref = INT_MIN;

  matrix = savchenko_m_min_matrix_seq::getRandomMatrix(rows, columns, gen_min, gen_max);
  const int index = gen() % (rows * columns);
  matrix[index] = INT_MIN;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(columns);

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

TEST(savchenko_m_min_matrix_seq, test_min_1x100) {
  std::vector<int> matrix;
  std::vector<int32_t> min_value(1, INT_MAX);

  std::random_device dev;
  std::mt19937 gen(dev());

  // Create data
  const int rows = 1;
  const int columns = 100;
  const int gen_min = -1000;
  const int gen_max = 1000;
  const int ref = INT_MIN;

  matrix = savchenko_m_min_matrix_seq::getRandomMatrix(rows, columns, gen_min, gen_max);
  const int index = gen() % (rows * columns);
  matrix[index] = INT_MIN;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(columns);

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

TEST(savchenko_m_min_matrix_seq, test_min_1x1000) {
  std::vector<int> matrix;
  std::vector<int32_t> min_value(1, INT_MAX);

  std::random_device dev;
  std::mt19937 gen(dev());

  // Create data
  const int rows = 1;
  const int columns = 1000;
  const int gen_min = -1000;
  const int gen_max = 1000;
  const int ref = INT_MIN;

  matrix = savchenko_m_min_matrix_seq::getRandomMatrix(rows, columns, gen_min, gen_max);
  const int index = gen() % (rows * columns);
  matrix[index] = INT_MIN;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(columns);

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

TEST(savchenko_m_min_matrix_seq, test_min_0x10) {
  std::vector<int> matrix;
  std::vector<int32_t> min_value(1, INT_MAX);

  // Create data
  const int rows = 0;
  const int columns = 10;
  const int gen_min = -1000;
  const int gen_max = 1000;

  matrix = savchenko_m_min_matrix_seq::getRandomMatrix(rows, columns, gen_min, gen_max);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(columns);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(min_value.data()));
  taskDataSeq->outputs_count.emplace_back(min_value.size());

  // Create Task
  savchenko_m_min_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_FALSE(testTaskSequential.validation());
}

// TEST(savchenko_m_min_matrix_seq, test_min_0x10) {
//   std::vector<int> matrix(1, 0);
//   std::vector<int32_t> min_value(1, INT_MAX);

//  // Create data
//  const int rows = 0;
//  const int columns = 10;

//  // Create TaskData
//  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
//  taskDataSeq->inputs_count.emplace_back(rows);
//  taskDataSeq->inputs_count.emplace_back(columns);

//  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(min_value.data()));
//  taskDataSeq->outputs_count.emplace_back(min_value.size());

//  // Create Task
//  savchenko_m_min_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
//  ASSERT_EQ(testTaskSequential.validation(), false);
//}

// TEST(savchenko_m_min_matrix_seq, test_min_10x0) {
//   std::vector<int> matrix(1, 0);
//   std::vector<int32_t> min_value(1, INT_MAX);
//
//   // Create data
//   const int rows = 10;
//   const int columns = 0;
//
//   // Create TaskData
//   std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//   taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
//   taskDataSeq->inputs_count.emplace_back(rows);
//   taskDataSeq->inputs_count.emplace_back(columns);
//
//   taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(min_value.data()));
//   taskDataSeq->outputs_count.emplace_back(min_value.size());
//
//   // Create Task
//   savchenko_m_min_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
//   ASSERT_EQ(testTaskSequential.validation(), false);
// }
