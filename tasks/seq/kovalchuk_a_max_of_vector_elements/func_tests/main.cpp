// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <limits>
#include <random>
#include <vector>

#include "seq/kovalchuk_a_max_of_vector_elements/include/ops_seq.hpp"

using namespace kovalchuk_a_max_of_vector_elements_seq;

std::vector<std::vector<int>> generateRandomMatrix(int rows, int columns, int start_gen = -99, int fin_gen = 99) {
  static std::random_device dev;
  static std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(start_gen, fin_gen);
  std::vector<std::vector<int>> matrix(rows, std::vector<int>(columns));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < columns; ++j) {
      matrix[i][j] = dist(gen);
    }
  }
  return matrix;
}

// Òåñò äëÿ ìàòðèöû 5x5
TEST(kovalchuk_a_max_of_vector_elements_seq, test_max_5x5) {
  const int rows = 5;
  const int columns = 5;
  int reference = std::numeric_limits<int>::max();

  std::vector<int> max_value(1, std::numeric_limits<int>::min());
  std::vector<std::vector<int>> matrix = generateRandomMatrix(rows, columns);
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> row_dist(0, rows - 1);
  std::uniform_int_distribution<int> col_dist(0, columns - 1);
  int random_row = row_dist(gen);
  int random_col = col_dist(gen);
  matrix[random_row][random_col] = reference;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < matrix.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix[i].data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(max_value.data()));
  taskDataSeq->outputs_count.emplace_back(max_value.size());

  TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(reference, max_value[0]);
}

// Òåñò äëÿ ìàòðèöû 10x10
TEST(kovalchuk_a_max_of_vector_elements_seq, test_max_10x10) {
  const int rows = 10;
  const int columns = 10;
  int reference = std::numeric_limits<int>::max();

  std::vector<int> max_value(1, std::numeric_limits<int>::min());
  std::vector<std::vector<int>> matrix = generateRandomMatrix(rows, columns);
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> row_dist(0, rows - 1);
  std::uniform_int_distribution<int> col_dist(0, columns - 1);
  int random_row = row_dist(gen);
  int random_col = col_dist(gen);
  matrix[random_row][random_col] = reference;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < matrix.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix[i].data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(max_value.data()));
  taskDataSeq->outputs_count.emplace_back(max_value.size());

  TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(reference, max_value[0]);
}

// Òåñò äëÿ ìàòðèöû 50x50
TEST(kovalchuk_a_max_of_vector_elements_seq, test_max_50x50) {
  const int rows = 50;
  const int columns = 50;
  int reference = std::numeric_limits<int>::max();

  std::vector<int> max_value(1, std::numeric_limits<int>::min());
  std::vector<std::vector<int>> matrix = generateRandomMatrix(rows, columns);
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> row_dist(0, rows - 1);
  std::uniform_int_distribution<int> col_dist(0, columns - 1);
  int random_row = row_dist(gen);
  int random_col = col_dist(gen);
  matrix[random_row][random_col] = reference;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < matrix.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix[i].data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(max_value.data()));
  taskDataSeq->outputs_count.emplace_back(max_value.size());

  TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(reference, max_value[0]);
}

// Òåñò äëÿ ìàòðèöû 100x100
TEST(kovalchuk_a_max_of_vector_elements_seq, test_max_100x100) {
  const int rows = 100;
  const int columns = 100;
  int reference = std::numeric_limits<int>::max();

  std::vector<int> max_value(1, std::numeric_limits<int>::min());
  std::vector<std::vector<int>> matrix = generateRandomMatrix(rows, columns);
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> row_dist(0, rows - 1);
  std::uniform_int_distribution<int> col_dist(0, columns - 1);
  int random_row = row_dist(gen);
  int random_col = col_dist(gen);
  matrix[random_row][random_col] = reference;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < matrix.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix[i].data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(max_value.data()));
  taskDataSeq->outputs_count.emplace_back(max_value.size());

  TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(reference, max_value[0]);
}

TEST(kovalchuk_a_max_of_vector_elements_seq, test_max_1x100) {
  const int rows = 1;
  const int columns = 100;
  int reference = std::numeric_limits<int>::max();

  std::vector<int> max_value(1, std::numeric_limits<int>::min());
  std::vector<std::vector<int>> matrix = generateRandomMatrix(rows, columns);
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> row_dist(0, rows - 1);
  std::uniform_int_distribution<int> col_dist(0, columns - 1);
  int random_row = row_dist(gen);
  int random_col = col_dist(gen);
  matrix[random_row][random_col] = reference;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < matrix.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix[i].data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(max_value.data()));
  taskDataSeq->outputs_count.emplace_back(max_value.size());

  TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(reference, max_value[0]);
}
