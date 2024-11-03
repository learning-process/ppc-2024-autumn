// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <climits>
#include <limits>
#include <random>
#include <vector>

#include "mpi/kovalchuk_a_max_of_vector_elements/include/ops_mpi.hpp"

using namespace kovalchuk_a_max_of_vector_elements_mpi;

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

// Тест для матрицы 5x5
TEST(kovalchuk_a_max_of_vector_elements_mpi, test_max_5_5) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

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

  TestTaskMPI testTaskMPI(taskDataSeq);
  ASSERT_EQ(testTaskMPI.validation(), true);
  testTaskMPI.pre_processing();
  testTaskMPI.run();
  testTaskMPI.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(reference, max_value[0]);
  }
}

// Тест для матрицы 10x10
TEST(kovalchuk_a_max_of_vector_elements_mpi, test_max_10_10) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

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

  TestTaskMPI testTaskMPI(taskDataSeq);
  ASSERT_EQ(testTaskMPI.validation(), true);
  testTaskMPI.pre_processing();
  testTaskMPI.run();
  testTaskMPI.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(reference, max_value[0]);
  }
}

// Тест для матрицы 50x50
TEST(kovalchuk_a_max_of_vector_elements_mpi, test_max_50_50) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

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

  TestTaskMPI testTaskMPI(taskDataSeq);
  ASSERT_EQ(testTaskMPI.validation(), true);
  testTaskMPI.pre_processing();
  testTaskMPI.run();
  testTaskMPI.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(reference, max_value[0]);
  }
}

// Тест для матрицы 100x100
TEST(kovalchuk_a_max_of_vector_elements_mpi, test_max_100_100) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

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

  TestTaskMPI testTaskMPI(taskDataSeq);
  ASSERT_EQ(testTaskMPI.validation(), true);
  testTaskMPI.pre_processing();
  testTaskMPI.run();
  testTaskMPI.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(reference, max_value[0]);
  }
}

TEST(kovalchuk_a_max_of_vector_elements_mpi, test_max_1_100) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

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

  TestTaskMPI testTaskMPI(taskDataSeq);
  ASSERT_EQ(testTaskMPI.validation(), true);
  testTaskMPI.pre_processing();
  testTaskMPI.run();
  testTaskMPI.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(reference, max_value[0]);
  }
}