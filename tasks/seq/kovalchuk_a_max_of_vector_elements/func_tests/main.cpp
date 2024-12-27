// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/kovalchuk_a_max_of_vector_elements/include/ops_seq.hpp"

using namespace kovalchuk_a_max_of_vector_elements_seq;

static std::vector<int> getRandomVector(int sz, int min = MINIMALGEN, int max = MAXIMUMGEN);
static std::vector<std::vector<int>> getRandomMatrix(int rows, int columns, int min = MINIMALGEN, int max = MAXIMUMGEN);

static std::vector<int> getRandomVector(int sz, int min, int max) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = min + gen() % (max - min + 1);
  }
  return vec;
}

static std::vector<std::vector<int>> getRandomMatrix(int rows, int columns, int min, int max) {
  std::vector<std::vector<int>> vec(rows);
  for (int i = 0; i < rows; i++) {
    vec[i] = getRandomVector(columns, min, max);
  }
  return vec;
}

TEST(kovalchuk_a_max_of_vector_elements_seq, Test_Max_10_10) {
  const int count_rows = 10;
  const int count_columns = 10;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, INT_MIN);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  global_matrix = getRandomMatrix(count_rows, count_columns);
  std::random_device dev;
  std::mt19937 gen(dev());
  int index = gen() % (count_rows * count_columns);
  global_matrix[index / count_columns][index % count_columns] = INT_MAX;

  for (unsigned int i = 0; i < global_matrix.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
  taskDataSeq->outputs_count.emplace_back(global_max.size());
  // Create Task
  TestSequentialTask testSequentialTask(taskDataSeq);
  ASSERT_EQ(testSequentialTask.validation(), true);
  testSequentialTask.pre_processing();
  testSequentialTask.run();
  testSequentialTask.post_processing();

  ASSERT_EQ(global_max[0], INT_MAX);
}

TEST(kovalchuk_a_max_of_vector_elements_seq, Test_Max_50_20) {
  const int count_rows = 50;
  const int count_columns = 20;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, INT_MIN);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  global_matrix = getRandomMatrix(count_rows, count_columns);
  std::random_device dev;
  std::mt19937 gen(dev());
  int index = gen() % (count_rows * count_columns);
  global_matrix[index / count_columns][index % count_columns] = INT_MAX;

  for (unsigned int i = 0; i < global_matrix.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
  taskDataSeq->outputs_count.emplace_back(global_max.size());
  // Create Task
  TestSequentialTask testSequentialTask(taskDataSeq);
  ASSERT_EQ(testSequentialTask.validation(), true);
  testSequentialTask.pre_processing();
  testSequentialTask.run();
  testSequentialTask.post_processing();

  ASSERT_EQ(global_max[0], INT_MAX);
}

TEST(kovalchuk_a_max_of_vector_elements_seq, Test_Max_100_100) {
  const int count_rows = 100;
  const int count_columns = 100;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, INT_MIN);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  global_matrix = getRandomMatrix(count_rows, count_columns);
  std::random_device dev;
  std::mt19937 gen(dev());
  int index = gen() % (count_rows * count_columns);
  global_matrix[index / count_columns][index % count_columns] = INT_MAX;

  for (unsigned int i = 0; i < global_matrix.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
  taskDataSeq->outputs_count.emplace_back(global_max.size());
  // Create Task
  TestSequentialTask testSequentialTask(taskDataSeq);
  ASSERT_EQ(testSequentialTask.validation(), true);
  testSequentialTask.pre_processing();
  testSequentialTask.run();
  testSequentialTask.post_processing();

  ASSERT_EQ(global_max[0], INT_MAX);
}

TEST(kovalchuk_a_max_of_vector_elements_seq, Test_Max_1_100) {
  const int count_rows = 1;
  const int count_columns = 100;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, INT_MIN);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  global_matrix = getRandomMatrix(count_rows, count_columns);
  std::random_device dev;
  std::mt19937 gen(dev());
  int index = gen() % (count_rows * count_columns);
  global_matrix[index / count_columns][index % count_columns] = INT_MAX;

  for (unsigned int i = 0; i < global_matrix.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
  taskDataSeq->outputs_count.emplace_back(global_max.size());
  // Create Task
  TestSequentialTask testSequentialTask(taskDataSeq);
  ASSERT_EQ(testSequentialTask.validation(), true);
  testSequentialTask.pre_processing();
  testSequentialTask.run();
  testSequentialTask.post_processing();

  ASSERT_EQ(global_max[0], INT_MAX);
}

TEST(kovalchuk_a_max_of_vector_elements_seq, Test_Max_4_4) {
  const int count_rows = 4;
  const int count_columns = 4;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, INT_MIN);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  global_matrix = getRandomMatrix(count_rows, count_columns);
  std::random_device dev;
  std::mt19937 gen(dev());
  int index = gen() % (count_rows * count_columns);
  global_matrix[index / count_columns][index % count_columns] = INT_MAX;

  for (unsigned int i = 0; i < global_matrix.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
  taskDataSeq->outputs_count.emplace_back(global_max.size());
  // Create Task
  TestSequentialTask testSequentialTask(taskDataSeq);
  ASSERT_EQ(testSequentialTask.validation(), true);
  testSequentialTask.pre_processing();
  testSequentialTask.run();
  testSequentialTask.post_processing();

  ASSERT_EQ(global_max[0], INT_MAX);
}

TEST(kovalchuk_a_max_of_vector_elements_seq, Test_Max_Negative_Values) {
  const int count_rows = 1;
  const int count_columns = 100;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, INT_MIN);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  global_matrix = getRandomMatrix(count_rows, count_columns, -1, -999);
  std::random_device dev;
  std::mt19937 gen(dev());
  int index = gen() % (count_rows * count_columns);
  global_matrix[index / count_columns][index % count_columns] = INT_MAX;

  for (unsigned int i = 0; i < global_matrix.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
  taskDataSeq->outputs_count.emplace_back(global_max.size());
  // Create Task
  TestSequentialTask testSequentialTask(taskDataSeq);
  ASSERT_EQ(testSequentialTask.validation(), true);
  testSequentialTask.pre_processing();
  testSequentialTask.run();
  testSequentialTask.post_processing();

  ASSERT_EQ(global_max[0], INT_MAX);
}

TEST(kovalchuk_a_max_of_vector_elements_seq, Test_Max_Same_Values) {
  const int count_rows = 10;
  const int count_columns = 100;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, INT_MIN);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  global_matrix = getRandomMatrix(count_rows, count_columns, 20, 20);
  std::random_device dev;
  std::mt19937 gen(dev());
  int index = gen() % (count_rows * count_columns);
  global_matrix[index / count_columns][index % count_columns] = INT_MAX;

  for (unsigned int i = 0; i < global_matrix.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
  taskDataSeq->outputs_count.emplace_back(global_max.size());
  // Create Task
  TestSequentialTask testSequentialTask(taskDataSeq);
  ASSERT_EQ(testSequentialTask.validation(), true);
  testSequentialTask.pre_processing();
  testSequentialTask.run();
  testSequentialTask.post_processing();

  ASSERT_EQ(global_max[0], INT_MAX);
}
