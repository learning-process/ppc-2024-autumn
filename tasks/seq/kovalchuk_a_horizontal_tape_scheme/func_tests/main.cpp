#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/kovalchuk_a_horizontal_tape_scheme/include/ops_seq.hpp"

using namespace kovalchuk_a_horizontal_tape_scheme_seq;

std::vector<int> getRandomVector(int sz, int min = MINIMALGEN, int max = MAXIMUMGEN);
std::vector<std::vector<int>> getRandomMatrix(int rows, int columns, int min = MINIMALGEN, int max = MAXIMUMGEN);

std::vector<int> getRandomVector(int sz, int min, int max) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = min + gen() % (max - min + 1);
  }
  return vec;
}

std::vector<std::vector<int>> getRandomMatrix(int rows, int columns, int min, int max) {
  std::vector<std::vector<int>> vec(rows);
  for (int i = 0; i < rows; i++) {
    vec[i] = getRandomVector(columns, min, max);
  }
  return vec;
}

TEST(kovalchuk_a_horizontal_tape_scheme_seq, Test_Matrix_10_10) {
  const int count_rows = 10;
  const int count_columns = 10;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int> global_vector;
  std::vector<int> global_result(count_rows, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  global_matrix = getRandomMatrix(count_rows, count_columns);
  global_vector = getRandomVector(count_columns);
  for (unsigned int i = 0; i < global_matrix.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  taskDataSeq->outputs_count.emplace_back(global_result.size());
  // Create Task
  TestSequentialTask testSequentialTask(taskDataSeq);
  ASSERT_EQ(testSequentialTask.validation(), true);
  testSequentialTask.pre_processing();
  testSequentialTask.run();
  testSequentialTask.post_processing();

  std::vector<int> reference_result(count_rows, 0);
  for (int i = 0; i < count_rows; i++) {
    for (int j = 0; j < count_columns; j++) {
      reference_result[i] += global_matrix[i][j] * global_vector[j];
    }
  }

  ASSERT_EQ(global_result, reference_result);
}
