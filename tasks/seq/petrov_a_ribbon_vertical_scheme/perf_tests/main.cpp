// Copyright 2024 Nesterov Alexander
#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/petrov_a_ribbon_vertical_scheme/include/ops_seq.hpp"

TEST(petrov_a_ribbon_vertical_scheme_seq, PerformanceTest) {
  const int rows = 10;
  const int cols = 10;

  std::vector<int> matrix(rows * cols);
  std::vector<int> vector(cols);
  std::vector<int> result(rows, 0);

  srand(time(nullptr));

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix[i * cols + j] = rand() % 200 - 100;
    }
  }

  for (int i = 0; i < cols; ++i) {
    vector[i] = rand() % 200 - 100;
  }
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
  taskDataSeq->inputs_count.push_back(vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(rows);

  // Create Task
  auto testTaskSeq = std::make_shared<petrov_a_ribbon_vertical_scheme_seq::TestTaskSequential>(taskDataSeq);

  ASSERT_EQ(testTaskSeq->validation(), true);

  auto start_time = std::chrono::high_resolution_clock::now();

  testTaskSeq->pre_processing();
  ASSERT_TRUE(testTaskSeq->run());
  testTaskSeq->post_processing();

  auto end_time = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
  std::cout << "Sequential implementation took " << duration << " ms." << std::endl;

  for (int i = 0; i < rows; ++i) {
    int expected_result = 0;
    for (int j = 0; j < cols; ++j) {
      expected_result += matrix[i * cols + j] * vector[j];
    }
    ASSERT_EQ(result[i], expected_result) << "Mismatch at row " << i;
  }

  ASSERT_FALSE(result.empty());
}

TEST(petrov_a_ribbon_vertical_scheme_seq, AdditionalTest) {
  const int rows = 5;
  const int cols = 5;

  std::vector<int> matrix(rows * cols);
  std::vector<int> vector(cols);
  std::vector<int> result(rows, 0);

  srand(time(nullptr));

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix[i * cols + j] = rand() % 200 - 100;
    }
  }

  for (int i = 0; i < cols; ++i) {
    vector[i] = rand() % 200 - 100;
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
  taskDataSeq->inputs_count.push_back(vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(rows);

  // Create Task
  auto testTaskSeq = std::make_shared<petrov_a_ribbon_vertical_scheme_seq::TestTaskSequential>(taskDataSeq);

  ASSERT_EQ(testTaskSeq->validation(), true);

  testTaskSeq->pre_processing();
  ASSERT_TRUE(testTaskSeq->run());
  testTaskSeq->post_processing();

  for (int i = 0; i < rows; ++i) {
    int expected_result = 0;
    for (int j = 0; j < cols; ++j) {
      expected_result += matrix[i * cols + j] * vector[j];
    }
    ASSERT_EQ(result[i], expected_result) << "Mismatch at row " << i;
  }

  ASSERT_FALSE(result.empty());
}