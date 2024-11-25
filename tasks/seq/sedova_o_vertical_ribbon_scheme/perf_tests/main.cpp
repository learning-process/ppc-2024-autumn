// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/sedova_o_vertical_ribbon_scheme/include/ops_seq.hpp"



TEST(sedova_o_vertical_ribbon_scheme, test_small_matrix) {
  const int rows = 100;
  const int cols = 50;
  const int iterations = 10;

  std::vector<int> matrix(rows * cols);
  std::vector<int> vector(cols);
  std::vector<int> result(rows, 0);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(-10, 10);

  for (int i = 0; i < rows * cols; ++i) matrix[i] = distrib(gen);
  for (int i = 0; i < cols; ++i) vector[i] = distrib(gen);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.push_back(reinterpret_cast<std::uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.push_back(matrix.size());
  taskDataSeq->inputs.push_back(reinterpret_cast<std::uint8_t*>(vector.data()));
  taskDataSeq->inputs_count.push_back(vector.size());
  taskDataSeq->outputs.push_back(reinterpret_cast<std::uint8_t*>(result.data()));
  taskDataSeq->outputs_count.push_back(result.size());
  sedova_o_vertical_ribbon_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    testTaskSequential.validation();
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Average execution time (small): " << (double)duration.count() / iterations << " microseconds\n";
}

TEST(sedova_o_vertical_ribbon_scheme, test_medium_matrix) {
  const int rows = 1000;
  const int cols = 500;
  const int iterations = 5;  // Reduce iterations for larger matrices

  std::vector<int> matrix(rows * cols);
  std::vector<int> vector(cols);
  std::vector<int> result(rows, 0);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(-10, 10);

  for (int i = 0; i < rows * cols; ++i) matrix[i] = distrib(gen);
  for (int i = 0; i < cols; ++i) vector[i] = distrib(gen);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.push_back(reinterpret_cast<std::uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.push_back(matrix.size());
  taskDataSeq->inputs.push_back(reinterpret_cast<std::uint8_t*>(vector.data()));
  taskDataSeq->inputs_count.push_back(vector.size());
  taskDataSeq->outputs.push_back(reinterpret_cast<std::uint8_t*>(result.data()));
  taskDataSeq->outputs_count.push_back(result.size());
  sedova_o_vertical_ribbon_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    testTaskSequential.validation();
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Average execution time (medium): " << (double)duration.count() / iterations << " microseconds\n";
}

TEST(sedova_o_vertical_ribbon_scheme, test_large_matrix) {
  const int rows = 5000;
  const int cols = 2500;
  const int iterations = 2;  // Reduce iterations further for large matrices

  std::vector<int> matrix(rows * cols);
  std::vector<int> vector(cols);
  std::vector<int> result(rows, 0);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(-10, 10);

  for (int i = 0; i < rows * cols; ++i) matrix[i] = distrib(gen);
  for (int i = 0; i < cols; ++i) vector[i] = distrib(gen);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.push_back(reinterpret_cast<std::uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.push_back(matrix.size());
  taskDataSeq->inputs.push_back(reinterpret_cast<std::uint8_t*>(vector.data()));
  taskDataSeq->inputs_count.push_back(vector.size());
  taskDataSeq->outputs.push_back(reinterpret_cast<std::uint8_t*>(result.data()));
  taskDataSeq->outputs_count.push_back(result.size());
  sedova_o_vertical_ribbon_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    testTaskSequential.validation();
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Average execution time (large): " << (double)duration.count() / iterations << " microseconds\n";
}