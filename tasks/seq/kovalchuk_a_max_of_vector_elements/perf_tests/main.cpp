// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <chrono>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
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

TEST(kovalchuk_a_max_of_vector_elements_seq, test_pipeline_run) {
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, INT_MIN);
  int ref = INT_MAX;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::random_device dev;
  std::mt19937 gen(dev());
  int count_rows = 1000;
  int count_columns = 1000;
  global_matrix = getRandomMatrix(count_rows, count_columns);
  size_t index = gen() % (static_cast<size_t>(count_rows) * count_columns);
  global_matrix[index / count_columns][index % count_columns] = ref;
  for (unsigned int i = 0; i < global_matrix.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
  taskDataSeq->outputs_count.emplace_back(global_max.size());
  // Create Task
  auto testSequentialTask = std::make_shared<kovalchuk_a_max_of_vector_elements_seq::TestSequentialTask>(taskDataSeq);
  ASSERT_EQ(testSequentialTask->validation(), true);
  testSequentialTask->pre_processing();
  testSequentialTask->run();
  testSequentialTask->post_processing();
  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    return std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() -
                                                                     start_time)
        .count();
  };
  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testSequentialTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(ref, global_max[0]);
}

TEST(kovalchuk_a_max_of_vector_elements_seq, test_task_run) {
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max(1, INT_MIN);
  int ref = INT_MAX;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::random_device dev;
  std::mt19937 gen(dev());
  int count_rows = 3;
  int count_columns = 3;
  global_matrix = getRandomMatrix(count_rows, count_columns);
  size_t index = gen() % (static_cast<size_t>(count_rows) * count_columns);
  global_matrix[index / count_columns][index % count_columns] = ref;
  for (unsigned int i = 0; i < global_matrix.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix[i].data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
  taskDataSeq->outputs_count.emplace_back(global_max.size());
  // Create Task
  auto testSequentialTask = std::make_shared<kovalchuk_a_max_of_vector_elements_seq::TestSequentialTask>(taskDataSeq);
  ASSERT_EQ(testSequentialTask->validation(), true);
  testSequentialTask->pre_processing();
  testSequentialTask->run();
  testSequentialTask->post_processing();
  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    return std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() -
                                                                     start_time)
        .count();
  };
  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testSequentialTask);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(ref, global_max[0]);
}
