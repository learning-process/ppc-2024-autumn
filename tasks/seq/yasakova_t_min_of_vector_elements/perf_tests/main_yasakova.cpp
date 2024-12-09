// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/yasakova_t_min_of_vector_elements/include/ops_seq_yasakova.hpp"

std::vector<int> RandomVector(int size, int minimum = 0, int maximum = 100) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(size);
  for (int i = 0; i < size; i++) {
    vec[i] = minimum + gen() % (maximum - minimum + 1);
  }
  return vec;
}

std::vector<std::vector<int>> RandomMatrix(int rows, int columns, int minimum = 0, int maximum = 100) {
  std::vector<std::vector<int>> vec(rows);
  for (int i = 0; i < rows; i++) {
    vec[i] = RandomVector(columns, minimum, maximum);
  }
  return vec;
}

TEST(yasakova_t_min_of_vector_elements_seq, test_pipeline_run) {
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_minimum(1, INT_MAX);
  int ref = INT_MIN;
  std::random_device dev;
  std::mt19937 gen(dev());
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int count_rows = 4000;
  int count_columns = 4000;
  int gen_minimum = -500;
  int gen_maximum = 500;
  global_matrix = RandomMatrix(count_rows, count_columns, gen_minimum, gen_maximum);
  int index = gen() % (count_rows * count_columns);
  global_matrix[index / count_columns][index / count_rows] = ref;
  for (unsigned int i = 0; i < global_matrix.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix[i].data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_minimum.data()));
  taskDataSeq->outputs_count.emplace_back(global_minimum.size());
  auto testTaskSequential = std::make_shared<yasakova_t_min_of_vector_elements_seq::TestTaskSequential>(taskDataSeq);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(ref, global_minimum[0]);
}

TEST(yasakova_t_min_of_vector_elements_seq, test_task_run) {
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_minimum(1, INT_MAX);
  int ref = INT_MIN;
  std::random_device dev;
  std::mt19937 gen(dev());
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int count_rows = 4000;
  int count_columns = 4000;
  int gen_minimum = -500;
  int gen_maximum = 500;
  global_matrix = RandomMatrix(count_rows, count_columns, gen_minimum, gen_maximum);
  int index = gen() % (count_rows * count_columns);
  global_matrix[index / count_columns][index / count_rows] = ref;
  for (unsigned int i = 0; i < global_matrix.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix[i].data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_columns);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_minimum.data()));
  taskDataSeq->outputs_count.emplace_back(global_minimum.size());
  auto testTaskSequential = std::make_shared<yasakova_t_min_of_vector_elements_seq::TestTaskSequential>(taskDataSeq);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(ref, global_minimum[0]);
}