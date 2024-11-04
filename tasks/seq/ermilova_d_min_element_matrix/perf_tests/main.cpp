// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/ermilova_d_min_element_matrix/include/ops_seq.hpp"

std::vector<int> getRandomVector(int size, int upper_border, int lower_border) {
  std::random_device dev;
  std::mt19937 gen(dev());
  if (size <= 0) throw "Incorrect size";
  std::vector<int> vec(size);
  for (int i = 0; i < size; i++) {
    vec[i] = lower_border + gen() % (upper_border - lower_border + 1);
  }
  return vec;
}

std::vector<std::vector<int>> getRandomMatrix(int rows, int cols, int upper_border, int lower_border) {
  if (rows <= 0 || cols <= 0) throw "Incorrect size";
  std::vector<std::vector<int>> vec(rows);
  for (int i = 0; i < rows; i++) {
    vec[i] = getRandomVector(cols, upper_border, lower_border);
  }
  return vec;
}

TEST(ermilova_d_min_element_matrix_seq, test_pipeline_run) {
  const int rows_test = 1000;
  const int cols_test = 1000;
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;
  int reference_min = -5000;
  //  Create data
  std::vector<std::vector<int>> in = getRandomMatrix(rows_test, cols_test, upper_border_test, lower_border_test);
  std::vector<int> out(1, INT_MAX);

  std::random_device dev;
  std::mt19937 gen(dev());
  int rnd_rows = gen() % rows_test;
  int rnd_cols = gen() % cols_test;
  in[rnd_rows][rnd_cols] = reference_min;

  // Create  TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }
  taskDataSeq->inputs_count.emplace_back(rows_test);
  taskDataSeq->inputs_count.emplace_back(cols_test);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<ermilova_d_min_element_matrix_seq::TestTaskSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(reference_min, out[0]);
}

TEST(ermilova_d_min_element_matrix_seq, test_task_run) {
  const int rows_test = 1000;
  const int cols_test = 1000;
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;
  int reference_min = -5000;

  // Create data
  std::vector<std::vector<int>> in = getRandomMatrix(rows_test, cols_test, upper_border_test, lower_border_test);
  std::vector<int> out(1, INT_MAX);

  std::random_device dev;
  std::mt19937 gen(dev());
  int rnd_rows = gen() % rows_test;
  int rnd_cols = gen() % cols_test;
  in[rnd_rows][rnd_cols] = reference_min;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (unsigned int i = 0; i < in.size(); i++) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }
  taskDataSeq->inputs_count.emplace_back(rows_test);
  taskDataSeq->inputs_count.emplace_back(cols_test);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<ermilova_d_min_element_matrix_seq::TestTaskSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(reference_min, out[0]);
}
