// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <climits>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/savchenko_m_min_matrix/include/ops_seq.hpp"

TEST(savchenko_m_min_matrix_seq, test_pipeline_run) {
  std::vector<int> matrix;
  std::vector<int32_t> min_value(1, INT_MAX);
  
  std::random_device dev;
  std::mt19937 gen(dev());

  // Create data
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int rows = 5000;
  int columns = 5000;
  int gen_min = -1000;
  int gen_max = 1000;
  int ref = INT_MIN;

  matrix = savchenko_m_min_matrix_seq::getRandomMatrix(rows, columns, gen_min, gen_max);
  int index = gen() % (rows * columns);
  matrix[index] = INT_MIN;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(min_value.data()));
  taskDataSeq->outputs_count.emplace_back(min_value.size());

  // Create Task
  auto testTaskSequential = std::make_shared<savchenko_m_min_matrix_seq::TestTaskSequential>(taskDataSeq);

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
  ASSERT_EQ(ref, min_value[0]);
}

TEST(savchenko_m_min_matrix_seq, test_task_run) {
  std::vector<int> matrix;
  std::vector<int32_t> min_value(1, INT_MAX);
  int ref = INT_MIN;

  std::random_device dev;
  std::mt19937 gen(dev());

  // Create data
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int rows = 5000;
  int columns = 5000;
  int gen_min = -1000;
  int gen_max = 1000;

  matrix = savchenko_m_min_matrix_seq::getRandomMatrix(rows, columns, gen_min, gen_max);
  int index = gen() % (rows * columns);
  matrix[index] = INT_MIN;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(min_value.data()));
  taskDataSeq->outputs_count.emplace_back(min_value.size());

  // Create Task
  auto testTaskSequential = std::make_shared<savchenko_m_min_matrix_seq::TestTaskSequential>(taskDataSeq);

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
  ASSERT_EQ(ref, min_value[0]);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
