// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/ermolaev_v_min_matrix/include/ops_seq.hpp"

TEST(ermolaev_v_min_matrix_seq, test_pipeline_run) {
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_min(1, INT_MAX);
  int ref = INT_MIN;

  std::random_device dev;
  std::mt19937 gen(dev());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int count_rows = 4000;
  int count_columns = 4000;
  int gen_min = -500;
  int gen_max = 500;

  global_matrix = ermolaev_v_min_matrix_seq::getRandomMatrix(count_rows, count_columns, gen_min, gen_max);
  int index = gen() % (count_rows * count_columns);
  global_matrix[index / count_columns][index / count_rows] = ref;

  for (unsigned int i = 0; i < global_matrix.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix[i].data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_columns);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_min.data()));
  taskDataSeq->outputs_count.emplace_back(global_min.size());

  // Create Task
  auto testTaskSequential = std::make_shared<ermolaev_v_min_matrix_seq::TestTaskSequential>(taskDataSeq);

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
  ASSERT_EQ(ref, global_min[0]);
}

TEST(sequential_ermolaev_v_min_matrix_seq, test_task_run) {
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_min(1, INT_MAX);
  int ref = INT_MIN;

  std::random_device dev;
  std::mt19937 gen(dev());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  int count_rows = 4000;
  int count_columns = 4000;
  int gen_min = -500;
  int gen_max = 500;

  global_matrix = ermolaev_v_min_matrix_seq::getRandomMatrix(count_rows, count_columns, gen_min, gen_max);
  int index = gen() % (count_rows * count_columns);
  global_matrix[index / count_columns][index / count_rows] = ref;

  for (unsigned int i = 0; i < global_matrix.size(); i++)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix[i].data()));
  taskDataSeq->inputs_count.emplace_back(count_rows);
  taskDataSeq->inputs_count.emplace_back(count_columns);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_min.data()));
  taskDataSeq->outputs_count.emplace_back(global_min.size());

  // Create Task
  auto testTaskSequential = std::make_shared<ermolaev_v_min_matrix_seq::TestTaskSequential>(taskDataSeq);

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
  ASSERT_EQ(ref, global_min[0]);
}