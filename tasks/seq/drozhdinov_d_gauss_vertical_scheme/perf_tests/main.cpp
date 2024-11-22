// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>
// not example
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/drozhdinov_d_gauss_vertical_scheme/include/ops_seq.hpp"

TEST(drozhdinov_d_perf_test, test_pipeline_run) {
  int rows = 1000;
  int columns = 1000;
  std::vector<double> matrix = genElementaryMatrix(rows, columns);
  std::vector<double> b(rows * columns, 1);
  std::vector<double> expres(rows);
  std::vector<double> res(rows, 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(b.size());
  taskDataSeq->inputs_count.emplace_back(columns);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres.data()));
  taskDataSeq->outputs_count.emplace_back(expres.size());

  // Create Task
  auto testTaskSequential = std::make_shared<drozhdinov_d_gauss_vertical_scheme_seq::TestTaskSequential>(taskDataSeq);

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
  ASSERT_EQ(expres, res);
}

TEST(drozhdinov_d_perf_test, test_task_run) {
  int rows = 1000;
  int columns = 1000;
  std::vector<double> matrix = genElementaryMatrix(rows, columns);
  std::vector<double> b(rows * columns, 1);
  std::vector<double> expres(rows);
  std::vector<double> res(rows, 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(b.size());
  taskDataSeq->inputs_count.emplace_back(columns);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres.data()));
  taskDataSeq->outputs_count.emplace_back(expres.size());

  // Create Task
  auto testTaskSequential = std::make_shared<drozhdinov_d_gauss_vertical_scheme_seq::TestTaskSequential>(taskDataSeq);

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
  ASSERT_EQ(expres, res);
}