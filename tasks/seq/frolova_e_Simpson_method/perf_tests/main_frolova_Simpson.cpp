// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/frolova_e_Simpson_method/include/ops_seq_frolova_Simpson.hpp"

TEST(frolova_e_Simpson_method_seq, test_pipeline_run) {
  std::vector<int> values_1 = {1000, 2};
  std::vector<double> values_2 = {0.0, 1000.0, 0.0, 1000.0};

  std::vector<double> res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_1.data()));
  taskDataSeq->inputs_count.emplace_back(values_1.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
  taskDataSeq->inputs_count.emplace_back(values_2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size() * sizeof(double));

  // Create Task
  auto testTaskSequential = std::make_shared<frolova_e_Simpson_method_seq::Simpsonmethod>(
      taskDataSeq, frolova_e_Simpson_method_seq::ProductOfXAndY);

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

  ASSERT_EQ(250000000000, res[0]);
}

TEST(frolova_e_Simpson_method_seq, test_task_run) {
  // Create data
  std::vector<int> values_1 = {1000, 2};
  std::vector<double> values_2 = {0.0, 1000.0, 0.0, 1000.0};

  std::vector<double> res(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_1.data()));
  taskDataSeq->inputs_count.emplace_back(values_1.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
  taskDataSeq->inputs_count.emplace_back(values_2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size() * sizeof(double));

  // Create Task
  auto testTaskSequential = std::make_shared<frolova_e_Simpson_method_seq ::Simpsonmethod>(
      taskDataSeq, frolova_e_Simpson_method_seq::ProductOfXAndY);

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
  ASSERT_EQ(250000000000, res[0]);
}