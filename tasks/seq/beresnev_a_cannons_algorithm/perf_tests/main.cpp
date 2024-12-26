// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/beresnev_a_cannons_algorithm/include/ops_seq.hpp"

TEST(beresnev_a_cannons_algorithm_seq, test_pipeline_run) {
  size_t n = 500;
  std::vector<double> inA(n * n, 0.0);
  std::vector<double> inB(n * n, 1.0);
  std::vector<double> outC(n * n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inA.data()));
  taskDataSeq->inputs_count.emplace_back(inA.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inB.data()));
  taskDataSeq->inputs_count.emplace_back(inB.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&outC));
  taskDataSeq->outputs_count.emplace_back(outC.size());

  auto testTaskSequential = std::make_shared<beresnev_a_cannons_algorithm_seq::TestTaskSequential>(taskDataSeq);
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
  ASSERT_TRUE(
      std::equal(inA.begin(), inA.end(), outC.begin(), [](double a, double b) { return std::abs(a - b) < 1e-9; }));
}

TEST(beresnev_a_cannons_algorithm_seq, test_task_run) {
  size_t n = 500;
  std::vector<double> inA(n * n, 0.0);
  std::vector<double> inB(n * n, 1.0);
  std::vector<double> outC(n * n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inA.data()));
  taskDataSeq->inputs_count.emplace_back(inA.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inB.data()));
  taskDataSeq->inputs_count.emplace_back(inB.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&outC));
  taskDataSeq->outputs_count.emplace_back(outC.size());

  auto testTaskSequential = std::make_shared<beresnev_a_cannons_algorithm_seq::TestTaskSequential>(taskDataSeq);

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
  ASSERT_TRUE(
      std::equal(inA.begin(), inA.end(), outC.begin(), [](double a, double b) { return std::abs(a - b) < 1e-9; }));
}