// Copyright 2024 Khovansky Dmitry
#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/khovansky_d_ribbon_vertical_scheme/include/ops_seq.hpp"

TEST(khovansky_d_ribbon_vertical_scheme_seq, test_pipeline_run) {
  std::vector<int> matrix(2000000, 0);
  std::vector<int> vector(10, 0);
  std::vector<int> output(200000, 0);
  std::vector<int> result(200000, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
  taskDataSeq->inputs_count.emplace_back(vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());

  auto testTaskSequential = std::make_shared<khovansky_d_ribbon_vertical_scheme_seq::RibbonVerticalSchemeSeq>(taskDataSeq);
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
  ASSERT_EQ(result, output);
}

TEST(khovansky_d_ribbon_vertical_scheme_seq, test_task_run) {
  std::vector<int> matrix(2000000, 0);
  std::vector<int> vector(10, 0);
  std::vector<int> output(200000, 0);
  std::vector<int> result(200000, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vector.data()));
  taskDataSeq->inputs_count.emplace_back(vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());

  auto testTaskSequential = std::make_shared<khovansky_d_ribbon_vertical_scheme_seq::RibbonVerticalSchemeSeq>(taskDataSeq);
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
  ASSERT_EQ(result, output);
}