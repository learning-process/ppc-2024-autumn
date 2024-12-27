// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/zolotareva_a_smoothing_image/include/ops_seq.hpp"

TEST(sequential_zolotareva_a_smoothing_image_perf_test, test_pipeline_run) {
  const uint32_t height = 1000;
  const uint32_t width = 1000;
  std::vector<uint8_t> input(width * height, 0);
  std::vector<uint8_t> output(width * height);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(input.data());
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->outputs.emplace_back(output.data());
  taskDataSeq->outputs_count.emplace_back(output.size());

  auto testTaskSequential = std::make_shared<zolotareva_a_smoothing_image_seq::TestTaskSequential>(taskDataSeq);

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

  ASSERT_EQ(height, taskDataSeq->inputs_count.back());
}

TEST(sequential_zolotareva_a_smoothing_image_perf_test, test_task_run) {
  const uint32_t height = 1000;
  const uint32_t width = 1000;
  std::vector<uint8_t> input(width * height, 0);
  std::vector<uint8_t> output(width * height);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(input.data());
  taskDataSeq->inputs_count.emplace_back(height);
  taskDataSeq->inputs_count.emplace_back(width);
  taskDataSeq->outputs.emplace_back(output.data());
  taskDataSeq->outputs_count.emplace_back(output.size());

  auto testTaskSequential = std::make_shared<zolotareva_a_smoothing_image_seq::TestTaskSequential>(taskDataSeq);

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
  ASSERT_EQ(height, taskDataSeq->inputs_count.back());
}