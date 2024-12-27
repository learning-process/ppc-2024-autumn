// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/kondratev_ya_contrast_adjustment/include/ops_seq.hpp"

namespace kondratev_ya_contrast_adjustment_seq {
std::vector<kondratev_ya_contrast_adjustment_seq::Pixel> genRandomData(uint32_t size) {
  std::vector<kondratev_ya_contrast_adjustment_seq::Pixel> buff(size);
  std::random_device dev;
  std::mt19937 gen(dev());
  for (uint32_t i = 0; i < size; i++) {
    buff[i] = gen() % 256;
  }
  return buff;
}
}  // namespace kondratev_ya_contrast_adjustment_seq

TEST(kondratev_ya_contrast_adjustment_seq, test_pipeline_run) {
  const int size = 25'000'000;

  std::vector<kondratev_ya_contrast_adjustment_seq::Pixel> input;
  std::vector<kondratev_ya_contrast_adjustment_seq::Pixel> res;

  input = kondratev_ya_contrast_adjustment_seq::genRandomData(size);
  res.resize(input.size());

  auto contrast = std::make_shared<double>(1.25);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(contrast.get()));
  taskDataSeq->inputs_count.emplace_back(input.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  auto testTaskSequential = std::make_shared<kondratev_ya_contrast_adjustment_seq::TestTaskSequential>(taskDataSeq);

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
}

TEST(kondratev_ya_contrast_adjustment_seq, test_task_run) {
  const int size = 25'000'000;

  std::vector<kondratev_ya_contrast_adjustment_seq::Pixel> input;
  std::vector<kondratev_ya_contrast_adjustment_seq::Pixel> res;

  input = kondratev_ya_contrast_adjustment_seq::genRandomData(size);
  res.resize(input.size());

  auto contrast = std::make_shared<double>(1.25);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(contrast.get()));
  taskDataSeq->inputs_count.emplace_back(input.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  // Create Task
  auto testTaskSequential = std::make_shared<kondratev_ya_contrast_adjustment_seq::TestTaskSequential>(taskDataSeq);

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
}
