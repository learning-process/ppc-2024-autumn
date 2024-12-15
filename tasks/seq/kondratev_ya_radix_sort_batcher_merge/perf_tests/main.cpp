// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/kondratev_ya_radix_sort_batcher_merge/include/ops_seq.hpp"

namespace kondratev_ya_radix_sort_batcher_merge_seq {
std::vector<double> getRandomVector(uint32_t size) {
  std::srand(std::time(nullptr));
  std::vector<double> vec(size);

  double lower_bound = -10000;
  double upper_bound = 10000;
  for (uint32_t i = 0; i < size; i++) {
    vec[i] = lower_bound + std::rand() / (double)RAND_MAX * (upper_bound - lower_bound);
  }
  return vec;
}
}  // namespace kondratev_ya_radix_sort_batcher_merge_seq

TEST(kondratev_ya_radix_sort_batcher_merge_seq, test_pipeline_run) {
  std::vector<double> in;
  std::vector<double> out;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  uint32_t size = 5000000;
  in = kondratev_ya_radix_sort_batcher_merge_seq::getRandomVector(size);
  out.resize(size);

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<kondratev_ya_radix_sort_batcher_merge_seq::TestTaskSequential>(taskData);

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

TEST(kondratev_ya_radix_sort_batcher_merge_seq, test_task_run) {
  std::vector<double> in;
  std::vector<double> out;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  uint32_t size = 5000000;
  in = kondratev_ya_radix_sort_batcher_merge_seq::getRandomVector(size);
  out.resize(size);

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<kondratev_ya_radix_sort_batcher_merge_seq::TestTaskSequential>(taskData);

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
