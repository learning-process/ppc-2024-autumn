#include <gtest/gtest.h>

#include <chrono>
#include <vector>

#include "../include/ops_seq.hpp"
#include "core/perf/include/perf.hpp"

template <typename DataType>
static std::vector<DataType> generateRandomValues(int size) {
  std::vector<DataType> vec(size);
  for (int i = 0; i < size; ++i) {
    vec[i] = static_cast<DataType>(rand() % 100);
  }
  return vec;
}

TEST(moiseev_a_radix_merge_seq_perf, test_pipeline_run) {
  using DataType = int32_t;
  const size_t size = 1000000;

  std::vector<DataType> input = generateRandomValues<DataType>(size);
  std::vector<DataType> output(size, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.emplace_back(size);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskData->outputs_count.emplace_back(size);

  auto testTask = std::make_shared<moiseev_a_radix_merge_seq::TestSEQTaskSequential>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(output.size(), size);
}

TEST(moiseev_a_radix_merge_seq_perf, test_task_run) {
  using DataType = int32_t;
  const size_t size = 1000000;

  std::vector<DataType> input = generateRandomValues<DataType>(size);
  std::vector<DataType> output(size, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.emplace_back(size);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskData->outputs_count.emplace_back(size);

  auto testTask = std::make_shared<moiseev_a_radix_merge_seq::TestSEQTaskSequential>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->task_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(output.size(), size);
}