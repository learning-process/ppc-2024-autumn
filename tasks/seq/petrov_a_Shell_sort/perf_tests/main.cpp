#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/petrov_a_Shell_sort/include/ops_seq.hpp"

TEST(petrov_a_Shell_sort_seq, test_pipeline_run) {
  int vector_size = 6000000;
  std::vector<int> data(vector_size);
  int current = vector_size;
  std::generate(data.begin(), data.end(), [&current]() { return current--; });
  std::vector<int> result_data(vector_size);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
  taskData->inputs_count.emplace_back(data.size());

  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_data.data()));
  taskData->outputs_count.emplace_back(result_data.size());

  auto taskSequential = std::make_shared<petrov_a_Shell_sort_seq::TestTaskSequential>(taskData);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  auto start_time = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&start_time] {
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - start_time;
    return elapsed.count();
  };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  std::sort(data.begin(), data.end());

  EXPECT_EQ(data, result_data);
}

TEST(petrov_a_Shell_sort_seq, test_task_run) {
  int vector_size = 6000000;
  std::vector<int> data(vector_size);
  std::generate(data.begin(), data.end(), []() { return (rand() % 1000) - 500; });
  std::vector<int> result_data(vector_size);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
  taskData->inputs_count.emplace_back(data.size());

  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_data.data()));
  taskData->outputs_count.emplace_back(result_data.size());

  auto taskSequential = std::make_shared<petrov_a_Shell_sort_seq::TestTaskSequential>(taskData);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  auto start_time = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&start_time] {
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - start_time;
    return elapsed.count();
  };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  std::sort(data.begin(), data.end());
  EXPECT_EQ(data, result_data);
}
