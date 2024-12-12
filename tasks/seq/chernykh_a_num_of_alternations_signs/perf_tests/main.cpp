#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/chernykh_a_num_of_alternations_signs/include/ops_seq.hpp"

TEST(chernykh_a_num_of_alternations_signs_seq, test_pipeline_run) {
  // Create data
  auto input = std::vector<int>(10'000'000, 0);
  auto output = std::vector<int>(1, 0);
  auto want = 0;

  // Create TaskData
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  // Create Task
  auto task = std::make_shared<chernykh_a_num_of_alternations_signs_seq::Task>(task_data);

  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  // Create PerfAttributes
  auto perf_attributes = std::make_shared<ppc::core::PerfAttr>();
  perf_attributes->num_running = 10;
  auto start = std::chrono::high_resolution_clock::now();
  perf_attributes->current_timer = [&] {
    auto current = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current - start).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create PerfResults
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);

  perf_analyzer->pipeline_run(perf_attributes, perf_results);
  ppc::core::Perf::print_perf_statistic(perf_results);
  ASSERT_EQ(want, output[0]);
}

TEST(chernykh_a_num_of_alternations_signs_seq, test_task_run) {
  // Create data
  auto input = std::vector<int>(10'000'000, 0);
  auto output = std::vector<int>(1, 0);
  auto want = 0;

  // Create TaskData
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  // Create Task
  auto task = std::make_shared<chernykh_a_num_of_alternations_signs_seq::Task>(task_data);

  ASSERT_TRUE(task->validation());
  ASSERT_TRUE(task->pre_processing());
  ASSERT_TRUE(task->run());
  ASSERT_TRUE(task->post_processing());

  // Create PerfAttributes
  auto perf_attributes = std::make_shared<ppc::core::PerfAttr>();
  perf_attributes->num_running = 10;
  auto start = std::chrono::high_resolution_clock::now();
  perf_attributes->current_timer = [&] {
    auto current = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current - start).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create PerfResults
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);

  perf_analyzer->task_run(perf_attributes, perf_results);
  ppc::core::Perf::print_perf_statistic(perf_results);
  ASSERT_EQ(want, output[0]);
}