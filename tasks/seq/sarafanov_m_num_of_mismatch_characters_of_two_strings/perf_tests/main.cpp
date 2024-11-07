#include <gtest/gtest.h>

#include <chrono>
#include <string>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/sarafanov_m_num_of_mismatch_characters_of_two_strings/include/ops_seq.hpp"

TEST(sarafanov_m_num_of_mismatch_characters_of_two_strings_seq, test_pipeline_run) {
  auto input_a = std::string(5000000, 'a');
  auto input_b = std::string(5000000, 'a');
  auto output = std::vector<int>(1);
  auto expected = 0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_a.data()));
  task_data->inputs_count.emplace_back(input_a.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_b.data()));
  task_data->inputs_count.emplace_back(input_b.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = std::make_shared<sarafanov_m_num_of_mismatch_characters_of_two_strings_seq::SequentialTask>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->pipeline_run(perf_attr, perf_results);

  ppc::core::Perf::print_perf_statistic(perf_results);
  ASSERT_EQ(expected, output[0]);
}

TEST(sarafanov_m_num_of_mismatch_characters_of_two_strings_seq, test_task_run) {
  auto input_a = std::string(5000000, 'a');
  auto input_b = std::string(5000000, 'a');
  auto output = std::vector<int>(1);
  auto expected = 0;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_a.data()));
  task_data->inputs_count.emplace_back(input_a.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_b.data()));
  task_data->inputs_count.emplace_back(input_b.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = std::make_shared<sarafanov_m_num_of_mismatch_characters_of_two_strings_seq::SequentialTask>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->task_run(perf_attr, perf_results);

  ppc::core::Perf::print_perf_statistic(perf_results);
  ASSERT_EQ(expected, output[0]);
}
