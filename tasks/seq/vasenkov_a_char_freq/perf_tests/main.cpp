#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/vasenkov_a_char_freq/include/ops_seq.hpp"

TEST(vasenkov_a_char_frequency_seq, test_pipeline_run) {
  std::string input_str(100000, 'a');
  char target_char = 'a';
  int expected_frequency = 100000;

  std::vector<std::string> in_str(1, input_str);
  std::vector<char> in_char(1, target_char);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_str.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_char.data()));
  taskDataSeq->inputs_count.emplace_back(in_str.size());
  taskDataSeq->inputs_count.emplace_back(in_char.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto charFrequencyTask = std::make_shared<vasenkov_a_char_frequency_seq::CharFrequencyTaskSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(charFrequencyTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(expected_frequency, out[0]);
}

TEST(vasenkov_a_char_frequency_seq, test_task_run) {
  std::string input_str(100000, 'a');
  char target_char = 'a';
  int expected_frequency = 100000;

  std::vector<std::string> in_str(1, input_str);
  std::vector<char> in_char(1, target_char);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_str.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_char.data()));
  taskDataSeq->inputs_count.emplace_back(in_str.size());
  taskDataSeq->inputs_count.emplace_back(in_char.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto charFrequencyTask = std::make_shared<vasenkov_a_char_frequency_seq::CharFrequencyTaskSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(charFrequencyTask);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(expected_frequency, out[0]);
}