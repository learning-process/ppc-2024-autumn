#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/strakhov_a_str_char_freq/include/ops_seq.hpp"

TEST(strakhov_a_str_char_freq_seq, test_pipeline_run) {
  std::string input(123, '1');
  char target = '1';
  int expected_res = 1;
  std::vector<std::string> input_vec(1, input);
  std::vector<char> target_vec(1, target);
  std::vector<int> out(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(target_vec.data()));
  taskDataSeq->inputs_count.emplace_back(input_vec.size());
  taskDataSeq->inputs_count.emplace_back(target_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  auto charFrequencyTask =
      std::make_shared<strakhov_a_str_char_freq_seq::TaskStringCharactersFrequencySequential>(taskDataSeq);
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
  ASSERT_EQ(expected_res, out[0]);
}

TEST(strakhov_a_str_char_freq_seq, test_task_run) {
  std::string input(1234567, '1');
  char target = '1';
  int expected_res = 1;
  std::vector<std::string> input_vec(1, input);
  std::vector<char> target_vec(1, target);
  std::vector<int> out(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(target_vec.data()));
  taskDataSeq->inputs_count.emplace_back(input_vec.size());
  taskDataSeq->inputs_count.emplace_back(target_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  auto charFrequencyTask =
      std::make_shared<strakhov_a_str_char_freq_seq::TaskStringCharactersFrequencySequential>(taskDataSeq);
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
  ASSERT_EQ(expected_res, out[0]);
}