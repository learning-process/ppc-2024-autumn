#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/lopatin_i_count_words/include/countWordsSeqHeader.hpp"

TEST(word_count_seq, test_pipeline_run) {
  std::string input = "This is a long sentence for performance testing of the word count algorithm";
  std::vector<int> word_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(word_count.data()));
  taskData->outputs_count.emplace_back(word_count.size());

  auto testTask = std::make_shared<lopatin_i_count_words_seq::TestTaskSequential>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
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

  ASSERT_EQ(12, word_count[0]);
}

TEST(word_count_seq, test_task_run) {
  std::string input = "This is another long sentence for performance testing of the word count algorithm";
  std::vector<int> word_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  taskData->inputs_count.emplace_back(input.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(word_count.data()));
  taskData->outputs_count.emplace_back(word_count.size());

  auto testTask = std::make_shared<lopatin_i_count_words_seq::TestTaskSequential>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
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

  ASSERT_EQ(13, word_count[0]);
}