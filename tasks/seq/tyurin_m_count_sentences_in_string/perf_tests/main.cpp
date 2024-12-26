#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/tyurin_m_count_sentences_in_string/include/ops_seq.hpp"

const size_t count_strings = 10000;

TEST(tyurin_m_count_sentences_in_string_seq, test_pipeline_run) {
  std::string str = "This is the first sentence. And this is the second! Finally, the third?";
  std::string input_str;
  input_str.resize(str.size() * count_strings);
  for (size_t i = 0; i < count_strings; i++) {
    std::copy(str.begin(), str.end(), input_str.begin() + i * str.size());
  }
  int expected_sentence_count = 30000;

  std::vector<std::string> in_str(1, input_str);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_str.data()));
  taskDataSeq->inputs_count.emplace_back(in_str.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto sentenceCountTask =
      std::make_shared<tyurin_m_count_sentences_in_string_seq::SentenceCountTaskSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(sentenceCountTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(expected_sentence_count, out[0]);
}

TEST(tyurin_m_count_sentences_in_string_seq, test_task_run) {
  std::string str = "This is the first sentence. And this is the second! Finally, the third?";
  std::string input_str;
  input_str.resize(str.size() * count_strings);
  for (size_t i = 0; i < count_strings; i++) {
    std::copy(str.begin(), str.end(), input_str.begin() + i * str.size());
  }
  int expected_sentence_count = 30000;

  std::vector<std::string> in_str(1, input_str);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_str.data()));
  taskDataSeq->inputs_count.emplace_back(in_str.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto sentenceCountTask =
      std::make_shared<tyurin_m_count_sentences_in_string_seq::SentenceCountTaskSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(sentenceCountTask);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(expected_sentence_count, out[0]);
}
