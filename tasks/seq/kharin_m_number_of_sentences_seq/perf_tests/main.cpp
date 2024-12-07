#include <gtest/gtest.h>

#include <chrono>
#include <string>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/kharin_m_number_of_sentences_seq/include/ops_seq.hpp"

TEST(seq_kharin_m_sentence_count_perf_test, test_pipeline_run) {
  std::string input_text;
  std::vector<int> sentence_count(1, 0);
  input_text = "This is a long text with many sentences. ";
  for (int i = 0; i < 10000000; i++) {
    input_text += "Sentence " + std::to_string(i + 1) + ". ";
  }
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input_text.c_str())));
  taskDataSeq->inputs_count.emplace_back(input_text.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(sentence_count.data()));
  taskDataSeq->outputs_count.emplace_back(sentence_count.size());

  auto testSeqTask = std::make_shared<kharin_m_number_of_sentences_seq::CountSentencesSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;  // Конвертируем в секунды
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testSeqTask);

  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(10000001, sentence_count[0]);
}

TEST(seq_kharin_m_sentence_count_perf_test, test_task_run) {
  std::string input_text;
  std::vector<int> sentence_count(1, 0);

  input_text = "This is a long text with many sentences. ";
  for (int i = 0; i < 10000000; i++) {
    input_text += "Sentence " + std::to_string(i + 1) + ". ";
  }
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<char*>(input_text.c_str())));
  taskDataSeq->inputs_count.emplace_back(input_text.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(sentence_count.data()));
  taskDataSeq->outputs_count.emplace_back(sentence_count.size());

  auto testSeqTask = std::make_shared<kharin_m_number_of_sentences_seq::CountSentencesSequential>(taskDataSeq);

  // Create Perf attributesы
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;  // Конвертируем в секунды
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testSeqTask);

  perfAnalyzer->task_run(perfAttr, perfResults);

  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(10000001, sentence_count[0]);
}
