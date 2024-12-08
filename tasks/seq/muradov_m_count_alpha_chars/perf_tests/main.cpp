#include <gtest/gtest.h>

#include <random>
#include <string>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/muradov_m_count_alpha_chars/include/ops_seq.hpp"

std::string generate_large_string(size_t length) {
  std::string characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()";
  std::string result;
  result.reserve(length);

  std::default_random_engine generator;
  std::uniform_int_distribution<size_t> distribution(0, characters.size() - 1);

  for (size_t i = 0; i < length; ++i) {
    result += characters[distribution(generator)];
  }

  return result;
}

TEST(muradov_m_count_alpha_chars_seq, test_pipeline_run) {
  std::string input_str = generate_large_string(1000000);
  int expected_alpha_count = std::count_if(input_str.begin(), input_str.end(),
                                           [](char c) { return std::isalpha(static_cast<unsigned char>(c)); });

  std::vector<std::string> in_str(1, input_str);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_str.data()));
  taskDataSeq->inputs_count.emplace_back(in_str.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto alphaCharCountTask =
      std::make_shared<muradov_m_count_alpha_chars_seq::AlphaCharCountTaskSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(alphaCharCountTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(expected_alpha_count, out[0]);
}

TEST(muradov_m_count_alpha_chars_seq, test_task_run) {
  std::string input_str = generate_large_string(1000000);
  int expected_alpha_count = std::count_if(input_str.begin(), input_str.end(),
                                           [](char c) { return std::isalpha(static_cast<unsigned char>(c)); });

  std::vector<std::string> in_str(1, input_str);
  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_str.data()));
  taskDataSeq->inputs_count.emplace_back(in_str.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto alphaCharCountTask =
      std::make_shared<muradov_m_count_alpha_chars_seq::AlphaCharCountTaskSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(alphaCharCountTask);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(expected_alpha_count, out[0]);
}