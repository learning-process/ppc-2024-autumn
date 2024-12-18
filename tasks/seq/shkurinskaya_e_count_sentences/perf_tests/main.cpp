#include <gtest/gtest.h>

#include <chrono>

#include "core/perf/include/perf.hpp"
#include "seq/shkurinskaya_e_count_sentences/include/ops_seq.hpp"

TEST(shkurinskaya_e_count_sentences, test_pipeline_run) {
  std::string input_text = "This is another test. I love testing stuff! Don't you like it too?";

  std::vector<int> out(1, 0);
  for (int i = 0; i < 999999; i++) {
    input_text += "Let's do it " + std::to_string(i + 1) + " time!";
  }
  std::vector<std::string> in(1, input_text);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<shkurinskaya_e_count_sentences::TestTaskSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(1000002, out[0]);
}

TEST(shkurinskaya_e_count_sentences, test_task_run) {
  std::string input_text = "This is another test. I love testing stuff! Don't you like it too?";

  std::vector<int> out(1, 0);
  for (int i = 0; i < 999999; i++) {
    input_text += "Let's do it " + std::to_string(i + 1) + " time!";
  }
  std::vector<std::string> in(1, input_text);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<shkurinskaya_e_count_sentences::TestTaskSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(1000002, out[0]);
}