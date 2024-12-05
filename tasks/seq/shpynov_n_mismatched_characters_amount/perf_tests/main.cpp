#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/shpynov_n_mismatched_characters_amount/include/mismatched_numbers.hpp"

TEST(shpynov_n_amount_of_mismatched_numbers_seq_perf_test, test_pipeline_run) {
  // Create data

  std::string str1, str2 = "";
  std::vector<std::string> v1;

  std::vector<int> out(1, 0);

  std::string S = "qwerty", S1 = "qwertY";

  for (int i = 0; i < 100000; i++) {
    str1 += S;
    str2 += S1;
  }
  v1.push_back(str1);
  v1.push_back(str2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[0].data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[1].data()));
  taskDataSeq->inputs_count.emplace_back(v1.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential =
      std::make_shared<shpynov_n_amount_of_mismatched_numbers_seq::TestTaskSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(100000, out[0]);
};

TEST(shpynov_n_amount_of_mismatched_numbers_seq_perf_test, test_task_run) {
  // Create data

  std::string str1, str2 = "";
  std::vector<std::string> v1;

  std::vector<int> out(1, 0);

  std::string S = "qwerty", S1 = "qwertY";

  for (int i = 0; i < 100000; i++) {
    str1 += S;
    str2 += S1;
  }
  v1.push_back(str1);
  v1.push_back(str2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[0].data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[1].data()));
  taskDataSeq->inputs_count.emplace_back(v1.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential =
      std::make_shared<shpynov_n_amount_of_mismatched_numbers_seq::TestTaskSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(100000, out[0]);
}
