// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/makhov_m_num_of_diff_elements_in_two_str/include/ops_seq.hpp"

TEST(sequential_makhov_m_num_of_diff_elements_in_two_str_perf_test, test_pipeline_run) {
  std::string str1;
  std::string str2;
  std::random_device dev;
  std::mt19937 gen(dev());
  const size_t size = 10000000;
  char min = '0';
  char max = '9';

  // Create data
  for (size_t i = 0; i < size; i++) {
    str1 += (char)(min + gen() % (max - min + 1));
    str2 += (char)(min + gen() % (max - min + 1));
  }
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str2.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str1.data()));
  taskDataSeq->inputs_count.emplace_back(str2.size());
  taskDataSeq->inputs_count.emplace_back(str1.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential =
      std::make_shared<makhov_m_num_of_diff_elements_in_two_str_seq::TestTaskSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
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
}

TEST(sequential_makhov_m_num_of_diff_elements_in_two_str_perf_test, test_task_run) {
  std::string str1;
  std::string str2;
  std::random_device dev;
  std::mt19937 gen(dev());
  const size_t size = 10000000;
  char min = '0';
  char max = '9';

  // Create data
  for (size_t i = 0; i < size; i++) {
    str1 += (char)(min + gen() % (max - min + 1));
    str2 += (char)(min + gen() % (max - min + 1));
  }
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str2.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str1.data()));
  taskDataSeq->inputs_count.emplace_back(str2.size());
  taskDataSeq->inputs_count.emplace_back(str1.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential =
      std::make_shared<makhov_m_num_of_diff_elements_in_two_str_seq::TestTaskSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
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
}
