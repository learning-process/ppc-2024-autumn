// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/ermilova_d_Shell_sort_simple_merge/include/ops_seq.hpp"

static std::vector<int> getRandomVector(int size, int upper_border, int lower_border) {
  std::random_device dev;
  std::mt19937 gen(dev());
  if (size <= 0) throw "Incorrect size";
  std::vector<int> vec(size);
  for (int i = 0; i < size; i++) {
    vec[i] = lower_border + gen() % (upper_border - lower_border + 1);
  }
  return vec;
}

TEST(ermilova_d_Shell_sort_simple_merge_seq, test_pipeline_run) {
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;

  const int size = 1000;
  bool is_resersed = false;

  std::vector<int> input = getRandomVector(size, upper_border_test, lower_border_test);
  std::vector<int> output(input.size(), 0);

  std::vector<int> sort_ref = input;
  std::sort(sort_ref.begin(), sort_ref.end());

  // Create  TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());

  // Create Task
  auto testTaskSequential = std::make_shared<ermilova_d_Shell_sort_simple_merge_seq::TestTaskSequential>(taskDataSeq);

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
  ASSERT_EQ(sort_ref, output);
}

TEST(ermilova_d_Shell_sort_simple_merge_seq, test_task_run) {
  const int upper_border_test = 1000;
  const int lower_border_test = -1000;

  const int size = 1000;
  bool is_resersed = false;

  std::vector<int> input = getRandomVector(size, upper_border_test, lower_border_test);
  std::vector<int> output(input.size(), 0);

  std::vector<int> sort_ref = input;
  std::sort(sort_ref.begin(), sort_ref.end());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&is_resersed));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());

  // Create Task
  auto testTaskSequential = std::make_shared<ermilova_d_Shell_sort_simple_merge_seq::TestTaskSequential>(taskDataSeq);

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
  ASSERT_EQ(sort_ref, output);
}
