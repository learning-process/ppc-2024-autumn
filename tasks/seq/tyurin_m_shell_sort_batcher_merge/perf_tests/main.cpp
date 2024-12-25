#include <gtest/gtest.h>

#include <random>

#include "core/perf/include/perf.hpp"
#include "seq/tyurin_m_shell_sort_batcher_merge/include/ops_seq.hpp"

namespace tyurin_m_shell_sort_batcher_merge_seq {

std::vector<int> random_vector(int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(-100, 100);

  std::vector<int> arr(size);
  for (size_t i = 0; i < arr.size(); i++) {
    arr[i] = dist(gen);
  }
  return arr;
}

}  // namespace tyurin_m_shell_sort_batcher_merge_seq

TEST(tyurin_m_shell_sort_batcher_merge_seq, test_pipeline_run) {
  const int n = std::pow(2, 15);
  std::vector<int> input_vec;
  std::vector<int> output_vec;

  auto task = std::make_shared<ppc::core::TaskData>();

  task->inputs_count.emplace_back(n);
  input_vec = tyurin_m_shell_sort_batcher_merge_seq::random_vector(n);
  output_vec.resize(n);

  task->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  task->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_vec.data()));

  auto test = std::make_shared<tyurin_m_shell_sort_batcher_merge_seq::ShellSortBatcherMerge>(task);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(test);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  std::vector<int> exp_result = input_vec;
  std::sort(exp_result.begin(), exp_result.end());
  EXPECT_EQ(exp_result, output_vec);
}

TEST(tyurin_m_shell_sort_batcher_merge_seq, test_task_run) {
  const int n = std::pow(2, 15);
  std::vector<int> input_vec;
  std::vector<int> output_vec;

  auto task = std::make_shared<ppc::core::TaskData>();

  task->inputs_count.emplace_back(n);
  input_vec = tyurin_m_shell_sort_batcher_merge_seq::random_vector(n);
  output_vec.resize(n);

  task->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vec.data()));
  task->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_vec.data()));

  auto test = std::make_shared<tyurin_m_shell_sort_batcher_merge_seq::ShellSortBatcherMerge>(task);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(test);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  std::vector<int> exp_result = input_vec;
  std::sort(exp_result.begin(), exp_result.end());
  EXPECT_EQ(exp_result, output_vec);
}
