// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/varfolomeev_g_quick_sort_simple_merge/include/ops_seq.hpp"

namespace varfolomeev_g_quick_sort_simple_merge_seq {

static std::vector<int> getAntisorted_seq(int sz, int a) {  // (a, a + sz]
  if (sz <= 0) {
    return {};
  }
  std::vector<int> vec(sz);
  for (int i = a + sz, j = 0; i > a && j < sz; i--, j++) {
    vec[j] = i;
  }
  return vec;
}
}  // namespace varfolomeev_g_quick_sort_simple_merge_seq

TEST(sequential_varfolomeev_g_quick_sort_simple_merge_perf_test, test_pipeline_run) {
  int count_size_vector = 524288 * 4;

  // Create data
  std::vector<int> global_vec = varfolomeev_g_quick_sort_simple_merge_seq::getAntisorted_seq(count_size_vector, 0);
  std::vector<int> global_res(count_size_vector);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
  taskDataSeq->inputs_count.emplace_back(global_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
  taskDataSeq->outputs_count.emplace_back(global_res.size());

  // Create Task
  auto testTaskSequential =
      std::make_shared<varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential>(taskDataSeq);

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
  bool isSorted = std::is_sorted(global_res.begin(), global_res.end());
  ASSERT_TRUE(isSorted);
}

TEST(sequential_varfolomeev_g_quick_sort_simple_merge_perf_test, test_task_run) {
  int count_size_vector = 524288 * 4;

  // Create data
  std::vector<int> global_vec = varfolomeev_g_quick_sort_simple_merge_seq::getAntisorted_seq(count_size_vector, 0);
  std::vector<int> global_res(count_size_vector);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
  taskDataSeq->inputs_count.emplace_back(global_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_res.data()));
  taskDataSeq->outputs_count.emplace_back(global_res.size());

  // Create Task
  auto testTaskSequential =
      std::make_shared<varfolomeev_g_quick_sort_simple_merge_seq::TestTaskSequential>(taskDataSeq);

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
  bool isSorted = std::is_sorted(global_res.begin(), global_res.end());
  ASSERT_TRUE(isSorted);
}
