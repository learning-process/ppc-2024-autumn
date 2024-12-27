#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/mironov_a_quick_sort/include/ops_mpi.hpp"

TEST(mironov_a_quick_sort_seq, test_pipeline_run) {
  const int count = 10000000;

  // Create data
  std::vector<int> in(count);
  std::vector<int> out(count);
  std::vector<int> gold(count);

  for (int i = 0; i < count; ++i) {
    in[i] = count - i * 10;
  }
  gold = in;
  std::sort(gold.begin(), gold.end());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> seqData = std::make_shared<ppc::core::TaskData>();
  seqData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  seqData->inputs_count.emplace_back(in.size());
  seqData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  seqData->outputs_count.emplace_back(out.size());

  // Create Task
  auto seqTask = std::make_shared<mironov_a_quick_sort_seq::QuickSortSequential>(seqData);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(seqTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(gold, out);
}

TEST(mironov_a_quick_sort_seq, test_task_run) {
  const int count = 10000000;

  // Create data
  std::vector<int> in(count);
  std::vector<int> out(count);
  std::vector<int> gold(count);
  for (int i = 0; i < count; ++i) {
    in[i] = count - i * 10;
  }
  gold = in;
  std::sort(gold.begin(), gold.end());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> seqData = std::make_shared<ppc::core::TaskData>();
  seqData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  seqData->inputs_count.emplace_back(in.size());
  seqData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  seqData->outputs_count.emplace_back(out.size());

  // Create Task
  auto seqTask = std::make_shared<mironov_a_quick_sort_seq::QuickSortSequential>(seqData);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(seqTask);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(gold, out);
}
