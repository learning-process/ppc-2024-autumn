#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/sarafanov_m_quick_sort_batcher_merge/include/ops_seq.hpp"

TEST(sarafanov_m_quick_sort_batcher_merge_seq, pipeline_run) {
  int vector_size = 524288;
  std::vector<int> data(vector_size);
  int current = vector_size;
  std::generate(data.begin(), data.end(), [&current]() { return current--; });
  std::vector<int> result_data(vector_size);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
  taskDataSeq->inputs_count.emplace_back(data.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_data.data()));
  taskDataSeq->outputs_count.emplace_back(result_data.size());

  auto taskSequential = std::make_shared<sarafanov_m_quick_sort_batcher_merge_seq::QuicksortBatcherMerge>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  auto start_time = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&start_time] {
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - start_time;
    return elapsed.count();
  };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  std::sort(data.begin(), data.end());
  EXPECT_EQ(data, result_data);
}

TEST(sarafanov_m_quick_sort_batcher_merge_seq, task_run) {
  int vector_size = 524288;
  std::vector<int> data(vector_size);
  int current = vector_size;
  std::generate(data.begin(), data.end(), [&current]() { return current--; });
  std::vector<int> result_data(vector_size);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
  taskDataSeq->inputs_count.emplace_back(data.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_data.data()));
  taskDataSeq->outputs_count.emplace_back(result_data.size());

  auto taskSequential = std::make_shared<sarafanov_m_quick_sort_batcher_merge_seq::QuicksortBatcherMerge>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  auto start_time = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&start_time] {
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - start_time;
    return elapsed.count();
  };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  std::sort(data.begin(), data.end());
  EXPECT_EQ(data, result_data);
}
