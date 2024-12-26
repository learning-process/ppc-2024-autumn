#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/lopatin_i_quick_batcher_mergesort/include/quickBatcherMergesortHeaderSeq.hpp"

namespace lopatin_i_quick_bathcer_sort_seq {

std::vector<int> generateArray(int size, int minValue, int maxValue) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(minValue, maxValue);

  std::vector<int> outputArray(size);
  for (int i = 0; i < size; i++) {
    outputArray[i] = dist(gen);
  }
  return outputArray;
}

}  // namespace lopatin_i_quick_bathcer_sort_seq

std::vector<int> testArray = lopatin_i_quick_bathcer_sort_seq::generateArray(48000, -999, 999);

TEST(lopatin_i_quick_batcher_mergesort_seq, test_pipeline_run) {
  std::vector<int> inputArray = testArray;
  std::vector<int> resultArray(48000, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  taskData->inputs_count.emplace_back(inputArray.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultArray.data()));
  taskData->outputs_count.emplace_back(resultArray.size());

  auto testTask = std::make_shared<lopatin_i_quick_batcher_mergesort_seq::TestTaskSequential>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(lopatin_i_quick_batcher_mergesort_seq, test_task_run) {
  std::vector<int> inputArray = testArray;
  std::vector<int> resultArray(48000, 0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputArray.data()));
  taskData->inputs_count.emplace_back(inputArray.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultArray.data()));
  taskData->outputs_count.emplace_back(resultArray.size());

  auto testTask = std::make_shared<lopatin_i_quick_batcher_mergesort_seq::TestTaskSequential>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}