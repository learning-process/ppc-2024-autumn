#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/yasakova_t_quick_sort_with_simple_merge/include/ops_seq.hpp"

TEST(yasakova_t_quick_sort_with_simple_merge_seq, ExecutesPipelineRunWithLargeDataSet) {
  const int kVectorSize = 10000;
  std::vector<int> inputData(kVectorSize);
  int currentValue = kVectorSize;
  std::generate(inputData.begin(), inputData.end(), [&currentValue]() { return currentValue--; });
  std::vector<int> outputData;

  std::shared_ptr<ppc::core::TaskData> taskDataPtr = std::make_shared<ppc::core::TaskData>();

  taskDataPtr->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
  taskDataPtr->inputs_count.emplace_back(inputData.size());

  outputData.resize(kVectorSize);
  taskDataPtr->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputData.data()));
  taskDataPtr->outputs_count.emplace_back(outputData.size());

  auto quickSortTask =
      std::make_shared<yasakova_t_quick_sort_with_simple_merge_seq::QuickSortWithMergeSeq>(taskDataPtr);
  ASSERT_TRUE(quickSortTask->validation());
  quickSortTask->pre_processing();
  quickSortTask->run();
  quickSortTask->post_processing();

  auto performanceAttributes = std::make_shared<ppc::core::PerfAttr>();
  performanceAttributes->num_running = 10;
  auto startTime = std::chrono::high_resolution_clock::now();
  performanceAttributes->current_timer = [&startTime] {
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - startTime;
    return elapsed.count();
  };
  auto performanceResults = std::make_shared<ppc::core::PerfResults>();
  auto performanceAnalyzer = std::make_shared<ppc::core::Perf>(quickSortTask);
  performanceAnalyzer->pipeline_run(performanceAttributes, performanceResults);
  ppc::core::Perf::print_perf_statistic(performanceResults);

  std::sort(inputData.begin(), inputData.end());
  EXPECT_EQ(inputData, outputData);
}

TEST(QuickSortWithMergeSeqTests, ExecutesTaskRunWithLargeDataSet) {
  const int kVectorSize = 10000;
  std::vector<int> inputData(kVectorSize);
  int currentValue = kVectorSize;
  std::generate(inputData.begin(), inputData.end(), [&currentValue]() { return currentValue--; });
  std::vector<int> outputData;

  std::shared_ptr<ppc::core::TaskData> taskDataPtr = std::make_shared<ppc::core::TaskData>();

  taskDataPtr->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
  taskDataPtr->inputs_count.emplace_back(inputData.size());

  outputData.resize(kVectorSize);
  taskDataPtr->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputData.data()));
  taskDataPtr->outputs_count.emplace_back(outputData.size());

  auto quickSortTask =
      std::make_shared<yasakova_t_quick_sort_with_simple_merge_seq::QuickSortWithMergeSeq>(taskDataPtr);
  ASSERT_TRUE(quickSortTask->validation());
  quickSortTask->pre_processing();
  quickSortTask->run();
  quickSortTask->post_processing();

  auto performanceAttributes = std::make_shared<ppc::core::PerfAttr>();
  performanceAttributes->num_running = 10;
  auto startTime = std::chrono::high_resolution_clock::now();
  performanceAttributes->current_timer = [&startTime] {
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - startTime;
    return elapsed.count();
  };
  auto performanceResults = std::make_shared<ppc::core::PerfResults>();
  auto performanceAnalyzer = std::make_shared<ppc::core::Perf>(quickSortTask);
  performanceAnalyzer->task_run(performanceAttributes, performanceResults);
  ppc::core::Perf::print_perf_statistic(performanceResults);

  std::sort(inputData.begin(), inputData.end());
  EXPECT_EQ(inputData, outputData);
}