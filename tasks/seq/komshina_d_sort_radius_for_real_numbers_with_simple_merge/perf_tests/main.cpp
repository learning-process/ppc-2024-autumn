#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_seq.hpp"

using komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential;

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq_perf, test_pipeline_run) {
  const int count = 10;

  std::vector<double> inputData = {3.14, 2.71, 1.41, 1.73, 0.577, 1.61, 1.618, 2.236, 2.718, 3.14159};
  std::vector<double> outputData(count, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&count)));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(inputData.data())));
  taskDataSeq->inputs_count.emplace_back(inputData.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputData.data()));
  taskDataSeq->outputs_count.emplace_back(outputData.size());

  auto taskSeq = std::make_shared<TestTaskSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskSeq);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  std::vector<double> refData = inputData;
  std::sort(refData.begin(), refData.end());
  for (size_t i = 0; i < refData.size(); ++i) {
    ASSERT_NEAR(refData[i], outputData[i], 1e-12);
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq_perf, test_task_run) {
  const int count = 10;

  std::vector<double> inputData = {3.14, 2.71, 1.41, 1.73, 0.577, 1.61, 1.618, 2.236, 2.718, 3.14159};

  std::vector<double> outputData(count, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&count)));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(inputData.data())));
  taskDataSeq->inputs_count.emplace_back(inputData.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputData.data()));
  taskDataSeq->outputs_count.emplace_back(outputData.size());

  auto taskSeq = std::make_shared<TestTaskSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskSeq);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  std::vector<double> refData = inputData;
  std::sort(refData.begin(), refData.end());
  for (size_t i = 0; i < refData.size(); ++i) {
    ASSERT_NEAR(refData[i], outputData[i], 1e-12);
  }
}
