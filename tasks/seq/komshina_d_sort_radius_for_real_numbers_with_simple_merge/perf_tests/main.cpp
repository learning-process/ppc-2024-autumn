#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_seq.hpp"

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq_perf, test_pipeline_run) {
  int count = 10;
  std::vector<double> inputData = {3.14, 2.71, 1.41, 1.73, 0.6, 1.61, 2.0, 2.236, 3.01, 4.0};
  std::vector<double> outputData(count, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
  taskDataSeq->inputs_count.emplace_back(count);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputData.data()));
  taskDataSeq->outputs_count.emplace_back(count);

  auto testTaskSequential =
      std::make_shared<komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential>(taskDataSeq);

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
  for (int i = 0; i < count; ++i) {
    ASSERT_NEAR(refData[i], outputData[i], 1e-12);
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq_perf, test_task_run) {
  int count = 10;
  std::vector<double> inputData = {3.14, 2.71, 1.41, 1.73, 0.6, 1.61, 2.0, 2.236, 3.01, 4.0};
  std::vector<double> outputData(count, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&count));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputData.data()));
  taskDataSeq->inputs_count.emplace_back(count);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(outputData.data()));
  taskDataSeq->outputs_count.emplace_back(count);

  auto testTaskSequential =
      std::make_shared<komshina_d_sort_radius_for_real_numbers_with_simple_merge_seq::TestTaskSequential>(taskDataSeq);

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
  for (int i = 0; i < count; ++i) {
    ASSERT_NEAR(refData[i], outputData[i], 1e-12);
  }
}