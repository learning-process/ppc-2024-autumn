#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/milovankin_m_sum_of_vector_elements/include/ops_seq.hpp"

TEST(milovankin_m_sum_of_vector_elements_seq, test_pipeline_run) {
  // Create data
  const int32_t vec_size = 50'000'000;
  std::vector<int32_t> input_data(vec_size, 1);
  auto expected_sum = static_cast<int64_t>(vec_size);
  int64_t actual_sum = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskDataSeq->inputs_count.emplace_back(input_data.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&actual_sum));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  auto vectorSumSequential = std::make_shared<milovankin_m_sum_of_vector_elements_seq::VectorSumSeq>(taskDataSeq);

  // Set up Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Initialize perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer and run
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(vectorSumSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(expected_sum, actual_sum);
}

TEST(milovankin_m_sum_of_vector_elements_seq, test_task_run) {
  const int32_t count = 50'000'000;
  std::vector<int32_t> input_data(count, 1);
  auto expected_sum = static_cast<int64_t>(count);
  int64_t actual_sum = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskDataSeq->inputs_count.emplace_back(input_data.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&actual_sum));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  auto vectorSumSequential = std::make_shared<milovankin_m_sum_of_vector_elements_seq::VectorSumSeq>(taskDataSeq);

  // Set up Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Initialize perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer and run
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(vectorSumSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(expected_sum, actual_sum);
}
