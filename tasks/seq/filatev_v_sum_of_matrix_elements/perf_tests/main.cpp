// filatev_v_sum_of_matrix_elements_seq
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/filatev_v_sum_of_matrix_elements/include/ops_seq.hpp"

TEST(filatev_v_sum_of_matrix_elements, test_pipeline_run) {
  const int count = 10000;

  // Create data
  std::vector<std::vector<int>> in(count, std::vector<int>(count, 1));
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (int i = 0; i < count; i++) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }
  taskDataSeq->inputs_count.emplace_back(count);
  taskDataSeq->inputs_count.emplace_back(count);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  auto sumMatrix = std::make_shared<filatev_v_sum_of_matrix_elements_seq::SumMatrix>(taskDataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(sumMatrix);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(100000000, out[0]);
}

TEST(filatev_v_sum_of_matrix_elements, test_task_run) {
  const int count = 10000;

  // Create data
  std::vector<std::vector<int>> in(count, std::vector<int>(count, 1));
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (int i = 0; i < count; i++) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
  }
  taskDataSeq->inputs_count.emplace_back(count);
  taskDataSeq->inputs_count.emplace_back(count);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  auto sumMatrix = std::make_shared<filatev_v_sum_of_matrix_elements_seq::SumMatrix>(taskDataSeq);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(sumMatrix);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(100000000, out[0]);
}
