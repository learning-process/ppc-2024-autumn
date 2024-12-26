#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned/include/ops_seq.hpp"

TEST(sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_seq, test_pipeline_run) {
  // Create data
  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_result;

  int num_rows = 512;
  int num_cols = 512;

  global_A.resize(num_rows * num_cols, 0);

  // Generate random vector
  global_B.resize(num_cols * num_rows, 0);
  global_result.resize(512 * 512, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_A.data()));
  taskDataSeq->inputs_count.emplace_back(num_rows);
  taskDataSeq->inputs_count.emplace_back(num_cols);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_B.data()));
  taskDataSeq->inputs_count.emplace_back(num_rows);
  taskDataSeq->inputs_count.emplace_back(num_cols);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  taskDataSeq->outputs_count.emplace_back(global_result.size());

  // Create Task
  auto testTaskSequential =
      std::make_shared<sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_seq::TestTaskSequential>(
          taskDataSeq);

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
}

TEST(sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_seq, test_task_run) {
  // Create data
  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_result;

  int num_rows = 512;
  int num_cols = 512;

  global_A.resize(num_rows * num_cols, 0);

  // Generate random vector
  global_B.resize(num_cols * num_rows, 0);
  global_result.resize(512 * 512, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_A.data()));
  taskDataSeq->inputs_count.emplace_back(num_rows);
  taskDataSeq->inputs_count.emplace_back(num_cols);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_B.data()));
  taskDataSeq->inputs_count.emplace_back(num_rows);
  taskDataSeq->inputs_count.emplace_back(num_cols);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
  taskDataSeq->outputs_count.emplace_back(global_result.size());
  // Create Task
  auto testTaskSequential =
      std::make_shared<sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_seq::TestTaskSequential>(
          taskDataSeq);

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
}
