// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/kurakin_m_min_values_by_rows_matrix/include/ops_seq.hpp"

TEST(kurakin_m_min_values_by_rows_matrix_seq, test_pipeline_run) {
  int count_rows = 100;
  int size_rows = 400;

  // Create data
  std::vector<int> global_mat(count_rows * size_rows, 1);
  std::vector<int32_t> seq_min_vec(count_rows, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
  taskDataSeq->inputs_count.emplace_back(global_mat.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_min_vec.data()));
  taskDataSeq->outputs_count.emplace_back(seq_min_vec.size());

  // Create Task
  auto testTaskSequential =
      std::make_shared<kurakin_m_min_values_by_rows_matrix_seq::TestTaskSequential>(taskDataSeq, count_rows, size_rows);

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
  for (unsigned i = 0; i < seq_min_vec.size(); i++) {
    // std::cout << par_min_vec[0] << " ";
    EXPECT_EQ(0, seq_min_vec[0]);
  }
}

TEST(sequential_example_perf_test, test_task_run) {
  int count_rows = 100;
  int size_rows = 400;

  // Create data
  std::vector<int> global_mat(count_rows * size_rows, 1);
  std::vector<int32_t> seq_min_vec(count_rows, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
  taskDataSeq->inputs_count.emplace_back(global_mat.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_min_vec.data()));
  taskDataSeq->outputs_count.emplace_back(seq_min_vec.size());

  // Create Task
  auto testTaskSequential =
      std::make_shared<kurakin_m_min_values_by_rows_matrix_seq::TestTaskSequential>(taskDataSeq, count_rows, size_rows);

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
  for (unsigned i = 0; i < seq_min_vec.size(); i++) {
    // std::cout << par_min_vec[0] << " ";
    EXPECT_EQ(1, seq_min_vec[0]);
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
