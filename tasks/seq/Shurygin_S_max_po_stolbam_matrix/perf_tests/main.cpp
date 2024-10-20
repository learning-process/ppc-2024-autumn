// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/Shurygin_S_max_po_stolbam_matrix/include/ops_seq.hpp"

TEST(Shurygin_S_max_po_stolbam_matrix_seq, test_pipeline_run) {
  const int rows = 5000;
  const int cols = 5000;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  auto testTaskSequential = std::make_shared<Shurygin_S_max_po_stolbam_matrix_seq::TestTaskSequential>(taskDataSeq);
  std::vector<std::vector<int>> matrix_rnd =
      Shurygin_S_max_po_stolbam_matrix_seq::TestTaskSequential::generate_random_matrix(rows, cols);
  for (auto& row : matrix_rnd) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
  }
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  std::vector<int> v_res(cols, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(v_res.data()));
  taskDataSeq->outputs_count.emplace_back(v_res.size());

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;  
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  for (int j = 0; j < cols; j++) {
    ASSERT_EQ(v_res[j], 101);  
  }
}

TEST(Shurygin_S_max_po_stolbam_matrix_seq, test_task_run) {
  const int rows = 5000;
  const int cols = 5000;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  auto testTaskSequential = std::make_shared<Shurygin_S_max_po_stolbam_matrix_seq::TestTaskSequential>(taskDataSeq);
  std::vector<std::vector<int>> matrix_rnd =
      Shurygin_S_max_po_stolbam_matrix_seq::TestTaskSequential::generate_random_matrix(rows, cols);
  for (auto& row : matrix_rnd) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
  }
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  std::vector<int> v_res(cols, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(v_res.data()));
  taskDataSeq->outputs_count.emplace_back(v_res.size());

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  for (int j = 0; j < cols; j++) {
    ASSERT_EQ(v_res[j], 101);  
  }
}
