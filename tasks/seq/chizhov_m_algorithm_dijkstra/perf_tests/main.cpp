// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/chizhov_m_algorithm_dijkstra/include/ops_seq.hpp"

TEST(chizhov_m_dijkstra_seq_perf_test, test_pipeline_run) {
  // Create data
  int count_size_vector = 5000;
  int st = 0;
  std::vector<int> global_matrix(count_size_vector * count_size_vector, 1);
  std::vector<int32_t> global_path(count_size_vector, 1);

  for (int i = 0; i < count_size_vector; i++) {
    global_matrix[i * count_size_vector + i] = 0;
  }
  global_path[0] = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(global_matrix.size());
  taskDataSeq->inputs_count.emplace_back(count_size_vector);
  taskDataSeq->inputs_count.emplace_back(st);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_path.data()));
  taskDataSeq->outputs_count.emplace_back(global_path.size());

  // Create Task
  auto testTaskSequential = std::make_shared<chizhov_m_dijkstra_seq::TestTaskSequential>(taskDataSeq);

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
  ASSERT_EQ(1, global_path[1]);
}

TEST(chizhov_m_dijkstra_seq_perf_test, test_task_run) {
  // Create data
  int count_size_vector = 5000;
  int st = 0;
  std::vector<int> global_matrix(count_size_vector * count_size_vector, 1);
  std::vector<int32_t> global_path(count_size_vector, 1);

  for (int i = 0; i < count_size_vector; i++) {
    global_matrix[i * count_size_vector + i] = 0;
  }
  global_path[0] = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(global_matrix.size());
  taskDataSeq->inputs_count.emplace_back(count_size_vector);
  taskDataSeq->inputs_count.emplace_back(st);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_path.data()));
  taskDataSeq->outputs_count.emplace_back(global_path.size());

  // Create Task
  auto testTaskSequential = std::make_shared<chizhov_m_dijkstra_seq::TestTaskSequential>(taskDataSeq);

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
  ASSERT_EQ(1, global_path[1]);
}
