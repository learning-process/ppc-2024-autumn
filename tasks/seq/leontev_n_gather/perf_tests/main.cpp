// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/leontev_n_gather/include/ops_seq.hpp"

TEST(leontev_n_mat_vec_seq, test_pipeline_run) {
  const int count = 3000;
  // Create data
  std::vector<int> invec(count, 1);
  std::vector<int> inmat(count * count, 1);
  std::vector<int> out(count, 0);
  const std::vector<int> expected_vec(count, count);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inmat.data()));
  taskDataSeq->inputs_count.emplace_back(inmat.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(invec.data()));
  taskDataSeq->inputs_count.emplace_back(invec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  // Create Task
  auto matVecSequential = std::make_shared<leontev_n_mat_vec_seq::MatVecSequential<int>>(taskDataSeq);
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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(matVecSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(expected_vec, out);
}

TEST(leontev_n_mat_vec_seq, test_task_run) {
  const int count = 3000;
  // Create data
  std::vector<int> invec(count, 1);
  std::vector<int> inmat(count * count, 1);
  std::vector<int> out(count, 0);
  const std::vector<int> expected_vec(count, count);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(inmat.data()));
  taskDataSeq->inputs_count.emplace_back(inmat.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(invec.data()));
  taskDataSeq->inputs_count.emplace_back(invec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  // Create Task
  auto matVecSequential = std::make_shared<leontev_n_mat_vec_seq::MatVecSequential<int>>(taskDataSeq);
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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(matVecSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(expected_vec, out);
}
