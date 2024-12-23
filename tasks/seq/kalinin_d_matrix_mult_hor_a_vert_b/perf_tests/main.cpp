// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/kalinin_d_matrix_mult_hor_a_vert_b/include/ops_seq.hpp"

TEST(kalinin_d_matrix_mult_hor_a_vert_b_seq, test_pipeline_run) {
  std::vector<size_t> dimensions = {100, 100, 100, 100};
  std::vector<int> matrix_a(10000, 1);
  std::vector<int> matrix_b(10000, 1);
  std::vector<int> expected_result(10000, 100);
  std::vector<int> matrix_c(10000, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(dimensions[0]);
  taskDataSeq->inputs_count.emplace_back(dimensions[1]);
  taskDataSeq->inputs_count.emplace_back(dimensions[2]);
  taskDataSeq->inputs_count.emplace_back(dimensions[3]);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(matrix_a.data())));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(matrix_b.data())));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_c.data()));
  taskDataSeq->outputs_count.emplace_back(19000);

  auto MultHorAVertBTaskSequential =
      std::make_shared<kalinin_d_matrix_mult_hor_a_vert_b_seq::MultHorAVertBTaskSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(MultHorAVertBTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  ASSERT_EQ(matrix_c, expected_result);
}

TEST(kalinin_d_matrix_mult_hor_a_vert_b_seq, test_task_run) {
  std::vector<size_t> dimensions = {100, 100, 100, 100};
  std::vector<int> matrix_a(10000, 1);
  std::vector<int> matrix_b(10000, 1);
  std::vector<int> expected_result(10000, 100);
  std::vector<int> matrix_c(10000, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.emplace_back(dimensions[0]);
  taskDataSeq->inputs_count.emplace_back(dimensions[1]);
  taskDataSeq->inputs_count.emplace_back(dimensions[2]);
  taskDataSeq->inputs_count.emplace_back(dimensions[3]);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(matrix_a.data())));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(matrix_b.data())));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_c.data()));
  taskDataSeq->outputs_count.emplace_back(matrix_c.size());

  auto MultHorAVertBTaskSequential =
      std::make_shared<kalinin_d_matrix_mult_hor_a_vert_b_seq::MultHorAVertBTaskSequential>(taskDataSeq);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(MultHorAVertBTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(matrix_c, expected_result);
}
