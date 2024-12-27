// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/prokhorov_n_global_search_algorithm_strongin/include/ops_seq.hpp"

TEST(prokhorov_n_global_search_algorithm_strongin_seq_perf_test, test_pipeline_run) {
  std::vector<double> in_a = {-10.0};
  std::vector<double> in_b = {10.0};
  std::vector<double> in_epsilon = {0.001};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_a.data()));
  taskDataSeq->inputs_count.emplace_back(in_a.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_b.data()));
  taskDataSeq->inputs_count.emplace_back(in_b.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_epsilon.data()));
  taskDataSeq->inputs_count.emplace_back(in_epsilon.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential>(taskDataSeq);

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

  EXPECT_NEAR(out[0], 0.0, 0.001);
}

TEST(prokhorov_n_global_search_algorithm_strongin_seq_perf_test, test_task_run) {
  std::vector<double> in_a = {-10.0};
  std::vector<double> in_b = {10.0};
  std::vector<double> in_epsilon = {0.001};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_a.data()));
  taskDataSeq->inputs_count.emplace_back(in_a.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_b.data()));
  taskDataSeq->inputs_count.emplace_back(in_b.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_epsilon.data()));
  taskDataSeq->inputs_count.emplace_back(in_epsilon.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential>(taskDataSeq);

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

  EXPECT_NEAR(out[0], 0.0, 0.001);
}
