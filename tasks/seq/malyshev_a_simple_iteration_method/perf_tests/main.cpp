// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/malyshev_a_simple_iteration_method/include/ops_seq.hpp"

TEST(malyshev_a_simple_iteration_method, test_pipeline_run) {
  const int size = 300;
  std::vector<double> A;
  std::vector<double> B;
  malyshev_a_simple_iteration_method_seq::getRandomData(size, A, B);

  std::vector<double> X(size, 0);
  std::vector<double> X0(size, 0);
  double eps = 1e-4;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&eps));
  taskDataSeq->inputs_count.emplace_back(X.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
  taskDataSeq->outputs_count.emplace_back(X.size());

  // Create Task
  auto testTaskSequential = std::make_shared<malyshev_a_simple_iteration_method_seq::TestTaskSequential>(taskDataSeq);

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

TEST(malyshev_a_simple_iteration_method, test_task_run) {
  const int size = 300;
  std::vector<double> A;
  std::vector<double> B;
  malyshev_a_simple_iteration_method_seq::getRandomData(size, A, B);

  std::vector<double> X(size, 0);
  std::vector<double> X0(size, 0);
  double eps = 1e-4;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&eps));
  taskDataSeq->inputs_count.emplace_back(X.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
  taskDataSeq->outputs_count.emplace_back(X.size());

  // Create Task
  auto testTaskSequential = std::make_shared<malyshev_a_simple_iteration_method_seq::TestTaskSequential>(taskDataSeq);

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