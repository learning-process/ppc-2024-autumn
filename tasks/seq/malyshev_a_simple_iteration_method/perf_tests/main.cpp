// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/malyshev_a_simple_iteration_method/include/ops_seq.hpp"

namespace malyshev_a_simple_iteration_method_seq {
void getRandomData(uint32_t n, std::vector<double> &A, std::vector<double> &B) {
  const auto random_double = [&](double lower_bound, double upper_bound) {
    return lower_bound + (upper_bound - lower_bound) * (std::rand() % RAND_MAX) / RAND_MAX;
  };

  std::srand(std::time(nullptr));

  std::vector<double> X(n);
  for (uint32_t i = 0; i < n; i++) {
    X[i] = random_double(50, 100);
    if (std::rand() % 2 == 0) X[i] *= -1;
  }

  A.resize(n * n);
  B.resize(n);

  double sum_by_row_for_C;
  double sum_by_row_for_B;
  for (uint32_t i = 0; i < n; i++) {
    A[i * n + i] = random_double(50, 100);
    if (std::rand() % 2 == 0) A[i * n + i] *= -1;

    sum_by_row_for_C = 0.01;
    sum_by_row_for_B = A[i * n + i] * X[i];

    for (uint32_t j = 0; j < n; j++) {
      if (i != j) {
        A[i * n + j] =
            random_double(std::abs(A[i * n + i]) * (-1 + sum_by_row_for_C + std::numeric_limits<double>::epsilon()),
                          std::abs(A[i * n + i]) * (1 - sum_by_row_for_C - std::numeric_limits<double>::epsilon()));

        sum_by_row_for_C += std::abs(A[i * n + j] / A[i * n + i]);
        sum_by_row_for_B += A[i * n + j] * X[j];
      }
    }

    B[i] = sum_by_row_for_B;
  }
}
}  // namespace malyshev_a_simple_iteration_method_seq

TEST(malyshev_a_simple_iteration_method_seq, test_pipeline_run) {
  const int size = 500;
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

TEST(malyshev_a_simple_iteration_method_seq, test_task_run) {
  const int size = 500;
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