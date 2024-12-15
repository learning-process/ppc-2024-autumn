#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/korovin_n_matrix_multiple_cannon/include/ops_seq.hpp"

TEST(korovin_n_matrix_multiple_cannon_seq, test_task_run) {
  int numRowsA = 512;
  int numColsA_RowsB = 512;
  int numColsB = 512;

  std::vector<double> A(numRowsA * numColsA_RowsB, 0.0);
  for (int i = 0; i < numRowsA * numColsA_RowsB; i++) {
    A[i] = i % 100 + 1;
  }

  std::vector<double> B(numColsA_RowsB * numColsB, 0.0);
  for (int i = 0; i < numColsA_RowsB; i++) {
    for (int j = 0; j < numColsB; j++) {
      B[i * numColsB + j] = (i + j) % 50 + 1;
    }
  }

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(numRowsA);
  taskData->inputs_count.emplace_back(numColsA_RowsB);
  taskData->inputs_count.emplace_back(numColsB);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));

  std::vector<double> C(numRowsA * numColsB);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));

  auto testTask = std::make_shared<korovin_n_matrix_multiple_cannon_seq::TestTaskSequential>(taskData);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(korovin_n_matrix_multiple_cannon_seq, test_pipeline_run) {
  int numRowsA = 512;
  int numColsA_RowsB = 512;
  int numColsB = 512;

  std::vector<double> A(numRowsA * numColsA_RowsB, 0.0);
  for (int i = 0; i < numRowsA * numColsA_RowsB; i++) {
    A[i] = i % 100 + 1;
  }

  std::vector<double> B(numColsA_RowsB * numColsB, 0.0);
  for (int i = 0; i < numColsA_RowsB; i++) {
    for (int j = 0; j < numColsB; j++) {
      B[i * numColsB + j] = (i + j) % 50 + 1;
    }
  }

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(numRowsA);
  taskData->inputs_count.emplace_back(numColsA_RowsB);
  taskData->inputs_count.emplace_back(numColsB);
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));

  std::vector<double> C(numRowsA * numColsB);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));

  auto testTask = std::make_shared<korovin_n_matrix_multiple_cannon_seq::TestTaskSequential>(taskData);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}
