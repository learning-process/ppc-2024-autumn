#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/belov_a_gauss_seidel_iter_method/include/ops_seq.hpp"

using namespace belov_a_gauss_seidel_seq;

namespace belov_a_gauss_seidel_seq {
std::vector<double> generateDiagonallyDominantMatrix(int n) {
  std::vector<double> A_local(n * n, 0.0);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-10.0, 10.0);

  for (int i = 0; i < n; ++i) {
    double row_sum = 0.0;
    for (int j = 0; j < n; ++j) {
      if (i != j) {
        A_local[i * n + j] = dis(gen);
        row_sum += abs(A_local[i * n + j]);
      }
    }
    A_local[i * n + i] = row_sum + abs(dis(gen)) + 1.0;
  }
  return A_local;
}

std::vector<double> generateFreeMembers(int n) {
  std::vector<double> freeMembers(n, 0.0);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-10.0, 10.0);

  for (int i = 0; i < n; ++i) {
    freeMembers[i] = dis(gen);
  }
  return freeMembers;
}
}  // namespace belov_a_gauss_seidel_seq

TEST(belov_a_gauss_seidel_perf_test, test_pipeline_run) {
  // Create data

  int n = 1000;
  double epsilon = 0.001;
  std::vector<double> input_matrix = generateDiagonallyDominantMatrix(n);
  std::vector<double> freeMembersVector = generateFreeMembers(n);
  std::vector<double> solutionVector(n, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(freeMembersVector.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(freeMembersVector.size());
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(solutionVector.data()));

  // Create Task
  auto testTaskSequential = std::make_shared<belov_a_gauss_seidel_seq::GaussSeidelSequential>(taskDataSeq);

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

  ASSERT_TRUE(testTaskSequential->validation());
}

TEST(belov_a_gauss_seidel_perf_test, test_task_run) {
  // Create data
  int n = 1000;
  double epsilon = 0.001;
  std::vector<double> input_matrix = generateDiagonallyDominantMatrix(n);
  std::vector<double> freeMembersVector = generateFreeMembers(n);
  std::vector<double> solutionVector(n, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(freeMembersVector.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(freeMembersVector.size());
  taskDataSeq->inputs_count.emplace_back(input_matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(solutionVector.data()));

  // Create Task
  auto testTaskSequential = std::make_shared<belov_a_gauss_seidel_seq::GaussSeidelSequential>(taskDataSeq);

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

  ASSERT_TRUE(testTaskSequential->validation());
}