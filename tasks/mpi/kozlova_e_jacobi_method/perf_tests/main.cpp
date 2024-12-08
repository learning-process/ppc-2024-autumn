#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/kozlova_e_jacobi_method/include/ops_mpi.hpp"

namespace kozlova_e_generate_matrix {
void generate_diag_dominant_matrix(int N, std::vector<double> &A, std::vector<double> &B) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  for (int i = 0; i < N; ++i) {
    double diagonal_value = 0.0;
    double off_diagonal_sum = 0.0;
    for (int j = 0; j < N; ++j) {
      if (i != j) {
        A[i * N + j] = dis(gen);
        off_diagonal_sum += std::abs(A[i * N + j]);
      }
    }

    diagonal_value = off_diagonal_sum * 10.0;
    A[i * N + i] = diagonal_value;

    B[i] = diagonal_value * dis(gen);
  }
}
}  // namespace kozlova_e_generate_matrix

TEST(kozlova_e_jacobi_method_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  const int N = 900;
  std::vector<double> A(N * N, 0.0);
  std::vector<double> B(N, 0.0);
  kozlova_e_generate_matrix::generate_diag_dominant_matrix(N, A, B);
  std::vector<double> X(N, 0.0);
  double epsilon = 1e-6;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs_count.emplace_back(N);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
    taskDataPar->outputs_count.emplace_back(X.size());
  }

  auto testMpiTaskParallel = std::make_shared<kozlova_e_jacobi_method_mpi::MethodJacobiMPI>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  std::vector<double> Ax(N, 0.0);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      Ax[i] += A[i * N + j] * X[j];
    }
  }
  std::vector<double> res(N, 0);
  for (int i = 0; i < N; ++i) {
    res[i] = abs(Ax[i] - B[i]);
  }

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (int i = 0; i < N; i++) ASSERT_LT(res[i], 1e-4);
  }
}

TEST(kozlova_e_jacobi_method_mpi, test_task_run) {
  boost::mpi::communicator world;
  const int N = 1050;
  std::vector<double> A(N * N, 0.0);
  std::vector<double> B(N, 0.0);
  kozlova_e_generate_matrix::generate_diag_dominant_matrix(N, A, B);
  std::vector<double> X(N, 0.0);
  double epsilon = 1e-6;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataPar->inputs_count.emplace_back(N);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
    taskDataPar->outputs_count.emplace_back(X.size());
  }

  auto testMpiTaskParallel = std::make_shared<kozlova_e_jacobi_method_mpi::MethodJacobiMPI>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  std::vector<double> Ax(N, 0.0);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      Ax[i] += A[i * N + j] * X[j];
    }
  }
  std::vector<double> res(N, 0);
  for (int i = 0; i < N; ++i) {
    res[i] = abs(Ax[i] - B[i]);
  }

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (int i = 0; i < N; i++) ASSERT_LT(res[i], 1e-4);
  }
}
