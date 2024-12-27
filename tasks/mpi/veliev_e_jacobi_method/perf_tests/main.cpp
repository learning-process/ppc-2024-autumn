#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/veliev_e_jacobi_method/include/ops_mpi.hpp"

namespace veliev_e_generate_matrix {
void create_diag_dominant_system(int N, std::vector<double> &matrixA, std::vector<double> &rshB) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  for (int i = 0; i < N; ++i) {
    double diagonal_value = 0.0;
    double off_diagonal_sum = 0.0;
    for (int j = 0; j < N; ++j) {
      if (i != j) {
        matrixA[i * N + j] = dis(gen);
        off_diagonal_sum += std::abs(matrixA[i * N + j]);
      }
    }

    diagonal_value = off_diagonal_sum * 10.0;
    matrixA[i * N + i] = diagonal_value;

    rshB[i] = diagonal_value * dis(gen);
  }
}
}  // namespace veliev_e_generate_matrix

TEST(veliev_e_jacobi_method_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  const uint32_t matrixSize = 900;
  std::vector<double> matrix(matrixSize * matrixSize, 0.0);
  std::vector<double> rhs(matrixSize, 0.0);
  veliev_e_generate_matrix::create_diag_dominant_system(matrixSize, matrix, rhs);
  std::vector<double> solution(matrixSize, 0.0);
  double epsilon = 1e-6;

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataMPI->inputs.push_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataMPI->inputs.push_back(reinterpret_cast<uint8_t *>(rhs.data()));
    taskDataMPI->inputs.push_back(reinterpret_cast<uint8_t *>(solution.data()));
    taskDataMPI->inputs.push_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataMPI->inputs_count.push_back(matrixSize);
    taskDataMPI->outputs.push_back(reinterpret_cast<uint8_t *>(solution.data()));
    taskDataMPI->outputs_count.push_back(solution.size());
  }

  auto mpiTask = std::make_shared<veliev_e_jacobi_method_mpi::MethodJacobiMPI>(taskDataMPI);

  ASSERT_TRUE(mpiTask->validation());
  mpiTask->pre_processing();
  mpiTask->run();
  mpiTask->post_processing();

  std::vector<double> matrixVectorProduct(matrixSize, 0.0);
  for (uint32_t i = 0; i < matrixSize; ++i) {
    for (uint32_t j = 0; j < matrixSize; ++j) {
      matrixVectorProduct[i] += matrix[i * matrixSize + j] * solution[j];
    }
  }

  std::vector<double> result(matrixSize, 0);
  for (uint32_t i = 0; i < matrixSize; ++i) {
    result[i] = std::abs(matrixVectorProduct[i] - rhs[i]);
  }

  auto performanceAttributes = std::make_shared<ppc::core::PerfAttr>();
  performanceAttributes->num_running = 10;
  const boost::mpi::timer timer;
  performanceAttributes->current_timer = [&] { return timer.elapsed(); };
  auto performanceResults = std::make_shared<ppc::core::PerfResults>();
  auto performanceAnalyzer = std::make_shared<ppc::core::Perf>(mpiTask);
  performanceAnalyzer->pipeline_run(performanceAttributes, performanceResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(performanceResults);
    for (uint32_t i = 0; i < matrixSize; ++i) {
      ASSERT_LT(result[i], 1e-4);
    }
  }
}

TEST(veliev_e_jacobi_method_mpi, test_task_run) {
  boost::mpi::communicator world;

  const uint32_t matrixSize = 1050;
  std::vector<double> matrix(matrixSize * matrixSize, 0.0);
  std::vector<double> rhs(matrixSize, 0.0);
  veliev_e_generate_matrix::create_diag_dominant_system(matrixSize, matrix, rhs);
  std::vector<double> solution(matrixSize, 0.0);
  double epsilon = 1e-6;

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataMPI->inputs.push_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataMPI->inputs.push_back(reinterpret_cast<uint8_t *>(rhs.data()));
    taskDataMPI->inputs.push_back(reinterpret_cast<uint8_t *>(solution.data()));
    taskDataMPI->inputs.push_back(reinterpret_cast<uint8_t *>(&epsilon));
    taskDataMPI->inputs_count.push_back(matrixSize);
    taskDataMPI->outputs.push_back(reinterpret_cast<uint8_t *>(solution.data()));
    taskDataMPI->outputs_count.push_back(solution.size());
  }

  auto mpiTask = std::make_shared<veliev_e_jacobi_method_mpi::MethodJacobiMPI>(taskDataMPI);

  ASSERT_TRUE(mpiTask->validation());
  mpiTask->pre_processing();
  mpiTask->run();
  mpiTask->post_processing();

  std::vector<double> matrixVectorProduct(matrixSize, 0.0);
  for (uint32_t i = 0; i < matrixSize; ++i) {
    for (uint32_t j = 0; j < matrixSize; ++j) {
      matrixVectorProduct[i] += matrix[i * matrixSize + j] * solution[j];
    }
  }

  std::vector<double> result(matrixSize, 0);
  for (uint32_t i = 0; i < matrixSize; ++i) {
    result[i] = std::abs(matrixVectorProduct[i] - rhs[i]);
  }

  auto performanceAttributes = std::make_shared<ppc::core::PerfAttr>();
  performanceAttributes->num_running = 10;
  const boost::mpi::timer timer;
  performanceAttributes->current_timer = [&] { return timer.elapsed(); };
  auto performanceResults = std::make_shared<ppc::core::PerfResults>();
  auto performanceAnalyzer = std::make_shared<ppc::core::Perf>(mpiTask);
  performanceAnalyzer->task_run(performanceAttributes, performanceResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(performanceResults);
    for (uint32_t i = 0; i < matrixSize; ++i) {
      ASSERT_LT(result[i], 1e-4);
    }
  }
}
