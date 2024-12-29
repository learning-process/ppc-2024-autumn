// Golovkin Maksim Task#2

#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <string>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/golovkin_rowwise_matrix_partitioning/include/ops_mpi.hpp"

using namespace golovkin_rowwise_matrix_partitioning;
using ppc::core::Perf;
using ppc::core::TaskData;

namespace golovkin_rowwise_matrix_partitioning {

void get_random_matrix(double *matr, int rows, int cols) {
  if (rows <= 0 || cols <= 0) {
    throw std::logic_error("wrong matrix size");
  }
  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      matr[i * cols + j] = static_cast<double>(std::rand()) / RAND_MAX;
    }
  }
}

void sequential_matrix_multiplication(const double *A, const double *B, double *C, int rows_A, int cols_A, int cols_B) {
  for (int i = 0; i < rows_A; ++i) {
    for (int j = 0; j < cols_B; ++j) {
      C[i * cols_B + j] = 0.0;
      for (int k = 0; k < cols_A; ++k) {
        C[i * cols_B + j] += A[i * cols_A + k] * B[k * cols_B + j];
      }
    }
  }
}

}  // namespace golovkin_rowwise_matrix_partitioning

TEST(golovkin_rowwise_matrix_partitioning, test_pipeline_run) {
  boost::mpi::communicator world;
  double *A = nullptr;
  double *B = nullptr;
  double *result = nullptr;
  int rows_A = 700;
  int cols_A = 800;
  int rows_B = 800;
  int cols_B = 300;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.size() < 5 || world.rank() >= 4) {
    A = new double[rows_A * cols_A]();
    B = new double[rows_B * cols_B]();

    golovkin_rowwise_matrix_partitioning::get_random_matrix(A, rows_A, cols_A);
    golovkin_rowwise_matrix_partitioning::get_random_matrix(B, rows_B, cols_B);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B));

    taskDataPar->inputs_count.emplace_back(rows_A);
    taskDataPar->inputs_count.emplace_back(cols_A);
    taskDataPar->inputs_count.emplace_back(rows_B);
    taskDataPar->inputs_count.emplace_back(cols_B);

    result = new double[rows_A * cols_B];
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result));
    taskDataPar->outputs_count.emplace_back(rows_A);
    taskDataPar->outputs_count.emplace_back(cols_B);
  }

  auto testMpiTaskParallel =
      std::make_shared<golovkin_rowwise_matrix_partitioning::MPIMatrixMultiplicationTask>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0 && world.size() < 5) {
    ppc::core::Perf::print_perf_statistic(perfResults);

    std::vector<double> expected_res(rows_A * cols_B, 0.0);
    golovkin_rowwise_matrix_partitioning::sequential_matrix_multiplication(A, B, expected_res.data(), rows_A, cols_A,
                                                                           cols_B);

    for (int i = 0; i < rows_A * cols_B; i++) {
      ASSERT_NEAR(expected_res[i], result[i], 1e-6) << "Mismatch at index " << i;
    }

    delete[] result;
    delete[] A;
    delete[] B;
  }
}

TEST(golovkin_rowwise_matrix_partitioning, test_task_run) {
  boost::mpi::communicator world;
  double *A = nullptr;
  double *B = nullptr;
  double *result = nullptr;
  int rows_A = 700;
  int cols_A = 800;
  int rows_B = 800;
  int cols_B = 300;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.size() < 5 || world.rank() >= 4) {
    A = new double[rows_A * cols_A]();
    B = new double[rows_B * cols_B]();

    golovkin_rowwise_matrix_partitioning::get_random_matrix(A, rows_A, cols_A);
    golovkin_rowwise_matrix_partitioning::get_random_matrix(B, rows_B, cols_B);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B));

    taskDataPar->inputs_count.emplace_back(rows_A);
    taskDataPar->inputs_count.emplace_back(cols_A);
    taskDataPar->inputs_count.emplace_back(rows_B);
    taskDataPar->inputs_count.emplace_back(cols_B);

    result = new double[rows_A * cols_B];
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(result));
    taskDataPar->outputs_count.emplace_back(rows_A);
    taskDataPar->outputs_count.emplace_back(cols_B);
  }

  auto testMpiTaskParallel =
      std::make_shared<golovkin_rowwise_matrix_partitioning::MPIMatrixMultiplicationTask>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0 && world.size() < 5) {
    ppc::core::Perf::print_perf_statistic(perfResults);

    std::vector<double> expected_res(rows_A * cols_B, 0.0);
    golovkin_rowwise_matrix_partitioning::sequential_matrix_multiplication(A, B, expected_res.data(), rows_A, cols_A,
                                                                           cols_B);

    for (int i = 0; i < rows_A * cols_B; i++) {
      ASSERT_NEAR(expected_res[i], result[i], 1e-6) << "Mismatch at index " << i;
    }

    delete[] result;
    delete[] A;
    delete[] B;
  }
}