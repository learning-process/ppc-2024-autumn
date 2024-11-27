#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/afanasyev_a_jacobi_method/include/ops_mpi.hpp"

std::pair<std::vector<double>, std::vector<double>> get_random_diagonally_matrix(int matrix_size,
                                                                                 double min_val = -10.0,
                                                                                 double max_val = 10.0) {
  std::vector<double> matrix(matrix_size * matrix_size);
  std::vector<double> rhs(matrix_size);

  std::random_device random_device;
  std::mt19937 generator(random_device());
  std::uniform_real_distribution<> distribution(min_val, max_val);

  for (int row = 0; row < matrix_size; ++row) {
    double off_diagonal_sum = 0.0;

    for (int col = 0; col < matrix_size; ++col) {
      if (row != col) {
        matrix[row * matrix_size + col] = distribution(generator);
        off_diagonal_sum += std::abs(matrix[row * matrix_size + col]);
      }
    }

    matrix[row * matrix_size + row] = off_diagonal_sum + std::abs(distribution(generator)) + 1.0;

    rhs[row] = distribution(generator);
  }

  return {matrix, rhs};
}

TEST(afanasyev_a_jacobi_method, test_pipeline_run) {
  boost::mpi::communicator world;

  const size_t matrix_size = 512;
  auto [matrix_A, rsh] = get_random_diagonally_matrix(matrix_size);

  std::vector<size_t> in_size(1, matrix_size);
  std::vector<double> out(matrix_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
    taskDataPar->inputs_count.emplace_back(in_size.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_A.data()));
    taskDataPar->inputs_count.emplace_back(matrix_A.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(rsh.data()));
    taskDataPar->inputs_count.emplace_back(rsh.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto jacobiTaskParallel = std::make_shared<afanasyev_a_jacobi_method_mpi::JacobiMethodParallelTask>(taskDataPar);
  ASSERT_EQ(jacobiTaskParallel->validation(), true);
  jacobiTaskParallel->pre_processing();
  jacobiTaskParallel->run();
  jacobiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(jacobiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(matrix_size, out.size());
  }
}

TEST(afanasyev_a_jacobi_method, test_task_run) {
  boost::mpi::communicator world;

  const size_t matrix_size = 512;
  auto [matrix_A, rsh] = get_random_diagonally_matrix(matrix_size);

  std::vector<size_t> in_size(1, matrix_size);
  std::vector<double> out(matrix_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
    taskDataPar->inputs_count.emplace_back(in_size.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix_A.data()));
    taskDataPar->inputs_count.emplace_back(matrix_A.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(rsh.data()));
    taskDataPar->inputs_count.emplace_back(rsh.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto jacobiTaskParallel = std::make_shared<afanasyev_a_jacobi_method_mpi::JacobiMethodParallelTask>(taskDataPar);
  ASSERT_EQ(jacobiTaskParallel->validation(), true);
  jacobiTaskParallel->pre_processing();
  jacobiTaskParallel->run();
  jacobiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(jacobiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(matrix_size, out.size());
  }
}