// Copyright 2024 Ivanov Mike
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>

#include "core/perf/include/perf.hpp"
#include "mpi/ivanov_m_gauss_horizontal/include/ops_mpi.hpp"

namespace ivanov_m_gauss_horizontal_mpi {
std::vector<double> GenSolution(int size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> generator(-2, 2);
  std::vector<double> solution(size, 0);
  for (int i = 0; i < size; i++) {
    solution[i] = static_cast<double>(generator(gen));  // generating random coefficient in range [-2, 2]
  }
  return solution;
}

std::vector<double> GenMatrix(const std::vector<double> &solution) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> generator(-2, 2);
  std::vector<double> extended_matrix;
  int size = static_cast<int>(solution.size());

  // generate identity matrix
  for (int row = 0; row < size; row++) {
    for (int column = 0; column < size; column++) {
      if (row == column) {
        extended_matrix.push_back(1);
      } else {
        extended_matrix.push_back(0);
      }
    }
    extended_matrix.push_back(solution[row]);
  }

  // saturation left triangle
  for (int row = 1; row < size; row++) {
    for (int column = 0; column < row; column++) {
      extended_matrix[get_linear_index(row, column, size + 1)] +=
          extended_matrix[get_linear_index(row - 1, column, size + 1)];
    }
    extended_matrix[get_linear_index(row, size, size + 1)] +=
        extended_matrix[get_linear_index(row - 1, size, size + 1)];
  }

  // saturation of matrix by random numbers
  for (int row = size - 1; row > 0; row--) {
    int coef = generator(gen);
    for (int column = 0; column < size + 1; column++) {
      extended_matrix[get_linear_index(row - 1, column, size + 1)] +=
          coef * extended_matrix[get_linear_index(row, column, size + 1)];
    }
  }

  // saturation of matrix by random numbers
  for (int row = 0; row < size - 1; row++) {
    int coef = generator(gen);
    for (int column = 0; column < size + 1; column++) {
      extended_matrix[get_linear_index(row + 1, column, size + 1)] +=
          coef * extended_matrix[get_linear_index(row, column, size + 1)];
    }
  }

  return extended_matrix;
}
}  // namespace ivanov_m_gauss_horizontal_mpi

TEST(ivanov_m_gauss_horizontal_mpi_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;
  int n = 1000;
  std::vector<double> ans = ivanov_m_gauss_horizontal_mpi::GenSolution(n);
  std::vector<double> matrix = ivanov_m_gauss_horizontal_mpi::GenMatrix(ans);
  std::vector<double> out_par(n, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    taskDataPar->outputs_count.emplace_back(out_par.size());
  }

  auto testMpiTaskParallel = std::make_shared<ivanov_m_gauss_horizontal_mpi::TestMPITaskParallel>(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

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
    for (int i = 0; i < n; i++) {
      EXPECT_NEAR(ans[i], out_par[i], 1e-3);
    }
  }
}

TEST(ivanov_m_gauss_horizontal_mpi_perf_test, test_task_run) {
  boost::mpi::communicator world;
  int n = 1000;
  std::vector<double> ans = ivanov_m_gauss_horizontal_mpi::GenSolution(n);
  std::vector<double> matrix = ivanov_m_gauss_horizontal_mpi::GenMatrix(ans);
  std::vector<double> out_par(n, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.emplace_back(matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&n));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    taskDataPar->outputs_count.emplace_back(out_par.size());
  }

  auto testMpiTaskParallel = std::make_shared<ivanov_m_gauss_horizontal_mpi::TestMPITaskParallel>(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

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
    for (int i = 0; i < n; i++) {
      EXPECT_NEAR(ans[i], out_par[i], 1e-3);
    }
  }
}