// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/lysov_i_simple_iteration_method/include/ops_mpi.hpp"

void generate_diagonally_dominant_matrix(int size, std::vector<double> &matrix, std::vector<double> &b) {
  matrix.resize(size * size);
  b.resize(size);

  for (int i = 0; i < size; ++i) {
    double sum = 0.0;
    matrix[i * size + i] = 2.0 * size;

    for (int j = 0; j < size; ++j) {
      if (i != j) {
        matrix[i * size + j] = 1.0;
        sum += std::abs(matrix[i * size + j]);
      }
    }

    b[i] = matrix[i * size + i] * 1.0 + sum;
  }
}

TEST(lysov_i_simple_iteration_method_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  const int input_size = 800;
  std::vector<double> matrix;
  std::vector<double> g;
  std::vector<double> expected_solution;
  std::vector<double> x;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    generate_diagonally_dominant_matrix(input_size, matrix, g);
    x.resize(input_size, 0.0);
    expected_solution.resize(input_size, 1.0);
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.push_back(input_size);
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t *>(g.data()));
    taskDataPar->inputs_count.push_back(input_size);
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
    taskDataPar->outputs_count.push_back(input_size);
  }

  auto testMpiTaskParallel = std::make_shared<lysov_i_simple_iteration_method_mpi::SlaeIterationTaskMPI>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  ASSERT_TRUE(testMpiTaskParallel->run());
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (int i = 0; i < static_cast<int>(x.size()); i++) ASSERT_NEAR(x[i], expected_solution[i], 1e-4);
  }
}

TEST(lysov_i_simple_iteration_method_mpi, test_task_run) {
  boost::mpi::communicator world;
  const int input_size = 800;
  std::vector<double> matrix;
  std::vector<double> g;
  std::vector<double> expected_solution;
  std::vector<double> x;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    generate_diagonally_dominant_matrix(input_size, matrix, g);
    x.resize(input_size, 0.0);
    expected_solution.resize(input_size, 1.0);
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t *>(matrix.data()));
    taskDataPar->inputs_count.push_back(input_size);
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t *>(g.data()));
    taskDataPar->inputs_count.push_back(input_size);
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
    taskDataPar->outputs_count.push_back(input_size);
  }

  auto testMpiTaskParallel = std::make_shared<lysov_i_simple_iteration_method_mpi::SlaeIterationTaskMPI>(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel->validation());
  testMpiTaskParallel->pre_processing();
  ASSERT_TRUE(testMpiTaskParallel->run());
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (int i = 0; i < static_cast<int>(x.size()); i++) ASSERT_NEAR(x[i], expected_solution[i], 1e-4);
  }
}
