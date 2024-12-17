// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/malyshev_v_conjugate_gradient_method/include/ops_mpi.hpp"
boost::mpi::environment env;
TEST(malyshev_v_conjugate_gradient_method_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  size_t matrix_size = 4;
  std::vector<double> global_result(matrix_size, 0.0);
  std::vector<double> Matrix = {4.0, 1.0, 2.0, 1.0, 1.0, 3.0, 1.0, 2.0, 2.0, 1.0, 5.0, 1.0, 1.0, 2.0, 1.0, 4.0};
  std::vector<double> Values = {1.0, 2.0, 3.0, 4.0};
  double epsilon = 0.001;
  std::vector<double> expected_output = {0.2, 0.4, 0.6, 0.8};
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(Matrix.data()));
    taskDataPar->inputs_count.emplace_back(Matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(Values.data()));
    taskDataPar->inputs_count.emplace_back(Values.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto MPIConjugateGradientParallel =
      std::make_shared<malyshev_v_conjugate_gradient_method::MPIConjugateGradientParallel>(taskDataPar);
  ASSERT_EQ(MPIConjugateGradientParallel->validation(), true);
  MPIConjugateGradientParallel->pre_processing();
  MPIConjugateGradientParallel->run();
  MPIConjugateGradientParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(MPIConjugateGradientParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (unsigned int i = 0; i < 4; ++i) {
      ASSERT_NEAR(global_result[i], expected_output[i], 0.001) << "Mismatch at index " << i;
    }
  }
}

TEST(malyshev_v_conjugate_gradient_method_mpi, test_task_run) {
  boost::mpi::communicator world;
  size_t matrix_size = 4;
  std::vector<double> global_result(matrix_size, 0.0);
  std::vector<double> Matrix = {4.0, 1.0, 2.0, 1.0, 1.0, 3.0, 1.0, 2.0, 2.0, 1.0, 5.0, 1.0, 1.0, 2.0, 1.0, 4.0};
  std::vector<double> Values = {1.0, 2.0, 3.0, 4.0};
  double epsilon = 0.001;
  std::vector<double> expected_output = {0.2, 0.4, 0.6, 0.8};
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(Matrix.data()));
    taskDataPar->inputs_count.emplace_back(Matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(Values.data()));
    taskDataPar->inputs_count.emplace_back(Values.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto MPIConjugateGradientParallel =
      std::make_shared<malyshev_v_conjugate_gradient_method::MPIConjugateGradientParallel>(taskDataPar);
  ASSERT_EQ(MPIConjugateGradientParallel->validation(), true);
  MPIConjugateGradientParallel->pre_processing();
  MPIConjugateGradientParallel->run();
  MPIConjugateGradientParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(MPIConjugateGradientParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (unsigned int i = 0; i < 4; ++i) {
      ASSERT_NEAR(global_result[i], expected_output[i], 0.001) << "Mismatch at index " << i;
    }
  }
}