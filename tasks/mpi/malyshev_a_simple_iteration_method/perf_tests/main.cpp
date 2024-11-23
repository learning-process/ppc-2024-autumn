// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/malyshev_a_simple_iteration_method/include/ops_mpi.hpp"

TEST(malyshev_a_simple_iteration_method_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  std::vector<double> A;
  std::vector<double> B;
  std::vector<double> X;
  double eps = 1e-4;
  if (world.rank() == 0) {
    // Create data
    const int size = 300;
    X.resize(size);
    malyshev_a_simple_iteration_method_mpi::getRandomData(size, A, B);
    std::vector<double> X0(size, 0);

    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&eps));
    taskDataPar->inputs_count.emplace_back(X.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
    taskDataPar->outputs_count.emplace_back(X.size());
  }

  auto testMpiTaskParallel = std::make_shared<malyshev_a_simple_iteration_method_mpi::TestMPITaskParallel>(taskDataPar);
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
  }
}

TEST(malyshev_a_simple_iteration_method_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  std::vector<double> A;
  std::vector<double> B;
  std::vector<double> X;
  double eps = 1e-4;
  if (world.rank() == 0) {
    // Create data
    const int size = 300;
    X.resize(size);
    malyshev_a_simple_iteration_method_mpi::getRandomData(size, A, B);
    std::vector<double> X0(size, 0);

    // Create TaskData
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&eps));
    taskDataPar->inputs_count.emplace_back(X.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
    taskDataPar->outputs_count.emplace_back(X.size());
  }

  auto testMpiTaskParallel = std::make_shared<malyshev_a_simple_iteration_method_mpi::TestMPITaskParallel>(taskDataPar);
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
  }
}
