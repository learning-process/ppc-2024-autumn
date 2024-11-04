// Copyright 2023 Nasedkin Egor
#include <gtest/gtest.h>
#include <boost/mpi/timer.hpp>
#include <vector>
#include "core/perf/include/perf.hpp"
#include "mpi/nasedkin_e_matrix_column_max_value/include/ops_mpi.hpp"

TEST(mpi_matrix_column_max_value_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int> global_max(3, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int rows, cols;
  if (world.rank() == 0) {
    rows = 120, cols = 3;
    global_matrix = std::vector<int>(rows * cols, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  auto matrixColumnMaxMPI = std::make_shared<nasedkin_e_matrix_column_max_value_mpi::MatrixColumnMaxMPI>(taskDataPar);
  ASSERT_EQ(matrixColumnMaxMPI->validation(), true);
  matrixColumnMaxMPI->pre_processing();
  matrixColumnMaxMPI->run();
  matrixColumnMaxMPI->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(matrixColumnMaxMPI);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    std::vector<int> reference_max(3, 1);
    ASSERT_EQ(reference_max, global_max);
  }
}

TEST(mpi_matrix_column_max_value_perf_test, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int> global_max(3, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int rows, cols;
  if (world.rank() == 0) {
    rows = 120, cols = 3;
    global_matrix = std::vector<int>(rows * cols, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  auto matrixColumnMaxMPI = std::make_shared<nasedkin_e_matrix_column_max_value_mpi::MatrixColumnMaxMPI>(taskDataPar);
  ASSERT_EQ(matrixColumnMaxMPI->validation(), true);
  matrixColumnMaxMPI->pre_processing();
  matrixColumnMaxMPI->run();
  matrixColumnMaxMPI->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(matrixColumnMaxMPI);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    std::vector<int> reference_max(3, 1);
    ASSERT_EQ(reference_max, global_max);
  }
}

int main(int argc, char** argv) {
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (world.rank() != 0) {
    delete listeners.Release(listeners.default_result_printer());
  }
  return RUN_ALL_TESTS();
}