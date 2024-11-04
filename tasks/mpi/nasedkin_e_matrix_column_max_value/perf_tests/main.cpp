#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/nasedkin_e_matrix_column_max_value/include/ops_mpi.hpp"

TEST(nasedkin_e_matrix_column_max_value_perf_test, test_pipeline_run) {
  int numRows = 1000;
  int numCols = 4000;
  boost::mpi::communicator world;
  std::vector<int> matrix;
  std::vector<int32_t> maxVecMPI(numCols, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = std::vector<int>(numRows * numCols, 1);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataParallel->inputs_count.emplace_back(matrix.size());
    taskDataParallel->inputs_count.emplace_back(numCols);
    taskDataParallel->inputs_count.emplace_back(numRows);
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(maxVecMPI.data()));
    taskDataParallel->outputs_count.emplace_back(maxVecMPI.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<nasedkin_e_matrix_column_max_value_mpi::TestMPITaskParallel>(taskDataParallel);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer currentTimer;
  perfAttr->current_timer = [&] { return currentTimer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (unsigned i = 0; i < maxVecMPI.size(); i++) {
      EXPECT_EQ(1, maxVecMPI[0]);
    }
  }
}

TEST(nasedkin_e_matrix_column_max_value_perf_test, test_task_run) {
  int numRows = 1000;
  int numCols = 4000;
  boost::mpi::communicator world;
  std::vector<int> matrix;
  std::vector<int32_t> maxVecMPI(numCols, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    matrix = std::vector<int>(numRows * numCols, 1);
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataParallel->inputs_count.emplace_back(matrix.size());
    taskDataParallel->inputs_count.emplace_back(numCols);
    taskDataParallel->inputs_count.emplace_back(numRows);
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(maxVecMPI.data()));
    taskDataParallel->outputs_count.emplace_back(maxVecMPI.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<nasedkin_e_matrix_column_max_value_mpi::TestMPITaskParallel>(taskDataParallel);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer currentTimer;
  perfAttr->current_timer = [&] { return currentTimer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (unsigned i = 0; i < maxVecMPI.size(); i++) {
      EXPECT_EQ(1, maxVecMPI[0]);
    }
  }
}