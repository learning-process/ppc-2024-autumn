#include <gtest/gtest.h>
#include <boost/mpi/timer.hpp>
#include <vector>
#include "core/perf/include/perf.hpp"
#include "mpi/sotskov_a_sum_element_matrix/include/ops_mpi.hpp"

TEST(sotskov_a_sum_element_matrix, test_pipeline_run) {
  boost::mpi::communicator world;
  int rows = 1000;
  int cols = 1000;
  std::vector<double> matrix(rows * cols, 1.0);
  double output = 0.0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(&rows));
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(&cols));
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(&output));
    taskDataPar->outputs_count.push_back(1);
  }

  auto testMpiTaskParallel = std::make_shared<sotskov_a_sum_element_matrix_mpi::TestMPITaskParallel>(taskDataPar);

  ASSERT_TRUE(testMpiTaskParallel->validation());
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

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    auto exact = static_cast<double>(rows * cols);
    EXPECT_NEAR(output, exact, 1e-4);
  }
}

TEST(sotskov_a_sum_element_matrix, test_task_run) {
  boost::mpi::communicator world;
  int rows = 10000;
  int cols = 10000;
  std::vector<double> matrix(rows * cols, 1.0);
  double output = 0.0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(matrix.data()));
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(&rows));
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(&cols));
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(&output));
    taskDataPar->outputs_count.push_back(1);
  }

  auto testMpiTaskParallel = std::make_shared<sotskov_a_sum_element_matrix_mpi::TestMPITaskParallel>(taskDataPar);

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
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    auto exact = static_cast<double>(rows * cols);
    EXPECT_NEAR(output, exact, 1e-4);
  }
}